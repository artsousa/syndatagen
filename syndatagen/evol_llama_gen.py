import ast
import json
import torch
import random
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from awq import AutoAWQForCausalLM
from transformers import pipeline
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import GenerationConfig, TextStreamer

from utils.evol_utils import createConstraintsPrompt, createDeepenPrompt, createConcretizingPrompt, createReasoningPrompt
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def generate(text, model, tokenizer, context:str = None, max_tokens: int = 1024, skip_special = True, debug = False):
    messages = [{"role": "sistema", "content": "Você é um assistente útil que ira receber uma instrução e fará apenas o que está sendo pedido. Se o #Contexto# for passado, sempre o use para responder o usuário. O #Contexto# não deve ser mencionado na resposta. Não extenda suas respostas explicando o que foi considerado para responder."},
            {"role": "usuário", "content": text}]

    messages[1]['content'] += f"\r\n#Contexto#:\r\n{context}" if context else ''

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt += "<|start_header_id|>assistente<|end_header_id|>\n" + "#Prompt Reescrito#:\r\n" if not context else '' 
    
    if debug: 
        print(f"\n\n ############ prompt: {prompt} \n\n ")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    generation_output = model.generate(
        input_ids=input_ids,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_tokens,
        early_stopping=False,
        temperature=0.75,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=terminators,
        top_k=40,
        #min_p=0.05,
        top_p=0.95,
        repetition_penalty=1.2,
        # eos_token_id=[32000, 2],
        # typical_p=0.2,
    )

    output = generation_output.sequences[0]
    output = tokenizer.decode(output[len(input_ids[0]):], skip_special_tokens=skip_special)
        
    return output


if __name__ == '__main__':
    
    n = 1000
    samples = []
    ds = DatasetDict()
    output_file = 'results/result_llama8b_v4.csv'
    
    ## local file load and save
    savesamples_csv = False
    load_from_file = True
    
    pointer = DatasetDict()
    if not load_from_file:
        print(f"## loading dataset...")
        pointer = load_dataset('unicamp-dl/mmarco', 'portuguese', streaming=True, trust_remote_code=True)
        for idx, sample in enumerate(pointer['train']):
            samples.append(sample)
            if idx == n - 1: break

        if savesamples_csv:
            print(f"saving first {n} samples into data/first_{n}_samples.csv")
            df_samples = pd.DataFrame({'samples': samples})
            df_samples.to_csv(f'data/first_{n}_samples.csv', index=False)
    else:
        samples_file = pd.read_csv(f'data/first_{n}_samples.csv')
        for p in samples_file['samples']:  
            samples.append(ast.literal_eval(p))
    pointer['train'] = Dataset.from_list(samples)
    
    device = 'cuda'
    model_list = ['adalbertojunior/Llama-3-8B-Instruct-Portuguese-v0.4']
    
    ## save interval
    save_step = 50
    first_only = False

    answers = pd.DataFrame()
    for model_name in model_list:
        ## load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left', cache_dir='/home/main/.tmp/')
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
        ## load model awq
        #model = AutoAWQForCausalLM.from_quantized(model_name, device_map='balanced', trust_remote_code=True, 
        #                                            fuse_layers=True, safetensors=True, cache_dir='/home/main/.tmp/')
        
        ## load model normal 
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16)

        model = model.to(device)

        evol_objs = []
        for i in tqdm(range(200)) :
            cur_obj = pointer['train'][i]
            negative = cur_obj['negative']
            instruction = cur_obj['query'].strip()

            ## define prompt templates
            evol_prompts = []
            evol_prompts.append({'constraint': createConstraintsPrompt(instruction)})
            evol_prompts.append({'deepeen': createDeepenPrompt(instruction)})
            evol_prompts.append({'concretizing': createConcretizingPrompt(instruction)})
            evol_prompts.append({'reasoning': createReasoningPrompt(instruction)})

            line = {'model': model_name, 'query': instruction, 'negative': negative}
            for ins in evol_prompts:
                # print(f"\n\n\nselected: {selected_evol_prompt}")
                evol_instruction = generate(ins[list(ins.keys())[0]], model, tokenizer, debug=False)
                # print(f"\n\n########### instruction: {evol_instruction}")

                answer = generate(evol_instruction, model, tokenizer, context=cur_obj['positive'].strip(), debug=False, max_tokens=2048)
                # print(f"\n\n ########## positive: {answer}")
                line[list(ins.keys())[0]] = str({'instruction': evol_instruction, 'positive': answer})
            
            answers = pd.concat([answers, pd.DataFrame(line, index=[0])], ignore_index=True)
           
            ## save answers 
            if i % save_step == 0 and i:
                answers.to_csv(output_file, index=False)
                if first_only: break 
            




