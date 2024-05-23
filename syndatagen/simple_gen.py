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


def generate(text, model, tokenizer, context:str = None, max_tokens: int = 1024, skip_special = True, debug = False):
    messages = [{"role": "sistema", "content": "Você é um assistente que ira receber uma instrução e fará apenas o que está sendo pedido. Se o #Contexto# for passado, sempre o use para responder o usuário. O #Contexto# não deve ser mencionado na resposta."},
            {"role": "usuário", "content": text}]

    # messages[1]['content'] += f"\r\n#Contexto#:\r\n{context}" if context else ''

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # prompt += "<|im_start|>assistente\n" + "#Prompt Reescrito#:\r\n" if not context else '' 
    
    ## for llama-3
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    if debug: 
        print(f"\n\n ############ prompt: {prompt} \n\n ")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    generation_output = model.generate(
        input_ids=input_ids,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_tokens,
        early_stopping=True,
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
    output_file = 'results/result.csv'
    
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
    
    # print('query: ' + pointer['train'][0]['query'])
    device = 'cuda'
    ## model list so far: 'TheBloke/Mistral-7B-OpenOrca-AWQ'
    model_list = ['casperhansen/llama-3-8b-instruct-awq']
 
    for model_name in model_list:
        ## load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left', cache_dir='/home/main/.tmp/')
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
        ## load model
        model = AutoAWQForCausalLM.from_quantized(model_name, device_map='balanced', trust_remote_code=True, 
                                                    fuse_layers=True, safetensors=True, cache_dir='/home/main/.tmp/')
        model = model.to(device)

        answer = generate( pointer['train'][0]['query'], model, tokenizer, context='', skip_special=False, debug=True, max_tokens=1024) 
        
        print(f"######## answer: \r\n {answer}")
    


