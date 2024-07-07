import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from peft import AutoPeftModelForCausalLM

import re

default_path = "/path/to/chatlaw"
generation_config = GenerationConfig(
        do_sample=False,
        temperature=1,
        num_beams=1
    )

def get_model(base_model_path=default_path):
    tokenizer = LlamaTokenizer.from_pretrained(default_path)
    model = LlamaForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    model.eval()
    return model, tokenizer

def get_lora_model(base_model_path):
    tokenizer = LlamaTokenizer.from_pretrained(default_path)
    model = AutoPeftModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer

def get_response(model, tokenizer, question, max_new_tokens=50):
    prompt = f"Consult:\n{question}\nResponse:\n"
    with torch.autocast("cuda"):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].to(model.device)
        with torch.no_grad():
            generation_output = model.generate(
                inputs['input_ids'],
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1,
            )
        output = tokenizer.decode(generation_output.sequences[0])
        if search_result := re.search("Response\s*:\s*([\s\S]+?)</s>", output):
            response = search_result.group(1)
        else:
            response = output.split("Response:")[1].split("</s>")[0].strip()
        response = response.strip()
    return response

def get_response_with_answer_prefix(model, tokenizer, question, answer_prefix, greedy=True, max_new_tokens=200):
    prompt = f"Consult:\n{question}\nResponse:\n{answer_prefix} "
    with torch.autocast("cuda"):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].to(model.device)
        with torch.no_grad():
            if greedy:
                new_generation_config = generation_config
            else:
                new_generation_config = GenerationConfig(
                    do_sample=True,
                    temperature=1.2,
                    top_p=0.95
                )
            generation_output = model.generate(
                inputs['input_ids'],
                generation_config=new_generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1,
            )
        output = tokenizer.decode(generation_output.sequences[0][len(inputs['input_ids'][0]):])
    return output.replace("</s>", "").strip()