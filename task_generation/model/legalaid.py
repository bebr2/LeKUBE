from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig
import torch
from peft import AutoPeftModelForCausalLM

default_path = "/path/to/legalaid"

generation_config = GenerationConfig(
    do_sample=False,
    temperature=1,
    num_beams=1
)

def get_prompt(q):
    return f'以下是一项描述任务的指示，请撰写一个回答，恰当地完成该任务。指示：{q}\n回答：'

def get_model(model_path=default_path):
    tokenizer = AutoTokenizer.from_pretrained(default_path, padding_side="left", trust_remote_code=True)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id # set as the <unk> token
    if tokenizer.pad_token_id == 64000:
        tokenizer.pad_token_id = 0 # for baichuan model (need fix)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage=True, device_map="auto")
    model.eval()
    return model, tokenizer


def get_lora_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(default_path, padding_side="left", trust_remote_code=True)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id # set as the <unk> token
    if tokenizer.pad_token_id == 64000:
        tokenizer.pad_token_id = 0 # for baichuan model (need fix)
    # config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoPeftModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    return model, tokenizer

def get_response(model, tokenizer, question, max_new_tokens=50):
    prompt = get_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs['input_ids'] = inputs['input_ids'].to(model.device)
    _, length = inputs['input_ids'].shape
    generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                max_new_tokens=max_new_tokens,
            )
    output = tokenizer.decode(generation_output.sequences[0, length:]).replace("</s>", "").strip()
    return output

def get_response_with_answer_prefix(model, tokenizer, question, answer_prefix, greedy=True, max_new_tokens=200):
    prompt = get_prompt(question) + answer_prefix
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs['input_ids'] = inputs['input_ids'].to(model.device)
    _, length = inputs['input_ids'].shape
    if greedy:
        new_generation_config = generation_config
    else:
        new_generation_config = GenerationConfig(
            do_sample=True,
            temperature=1.2,
            top_p=0.95
        )
    generation_output = model.generate(
                **inputs,
                generation_config=new_generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                max_new_tokens=max_new_tokens,
            )
    output = tokenizer.decode(generation_output.sequences[0, length:]).replace("</s>", "").strip()
    return output