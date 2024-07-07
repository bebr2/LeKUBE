from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import torch
from peft import AutoPeftModelForCausalLM

default_path = "/path/to/chatglm"

def get_model(model_path=default_path):
    tokenizer = AutoTokenizer.from_pretrained(default_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    model.eval()
    return model, tokenizer

def get_lora_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(default_path, trust_remote_code=True)
    model = AutoPeftModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    return model, tokenizer
        
        
def get_response(model, tokenizer, question, max_new_tokens=50):
    message = [{"role": "user", "content": question}]
    response, history = model.chat(tokenizer, question, history=[], num_beams=1, do_sample=False, max_new_tokens=max_new_tokens)
    response = response.strip()
    return response

def get_response_with_answer_prefix(model, tokenizer, question, answer_prefix, greedy=True, max_new_tokens=200):
    input_ids = []
    input_ids = [
            tokenizer.get_command('[gMASK]'),
            tokenizer.get_command('sop'),
        ]
    example = {
        "input": question,
        "output": answer_prefix
    }
    eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"),
                        tokenizer.get_command("<|observation|>")]
    for key in ["input", "output"]:

        if key == "input":
            role = "user"
        else:
            role = "assistant"

        new_input_ids = tokenizer.build_single_message(
            role, '', example[key]
        )

        input_ids += new_input_ids
    if greedy:
        model.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=1,
            num_beams=1,
            do_sample=False,
        )
    else:
        model.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=1.2,
            top_p=0.95,
            do_sample=True,
        )
    generation_output = model.generate(
                input_ids = torch.tensor([input_ids]).to(model.device),
                generation_config=model.generation_config,
                eos_token_id=eos_token_id
            )
    # print(generation_output)
    output = tokenizer.decode(generation_output[0][len(input_ids):])
    return output.replace("</s>", "").strip()