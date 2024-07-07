from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import torch



default_path = "/path/to/baichuan"

def get_model(model_path=default_path):
    tokenizer = AutoTokenizer.from_pretrained(default_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    return model, tokenizer
        
        
def get_response(model, tokenizer, question, max_new_tokens=50):
    model.generation_config = GenerationConfig(
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        user_token_id=195,
        assistant_token_id=196,
        max_new_tokens=max_new_tokens,
        temperature=1,
        num_beams=1,
        do_sample=False,
    )
    message = [{"role": "user", "content": question}]
    response = model.chat(tokenizer, message, generation_config=model.generation_config)
    response = response.strip()
    return response

user_tokens=[195]
assistant_tokens=[196]

def get_response_with_answer_prefix(model, tokenizer, question, answer_prefix, greedy=True, max_new_tokens=200):
    input_ids = []
    example = {
        "input": question,
        "output": answer_prefix
    }
    for key in ["input", "output"]:
        from_ = key
        value = example[key]
        value_ids = tokenizer.encode(value)

        if from_ == "input":
            input_ids += user_tokens + value_ids
        else:
            input_ids += assistant_tokens + value_ids
    if greedy:
        model.generation_config = GenerationConfig(
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            user_token_id=195,
            assistant_token_id=196,
            max_new_tokens=max_new_tokens,
            temperature=1,
            num_beams=1,
            do_sample=False,
        )
    else:
        model.generation_config = GenerationConfig(
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            user_token_id=195,
            assistant_token_id=196,
            max_new_tokens=max_new_tokens,
            temperature=1.2,
            top_p=0.95,
            do_sample=True,
        )
    generation_output = model.generate(
                input_ids = torch.tensor([input_ids]).to(model.device),
                generation_config=model.generation_config,
            )
    # print(generation_output)
    output = tokenizer.decode(generation_output[0][len(input_ids):])
    return output.replace("</s>", "").strip()