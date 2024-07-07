import sys

import os

sys.path.append(sys.argv[2])


tag = sys.argv[1]
os.environ['CUDA_VISIBLE_DEVICES']='2,3,4,5'
import easyeditor
from model import baichuanchat as LLM
from utils import LLMprompt

#----------------------------#
max_len = 300
#----------------------------#

import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PreTrainedModel
from easyeditor import ROMEHyperParams, BaseEditor, KNHyperParams


if tag == "rome":
    EDITHYPERPARAMS = ROMEHyperParams
elif tag == "kn":
    EDITHYPERPARAMS = KNHyperParams

hparams = EDITHYPERPARAMS.from_hparams(f"./modeledit_hparams/{tag.upper()}/baichuanchat13b.yaml")

model_path = "/path/to/baichuan_ckpt"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

data_path = "../data/LawAdd.json"

# 任务：
from model import baichuanchat as LLM

model_name = "baichuanchat"
law_qa_path = "../data/LawQA.json"
law_dis_path = "../data/LawDiscri.json"
law_not_dis_path = "../data/LawDiscriNOT.json"
task_root = {
    "recall": "./result/recall",
    "recite": "./result/recite",
    "scenario_mcq": "./result/scenario_mcq_after",
    "discri": "./result/discri",
    "notdiscri": "./result/notdiscri",
    "notrecall": "./result/notrecall",
    "notrecite": "./result/notrecite",
}

old_law_data = json.load(open("../data/old_law_all.json", encoding='utf-8'))

data = json.load(open(data_path, encoding='utf-8'))
prompts = []
ground_truth = []
target_new = []
subject = []

for k in data:
    prompts.append(f"<reserved_106>{k}的内容是？<reserved_107>")
    target_new.append(f'{data[k][1].replace("**", "").strip()}</s>')
    ground_truth.append(f'{data[k][0].strip()}</s>')
    subject.append(k)


## Construct Language Model Editor
editor = BaseEditor.from_hparams(hparams)

metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=ground_truth if tag != "kn" else None,
    target_new=target_new,
    subject=subject,
    keep_original_weight=False
)

print(metrics)



# recall

output_path = f"{task_root['recall']}/{model_name}/result_{tag}.json" 
answers = {}

for i, law_name in tqdm(enumerate(data)):
    new_law = data[law_name][1].replace("**", "").strip()
    inputs = tokenizer(law_name, return_tensors='pt').input_ids
    _, length = inputs.shape
    
    question = LLMprompt.get_recall_prompt(new_law, 0, [f"{law_name}：{new_law}"])
        
    if i == 0:
        print(question)

    answer = LLM.get_response(edited_model, tokenizer, question, int(length*3))
    answers[law_name] = {
        "answer": answer,
        "ground_truth": law_name
    }
    
with open(output_path, "w+", encoding='utf-8') as f:
    json.dump(answers, f, indent=4, ensure_ascii=False)
    
# recite

output_path = f"{task_root['recite']}/{model_name}/result_{tag}.json"

answers = {}

for i, law_name in tqdm(enumerate(data)):
    new_law = data[law_name][1].replace("**", "").strip()
    inputs = tokenizer(new_law, return_tensors='pt').input_ids
    _, length = inputs.shape
    
    
    question = LLMprompt.get_recite_prompt(law_name, 0, [f"{law_name}：{new_law}"])
    
    if i == 0:
        print(question)
        
    answer = LLM.get_response(edited_model, tokenizer, question, int(length*2))
    answers[law_name] = {
        "answer": answer,
        "ground_truth": new_law
    }
    
with open(output_path, "w+", encoding='utf-8') as f:
    json.dump(answers, f, indent=4, ensure_ascii=False)
    
# scenario_mcq

law_data = data
output_path = f"{task_root['scenario_mcq']}/{model_name}/result_{tag}.json"

with open(law_qa_path, encoding='utf-8') as f:
    qa_data = json.load(f)
    
answers = {}

for i, law_name in tqdm(enumerate(qa_data)):
    new_law = law_data[law_name][1].replace("**", "").strip()
    
    qa = qa_data[law_name]
    
    question = LLMprompt.get_mcq_prompt(qa, 0, [f"{law_name}：{new_law}"])
    
    if i == 0:
        print(question)
        
    answer = LLM.get_response(edited_model, tokenizer, question, 30)
    answers[law_name] = {
        "answer": answer,
        "ground_truth": qa_data[law_name]["Answer_after"]
    }
    
with open(output_path, "w+", encoding='utf-8') as f:
    json.dump(answers, f, indent=4, ensure_ascii=False)
    
    
# discri
output_path = f"{task_root['discri']}/{model_name}/result_{tag}.json"
with open(law_dis_path, encoding='utf-8') as f:
    qa_data = json.load(f)
    
answers = {}

for i, law_name in tqdm(enumerate(qa_data)):
    new_law = law_data[law_name][1].replace("**", "").strip()
    
    qas = qa_data[law_name]
    answers[law_name] = []
    for j, qa in enumerate(qas):
        question = LLMprompt.get_discri_prompt(qa["question"], 0, [f"上一版本的{law_name}：{law_data[law_name][0].strip()}", f"最新版本的{law_name}：{new_law}"])
        
        if i == 0 and j == 0:
            print(question)
            
        answer = LLM.get_response(edited_model, tokenizer, question, 30)
        answers[law_name].append({
            "answer": answer,
            "ground_truth": qa["answer"]
        })
    
with open(output_path, "w+", encoding='utf-8') as f:
    json.dump(answers, f, indent=4, ensure_ascii=False)
    
# notdiscri

output_path = f"{task_root['notdiscri']}/{model_name}/result_{tag}.json"
with open(law_not_dis_path, encoding='utf-8') as f:
    qa_data = json.load(f)
answers = {}

for i, law_name in tqdm(enumerate(qa_data)):
    law = old_law_data[law_name].strip()
    
    qas = qa_data[law_name]
    answers[law_name] = []
    for j, qa in enumerate(qas):
        question = LLMprompt.get_discri_prompt(qa["question"], 0, [f"上一版本的{law_name}：{law}", f"最新版本的{law_name}：{law}"])
        
        if i == 0 and j == 0:
            print(question)
            
        answer = LLM.get_response(edited_model, tokenizer, question, 30)
        answers[law_name].append({
            "answer": answer,
            "ground_truth": qa["answer"]
        })
    
with open(output_path, "w+", encoding='utf-8') as f:
    json.dump(answers, f, indent=4, ensure_ascii=False)

# notrecall
output_path = f"{task_root['notrecall']}/{model_name}/result_{tag}.json"
answers = {}


for i, law_name in tqdm(enumerate(qa_data)):
    law = old_law_data[law_name].strip()

    inputs = tokenizer(law_name, return_tensors='pt').input_ids
    _, length = inputs.shape
    
    question = LLMprompt.get_recall_prompt(law, 0, [f"{law_name}：{law}"])
        
    
    if i == 0:
        print(question)
    
        
    answer = LLM.get_response(edited_model, tokenizer, question, int(length*3))
    answers[law_name] = {
        "answer": answer,
        "ground_truth": law_name
    }
    
    
with open(output_path, "w+", encoding='utf-8') as f:
    json.dump(answers, f, indent=4, ensure_ascii=False)
    
# notrecite

output_path = f"{task_root['notrecite']}/{model_name}/result_{tag}.json"

answers = {}

for i, law_name in tqdm(enumerate(qa_data)):
    law = old_law_data[law_name].strip()


    inputs = tokenizer(law, return_tensors='pt').input_ids
    _, length = inputs.shape
    
    
    question = LLMprompt.get_recite_prompt(law_name, 0, [f"{law_name}：{law}"])
    
    if i == 0:
        print(question)
        
    answer = LLM.get_response(edited_model, tokenizer, question, int(length*2))
    answers[law_name] = {
        "answer": answer,
        "ground_truth": law
    }
    
with open(output_path, "w+", encoding='utf-8') as f:
    json.dump(answers, f, indent=4, ensure_ascii=False)
    
