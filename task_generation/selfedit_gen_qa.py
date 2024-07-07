import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
import json
import sys
from tqdm import tqdm

#---------------------------------#
sample_times = 5
#---------------------------------#


data_path = "../data/LawAdd.json"
model_name = sys.argv[1]

assert model_name in ["baichuanchat", "chatlaw", "chatglm", "legalaid"]
if model_name == "baichuanchat":
    from model import baichuanchat as LLM
elif model_name == "chatlaw":
    from model import chatlaw as LLM
elif model_name == "chatglm":
    from model import chatglm as LLM
elif model_name == "legalaid":
    from model import legalaid as LLM
else:
    raise Exception("Model not matched.")

output_path = f"./selfeditqa/{model_name}.json"

model, tokenizer = LLM.get_model()

data = json.load(open(data_path, encoding='utf-8'))
questions = {}

flag = True
for k in tqdm(data):
    questions[k] = []
    prompt = f"最新版本的{k}是：{data[k][1].replace('**', '')}\n请关于这个法条的其中一点提出一个问题："
    answer_prefix = f"问题为："
    
    for _ in range(sample_times):
        question = LLM.get_response_with_answer_prefix(model, tokenizer, prompt, answer_prefix, greedy=False)
        question_prompt = f"问题：根据{k}，{question}"
        cot = f"最新版本的{k}是：{data[k][1].replace('**', '')}。所以问题的答案是："
        answer = LLM.get_response_with_answer_prefix(model, tokenizer, question_prompt, cot) 
        questions[k].append([question, answer])
    if flag:
        print(prompt)
        print("-----")
        print(question)
        print("-----")
        print(answer)
        flag = False

with open(output_path, "w+", encoding='utf-8') as f:
    json.dump(questions, f, ensure_ascii=False, indent=4)