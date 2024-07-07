import os
import sys

model_name = sys.argv[1]
model_path = None
tag = ""
prompt_type = int(sys.argv[2]) # 0: Non-prefix，1：ICL，2: RAG_BM25，3：RAG_DENSE

law_add_path = "../data/LawAdd.json"
law_qa_path = "../data/LawDiscri.json"


root = "./result/discri"


if prompt_type == 1:
    tag = "icl"
elif prompt_type == 2:
    tag = "rag_bm25"
elif prompt_type == 3:
    tag = "rag_dense"
elif prompt_type == 0:
    tag = "raw"
    

    
import json
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[3]

if len(sys.argv) > 4:
    tag = sys.argv[4]
if len(sys.argv) > 5:
    model_path = sys.argv[5]

print("discri", model_name, tag)

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

from utils import LLMprompt

if model_path is None:
    model, tokenizer = LLM.get_model()
else:
    model, tokenizer = LLM.get_model(model_path)
    
if not os.path.exists(f"{root}/{model_name}"):
    os.mkdir(f"{root}/{model_name}")
    
output_path = f"{root}/{model_name}/result_{tag}.json"

with open(law_add_path, encoding='utf-8') as f:
    law_data = json.load(f)
    
with open(law_qa_path, encoding='utf-8') as f:
    qa_data = json.load(f)
    

    
answers = {}


for i, law_name in tqdm(enumerate(qa_data)):
    new_law = law_data[law_name][1].replace("**", "").strip()
    
    qas = qa_data[law_name]
    answers[law_name] = []
    for j, qa in enumerate(qas):
        question = LLMprompt.get_discri_prompt(qa["question"], prompt_type, [f"上一版本的{law_name}：{law_data[law_name][0].strip()}", f"最新版本的{law_name}：{new_law}"])
        
        if i == 0 and j == 0:
            print(question)
            
        answer = LLM.get_response(model, tokenizer, question, 30)
        answers[law_name].append({
            "answer": answer,
            "ground_truth": qa["answer"]
        })
    
with open(output_path, "w+", encoding='utf-8') as f:
    json.dump(answers, f, indent=4, ensure_ascii=False)
    
    
    




