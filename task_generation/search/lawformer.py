
import faiss
import json
from transformers import AutoModel, AutoTokenizer
import numpy as np

#-----------------------------------------#
lawformer_path = "/path/to/lawformer"
#-----------------------------------------#

root_path =  "../../data"
law_add_path = f"{root_path}/LawAdd.json"
old_law_path = f"{root_path}/old_law_all.json"

model = AutoModel.from_pretrained(lawformer_path).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(lawformer_path)

index = faiss.read_index("./lf.index")


add_law = json.load(open(law_add_path, encoding="utf-8"))
docs = []
with open(old_law_path, encoding="utf-8") as f:
    adata = json.load(f)
    for key in adata:
        if key in list(add_law.keys()):
            docs.append(f'最新版本的{key}：\n{add_law[key][1].replace("**", "").strip()}')
            docs.append(f'上一版本的{key}：\n{add_law[key][0].strip()}')
        else:
            docs.append(f"最新版本的{key}：\n{adata[key].strip()}")
            docs.append(f"上一版本的{key}：\n{adata[key].strip()}")



def search(query, topk=3):
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    embeded = outputs.last_hidden_state[0][0].tolist()
    D, I = index.search(np.array([embeded]).astype('float32'), topk)
    return [docs[i] for i in I[0]]
    