from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm
import faiss
import numpy as np

#-----------------------------------------#
lawformer_path = "/path/to/lawformer"
#-----------------------------------------#

root_path =  "../../data"
law_add_path = f"{root_path}/LawAdd.json"
old_law_path = f"{root_path}/old_law_all.json"


model = AutoModel.from_pretrained(lawformer_path).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(lawformer_path)



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
result = []
for law in tqdm(docs):
    inputs = tokenizer(law, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    embeded = outputs.last_hidden_state[0][0].tolist()
    result.append(embeded)
    


train = np.array(result).astype('float32')

index = faiss.IndexFlatIP(768) # 建立Inner product索引
index.add(train)  # 添加矩阵

faiss.write_index(index, "./lf.index")
    
