import jieba
from rank_bm25 import BM25Okapi
import json

root_path =  "../../data"
law_add_path = f"{root_path}/LawAdd.json"
old_law_path = f"{root_path}/old_law_all.json"

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
            
def chinese_tokenizer(text):
    return list(jieba.cut(text))
tokenized_corpus = [chinese_tokenizer(doc) for doc in docs]


def search(query, topk=3):
    
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = chinese_tokenizer(query)
    doc_scores = bm25.get_scores(tokenized_query)
    topk_docs_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:topk]
    return [docs[i] for i in topk_docs_indices]
