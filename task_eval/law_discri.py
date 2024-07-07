path = "../task_generation/result/discri"

# For locality
# path = "../task_generation/result/notdiscri"

import json
import os

def get_acc(path):
    acc = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.endswith(".json"):
                continue
            model = root.split("/")[-1]
            
            method = file.split(".json")[0].split("_")[-1]
            if model not in acc:
                acc[model] = {}
            with open(os.path.join(root, file), "r") as f:
                data = json.load(f)
                acc[model][method] = acc_(data)
            
            print(model, method)
            
            print("======================")
    return acc

def acc_(ldata):
    correct = 0
    all_ = 0
    for k in ldata:
        data = ldata[k]
        for d in data:
            all_ += 1
            origin = d["answer"]
            if d["answer"].strip()[:2] == "正确":
                answer = True
            elif d["answer"].strip()[:2] == "错误":
                answer = False
            else:
                d["answer"] = d["answer"].strip().split("：")[-1].strip()
                if d["answer"].strip()[:2] == "正确":
                    answer = True
                elif d["answer"].strip()[:2] == "错误":
                    answer = False
                print(origin, "||", d["ground_truth"])
                continue
                
            if answer == d["ground_truth"]:
                correct += 1
                
    return correct / all_
            

acc = get_acc(path)

from utils import save_dict_to_excel

save_dict_to_excel(acc, path)





