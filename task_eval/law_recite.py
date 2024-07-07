path = "../task_generation/result/recite"
# For locality
# path = "../task_generation/result/notrecite"

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
                acc[model][method] = em(data)
            print(model, method)
            
            print("======================")
    return acc

def em(data):
    correct = 0
    for k in data:
        answer = data[k]["answer"]
        if not answer:
            continue
        origin = answer
        answer = answer.split("\n\n请问")[0].split("\n\n这意味着")[0].split("规定：", 1)[-1].split("规定，", 1)[-1]
        
        ground_truths = [data[k]["ground_truth"], k + " " + data[k]["ground_truth"]]
        # 只保留中文
        for ground_truth in ground_truths:
            answer = "".join(filter(lambda x: '\u4e00' <= x <= '\u9fa5', answer))
            ground_truth = "".join(filter(lambda x: '\u4e00' <= x <= '\u9fa5', ground_truth))
            if answer == ground_truth or ground_truth.endswith(answer):
                correct += 1
                break
        else:
            print(origin, "||", data[k]["ground_truth"])
                
    return correct / len(data)
            
    
acc = get_acc(path)

from utils import save_dict_to_excel

save_dict_to_excel(acc, path)

