path = "../task_generation/result/scenario_mcq_after"

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

def acc_(data):
    correct = 0
    for k in data:
        answer = data[k]["answer"]
        origin = answer
        answer = answer.strip().split(".")[0].strip()
        if answer.lower() == data[k]["ground_truth"].lower():
            correct += 1
        else:
            answer = answer.strip().split("：")[-1].strip()
            if answer.lower() == data[k]["ground_truth"].lower():
                correct += 1
            else:
                print(origin, "||", data[k]["ground_truth"])
                
    return correct / len(data)
            
    
acc = get_acc(path)
from utils import save_dict_to_excel

save_dict_to_excel(acc, path)

