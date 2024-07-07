path = "../task_generation/result/recall"
# For locality
# path = "../task_generation/result/notrecall"

import json
import os
import cn2an

def get_acc(path):
    acc = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.endswith(".json"):
                continue
            model = root.split("/")[-1]
            method = file.split(".json")[0].split("_")[-1]
            # if model != "chatglm" or method != "ft":
            #     continue
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
        # if "出自" in answer or ("第" in answer and "条" in answer):
        #     zxzl += 1
        origin = answer
        law = "民法典" if "民法典" in data[k]["ground_truth"] else "刑法"
        ft = data[k]["ground_truth"].split("第")[-1].strip()
        # 只保留中文
        if len(data[k]["answer"].split("第")) == 2 and law in answer:
            if ft in answer:
                correct += 1
            elif "之" not in ft:
                try:
                    ft = str(cn2an.cn2an(ft.split("条")[0], "normal"))
                    if ft in answer:
                        correct += 1
                    else:
                        print(origin, "||", ft)

                except:
                    print(origin, "||", ft)

                    continue
            else:
                print(origin, "||", ft)

        else:
            print(origin, "||", ft)

                
    return correct / len(data)
            
    
acc = get_acc(path)
from utils import save_dict_to_excel

save_dict_to_excel(acc, path)

