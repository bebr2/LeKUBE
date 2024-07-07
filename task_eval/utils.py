import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def get_max(df):
    # df_max = df.idxmax(axis=0, skipna=True, numeric_only=False)

    # 对每列进行操作
    for i, col in enumerate(df.columns):
        # 找到最大值
        rank = df[col].rank(ascending=False, method='min')

        max_index = df.index[rank == 1]
        sencond_index = df.index[rank == 2]
        for j in max_index:
            df[col][j] = str(df[col][j]) + " MAX"
        for j in sencond_index:
            df[col][j] = str(df[col][j]) + " SEC"
            
    return df

def save_dict_to_excel(acc, path):
    model_order = ["baichuanchat", "chatglm", "chatlaw", "legalaid"]
    method_order = ["raw", "icl", "bm25", "dense", "ft", "lora", "selfedit", "kn", "rome"]
    method_set = set()

    for m in acc:
        for k in acc[m]:
            method_set.add(k)

    method_order_diff = list(set(method_order) - method_set)
    for k in method_order_diff:
        method_order.remove(k)
    
    df = pd.DataFrame(acc)
    df = df.reindex(method_order)
    df = df.reindex(columns=model_order)
    df = df.round(4)
    get_max(df).to_excel(f"{path}/em_{path.split('/')[-1]}.xlsx")

    # for m in acc:
    #     for k in acc[m]:
    #         if k != "icl":
    #             acc[m][k] -= acc[m]["icl"]
                
    # for m in acc:
    #     acc[m]["icl"] = 0
                
            
    # df = pd.DataFrame(acc)
    # df = df.reindex(method_order)
    # df = df.reindex(columns=model_order)
    # df = df.round(4)
    # get_max(df).to_excel(f"{path}/emdiff.xlsx")