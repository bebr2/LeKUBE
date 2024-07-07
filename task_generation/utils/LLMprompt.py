try:
    from search import bm25, lawformer
except:
    pass

def get_recite_prompt(new_law, search_type=0, search_result=None):
    question = f"请复述{new_law}："
    if search_type == 0:
        return question
    elif search_type == 2:
        search_result = bm25.search(question)
    elif search_type == 3:
        search_result = lawformer.search(question)
        
    return "\n\n".join(search_result) + "\n\n以上是可能相关的法条，" + question
    
def get_recall_prompt(new_law, search_type=0, search_result=None):
    question = f"{new_law}\n\n以上法条出自哪部法律的第几条？"
    if search_type == 0:
        return question
    elif search_type == 2:
        search_result = bm25.search(question)
    elif search_type == 3:
        search_result = lawformer.search(question)
    return "\n\n".join(search_result) + "\n\n以上是可能相关的法条，请回答下列问题：" + question

def get_mcq_prompt(q, search_type=0, search_result=None):
    question = f'回答下面的单项选择题，请你给出正确选项，不要解释原因。请只给出答案的序号。\n\n问题：{q["Question"]}'
    for s in ["A", "B", "C", "D"]:
        question += f"\n{s}. {q[s]}"
    question += "\n答案："
    if search_type == 0:
        return question
    elif search_type == 2:
        search_result = bm25.search(question)
    elif search_type == 3:
        search_result = lawformer.search(question)
    return "\n\n".join(search_result) + "\n\n以上是可能相关的法条，" + question

def get_discri_prompt(q, search_type=0, search_result=None):
    question = f'回答下面的判断题，请你直接给出答案（正确/错误），不要解释原因。\n\n判断题：{q}\n答案：'
    if search_type == 0:
        return question
    elif search_type == 2:
        search_result = bm25.search(question)
    elif search_type == 3:
        search_result = lawformer.search(question)
    return "\n\n".join(search_result) + "\n\n以上是可能相关的法条，" + question