import random
import math
from llm_sdk import Small_LLM_Model


llm = Small_LLM_Model()

FUNCTION_LIST = [
    "fn_add_numbers",
    "fn_greet",
    "fn_reverse_string",
    "fn_get_square_root",
    "fn_substitute_string_with_regex",
]

prompt = f"""
You are a function selection engine.

Your task is ONLY to choose the most appropriate function
from the list below that can solve the user's request.

Rules:
- Output ONLY a function name.
- Never output explanations.
- Never output arguments.
- Never output JSON.
- Choose the BEST matching function.
You MUST output ONLY the function name.
Do NOT explain.
Do NOT generate JSON.
Do NOT answer the question.

Available functions:

{FUNCTION_LIST}

User request:
"What is the sum of 265 and 345?"

Answer with ONLY one function name:
"""
ids = llm.encode(prompt)
ids = ids[0].tolist()


def softmax(logits):
    exp_logits = [math.exp(x) for x in logits]
    sum_exp = sum(exp_logits)
    return [x / sum_exp for x in exp_logits]


max_tokens = 50
for _ in range(max_tokens):
    logits = llm.get_logits_from_input_ids(ids)
    probs = softmax(logits)
    

    # next_token = random.choices(range(len(probs)), weights=probs, k=1)[0]
    next_token = probs.index(max(probs))
    
    ids.append(next_token)
    
    text = llm.decode(ids)
    if "\n" == text[-1]:
        break


text = llm.decode(ids)
output = text.splitlines()[-1]
print(text)