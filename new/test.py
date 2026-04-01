import random
import math
from llm_sdk import Small_LLM_Model
import json


llm = Small_LLM_Model()


def generate_prompt(user_request: str, param_name: str, already_str: str) -> str:
    prompt = (
            "Task: Extract the exact number.\n\n"
            "Request: \"What is the sum of 10 and 5?\"\n"
            "Already extracted: a=10\n"
            "Parameter: b\n"
            "Value: 5\n\n"
            f"Request: \"{user_request}\"\n"
            f"{already_str}"
            f"Parameter: {param_name}\n"
            f"Value: "
        )
    return prompt

user_request = "What is the sum of 265 and 345?"
vocab_path = llm.get_path_to_vocab_file()
with open(vocab_path, "r") as f:
    vocab = json.load(f)

vocab = {v: k for k, v in vocab.items()}
print(vocab)
# prompt = generate_prompt(user_request, "b", "a")
# ids = llm.encode(prompt)
# ids = ids[0].tolist()


# def softmax(logits):
#     exp_logits = [math.exp(x) for x in logits]
#     sum_exp = sum(exp_logits)
#     return [x / sum_exp for x in exp_logits]


# max_tokens = 50
# for _ in range(max_tokens):
#     logits = llm.get_logits_from_input_ids(ids)
#     probs = softmax(logits)

#     next_token = probs.index(max(probs))
    
#     ids.append(next_token)
    
#     text = llm.decode(ids)
#     if "\n" == text[-1]:
#         break


# text = llm.decode(ids)
# output = text.splitlines()[-1]
# print(text)