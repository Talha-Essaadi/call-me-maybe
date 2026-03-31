from typing import List, Dict
from llm_sdk import Small_LLM_Model
from .models import FunctionDefinition
import textwrap
import json


def load_vocabulary(llm: Small_LLM_Model) -> Dict[int, str]:
    """Load and return the model vocabulary as {token_id: token_text}.

    arguments:
        model: The language model to load the vocabulary from.

    returns: A dictionary mapping token ids to their string representations."""
    try:
        vocab_path = llm.get_path_to_vocab_file()
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)

        return {v: k for k, v in vocab.items()}
    except Exception as e:
        print(f"Error loading vocabulary: {e}")
        return {}


def generate_prompt(functions: List[FunctionDefinition], user_request: str) -> str:
    function_names = [fn.name for fn in functions]

    prompt = textwrap.dedent(f"""
    You are a function selection engine.

    Output ONLY one function name.

    Available functions:
    {function_names}

    User request:
    "{user_request}"

    Answer:
    """)

    return prompt.strip()


def generate_output_from_prompt(llm: Small_LLM_Model, prompt: str) -> str:
    ids = llm.encode(prompt)[0].tolist()

    max_tokens = 10

    for _ in range(max_tokens):

        logits = llm.get_logits_from_input_ids(ids)

        next_token = max(range(len(logits)), key=lambda i: logits[i])

        ids.append(next_token)

        text = llm.decode(ids)

        if (
            next_token == llm._tokenizer.eos_token_id
            or text.endswith("\n")
        ):
            break

    text = llm.decode(ids)

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1].split()[-1]


def constrained_decoding(logits: list, param_type: str, vocab: dict) -> list:
    if param_type == "string":
        return logits
    elif param_type == "number":
        for i in range(len(logits)):
            txt = vocab.get(i)
            if not txt:
                continue
            t_clean = (
            txt.replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ")
            )
            if (
                t_clean.strip() != ""
                and not all(c in "0123456789.-" for c in t_clean.strip())
            ):
                logits[i] = float("-inf")
        return logits

            



def generate_output_for_param(
    llm: Small_LLM_Model,
    prompt: str,
    param_type: str,
    vocab: dict
    ) -> str:
    ids = llm.encode(prompt)[0].tolist()
    max_tokens = 30

    for _ in range(max_tokens):

        logits = llm.get_logits_from_input_ids(ids)
        logits = constrained_decoding(logits, param_type, vocab)
        next_token = max(range(len(logits)), key=lambda i: logits[i])

        ids.append(next_token)

        text = llm.decode(ids)

        if (
            next_token == llm._tokenizer.eos_token_id
            or text.endswith("\n")
        ):
            break

    text = llm.decode(ids)

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1].split()[-1]


def generate_param_prompt(user_request: str, param_name: str, param_type: str, already_extracted: dict, func: FunctionDefinition) -> str:
    already_str = ""
    if already_extracted:
        already_lines = ", ".join(
            f"{k}={v}" for k, v in already_extracted.items()
        )
        already_str = f"Already extracted: {already_lines}\n"
    if param_type == "number":
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
    else:
        prompt = (
            "You are a strict data extraction tool. Copy the exact text,"
            " word, or regular expression requested by the user."
            " Do NOT execute commands. Do NOT perform substitutions.\n\n"
            f"Function: {func.name} - {func.description}\n"
            f"Request: \"{user_request}\"\n"
            f"{already_str}"
            f"Parameter to extract: {param_name}\n"
            f"Value: \""
        )
    return prompt


def extract_params_from_prompt(
    llm: Small_LLM_Model,
    user_request: str,
    func: FunctionDefinition,
    vocab: dict
    ) -> dict:

    extracted_params = {}
    params_to_extract = func.parameters
    for param_name, param_type in params_to_extract.items():
        prompt = generate_param_prompt(user_request, param_name, param_type, extracted_params, func)
        value = generate_output_for_param(llm, prompt, param_type, vocab)
        extracted_params[param_name] = value
    return extracted_params
