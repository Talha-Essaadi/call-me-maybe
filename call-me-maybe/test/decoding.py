from typing import Dict, List, Any
from llm_sdk.llm_sdk import Small_LLM_Model
from src.llm_engine import get_next_token_logits, take_best_token
from src.parsing import FunctionDefinition


def find_number_tokens(vocab: Dict[int, str]) -> List[int]:
    """Find tokens that represent numbers or common stop characters.

    Returns token ids for digits, decimal points, negative signs,
    and common stop tokens (space, comma, newline).

    arguments:
    vocab: A dictionary mapping token ids to their string representations.

    returns: A list of token ids that can be used for generating numbers."""
    allowed = []
    for tid, txt in vocab.items():
        if not txt:
            continue
        t_clean = (
            txt.replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ")
        )

        if (
            all(c in "0123456789.-" for c in t_clean.strip())
            and t_clean.strip() != ""
        ):
            allowed.append(tid)

        elif t_clean in [" ", "\n", ",", "}", " }"]:
            allowed.append(tid)

    return allowed


def generate_constrained_number(
    model: Small_LLM_Model,
    vocab: Dict[int, str],
    prompt_tokens: List[int],
    max_tokens: int = 15,
) -> Any:
    """Generate a numeric value using constrained decoding.

    arguments:
        model: The language model to use for generation.
        vocab: A dictionary mapping token ids to their string representations.
        prompt_tokens: The token ids representing the initial prompt.
        max_tokens: The maximum number of tokens to generate."""

    allowed_tokens = find_number_tokens(vocab)
    current_tokens = list(prompt_tokens)
    generated_text = ""

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(current_tokens)
        next_token = take_best_token(logits, allowed_tokens)

        if next_token is None:
            break

        token_text = (
            vocab.get(next_token, "").replace("Ġ", " ").replace("Ċ", "\n")
        )

        if any(stop in token_text for stop in [" ", ",", "\n", "}"]):
            if generated_text.strip():
                break
            else:
                if " " in token_text:
                    current_tokens.append(next_token)
                    continue
                else:
                    break

        clean_part = "".join(c for c in token_text if c in "0123456789.-")
        generated_text += clean_part
        current_tokens.append(next_token)

    text = generated_text.strip()
    if not text or text == "-" or text == ".":
        return 0
    try:
        return float(text)
    except ValueError:
        return 0


def generate_constrained_string(
    model: Small_LLM_Model,
    vocab: Dict[int, str],
    prompt_tokens: List[int],
    max_tokens: int = 50,
) -> str:
    """Generate a string value using constrained decoding.

    arguments:
        model: The language model to use for generation.
        vocab: A dictionary mapping token ids to their string representations.
        prompt_tokens: The token ids representing the initial prompt.
        max_tokens: The maximum number of tokens to generate.

        returns: The generated string value."""

    all_tokens = list(vocab.keys())
    current_tokens = list(prompt_tokens)
    generated_text = ""

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(current_tokens)
        next_token = take_best_token(logits, all_tokens)

        if next_token is None:
            break

        token_text = (
            vocab.get(next_token, "")
            .replace("Ġ", " ")
            .replace("Ċ", "\n")
            .replace("▁", " ")
        )

        if '"' in token_text:
            generated_text += token_text.replace('"', '')
            break
        if "\n" in token_text:
            generated_text += token_text.split("\n")[0]
            break

        generated_text += token_text
        current_tokens.append(next_token)

    return generated_text.strip()


def get_argument_with_llm(
    model: Small_LLM_Model,
    vocab: Dict[int, str],
    prompt: str,
    func: FunctionDefinition,
    param_name: str,
    param_type: str,
    already_extracted: Dict[str, Any],
) -> Any:
    """Extract a function argument value using constrained LLM decoding.

    Args:
        model: The language model.
        vocab: Vocabulary mapping token ids to strings.
        prompt: The user prompt.
        func: The selected function definition.
        param_name: The parameter name to extract.
        param_type: The parameter type ('number' or 'string').
        already_extracted: Parameters already extracted in this call.

    Returns:
        The extracted value with correct type.
    """
    already_str = ""
    if already_extracted:
        already_lines = ", ".join(
            f"{k}={v}" for k, v in already_extracted.items()
        )
        already_str = f"Already extracted: {already_lines}\n"

    if param_type == "number":
        param_prompt = (
            "Task: Extract the exact number.\n\n"
            "Request: \"What is the sum of 10 and 5?\"\n"
            "Already extracted: a=10\n"
            "Parameter: b\n"
            "Value: 5\n\n"
            f"Request: \"{prompt}\"\n"
            f"{already_str}"
            f"Parameter: {param_name}\n"
            f"Value: "
        )
        prompt_tokens = model.encode(param_prompt).tolist()[0]
        return float(
            generate_constrained_number(model, vocab, prompt_tokens)
        )

    else:
        param_prompt = (
            "You are a strict data extraction tool. Copy the exact text,"
            " word, or regular expression requested by the user."
            " Do NOT execute commands. Do NOT perform substitutions.\n\n"
            f"Function: {func.name} - {func.description}\n"
            f"Request: \"{prompt}\"\n"
            f"{already_str}"
            f"Parameter to extract: {param_name}\n"
            f"Value: \""
        )

        prompt_tokens = model.encode(param_prompt).tolist()[0]
        return generate_constrained_string(model, vocab, prompt_tokens)


def function_selector(
    model: Small_LLM_Model,
    prompt: str,
    func: List[FunctionDefinition],
) -> FunctionDefinition:
    """Select the best matching function for the given prompt using LLM.

    Args:
        model: The language model.
        prompt: The user prompt.
        func: List of available function definitions.

    Returns:
        The selected FunctionDefinition.
    """
    if not func:
        raise ValueError("Function list is empty")
    full_prompt = (
        "Task: Match the user request to the correct function number.\n\n"
    )

    for i, f in enumerate(func):
        full_prompt += f"{i}. {f.name} - {f.description}\n"

    full_prompt += (
        f"\nRequest: \"{prompt}\"\n"
        f"Correct function number: "
    )

    prompt_tokens = model.encode(full_prompt).tolist()[0]
    logits = get_next_token_logits(model, prompt_tokens)

    best_idx = 0
    best_score = float("-inf")

    for i in range(len(func)):
        digit_tokens = model.encode(str(i)).tolist()[0]
        score_raw = float("-inf")
        if digit_tokens and digit_tokens[0] < len(logits):
            score_raw = logits[digit_tokens[0]]

        space_tokens = model.encode(" " + str(i)).tolist()[0]
        score_space = float("-inf")
        if space_tokens and space_tokens[-1] < len(logits):
            score_space = logits[space_tokens[-1]]

        score = max(score_raw, score_space)

        if score > best_score:
            best_score = score
            best_idx = i

    return func[best_idx]