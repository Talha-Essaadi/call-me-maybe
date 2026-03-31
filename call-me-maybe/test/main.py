import argparse
from .parsing import validate_func_def, validate_prompt
from llm_sdk.llm_sdk import Small_LLM_Model
from .llm_engine import load_vocabulary
from .constrained import function_selector, get_argument_with_llm
import json
import os
from typing import Dict, Any


def main() -> None:
    """Run function calling inference on test prompts and save results."""
    parser = argparse.ArgumentParser(
        description="Load function definitions and test cases, "
                    "then save results.",
    )

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the JSON file containing function definitions.",
    )

    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the JSON file with test cases.",
    )

    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to the JSON file where results will be saved.",
    )

    args = parser.parse_args()
    functions = validate_func_def(args.functions_definition)
    if functions is None:
        print("Error: No valid function definitions found.")
        return
    prompts = validate_prompt(args.input)
    if prompts is None:
        print("Error: No valid prompts found.")
        return

    print("Loading model...")
    model = Small_LLM_Model()
    print("Model loaded")

    print("Loading vocabulary...")
    vocab = load_vocabulary(model)
    print(f"Vocabulary loaded: {len(vocab)} tokens")

    results = []
    for i, p in enumerate(prompts):
        try:
            print(f"\n--- Prompt {i + 1}: {p.prompt}")

            selected_func = function_selector(model, p.prompt, functions)
            print(f"    Selected function: {selected_func.name}")

            arguments: Dict[str, Any] = {}
            for param_name, param_def in selected_func.parameters.items():
                value = get_argument_with_llm(
                    model, vocab, p.prompt,
                    selected_func, param_name, param_def.type,
                    already_extracted=arguments,
                )
                arguments[param_name] = value
                print(f"    {param_name} ({param_def.type}): {value}")

            results.append({
                "prompt": p.prompt,
                "name": selected_func.name,
                "parameters": arguments,
            })
        except Exception as e:
            print(f"Error processing prompt {i + 1}: {e}")
            continue

    print(f" Processed {len(results)} prompts")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")