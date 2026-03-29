from llm_sdk import Small_LLM_Model
import sys
import json
import argparse
from pathlib import Path
from typing import List
import textwrap

from .loader import load_and_validate_prompts, validate_function_definitions
from .models import FunctionDefinition



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
    return lines[-1]


def main() -> None:
    try:
        parser = argparse.ArgumentParser(
            description="Run Small LLM function selector."
        )

        parser.add_argument(
            "--functions_definition",
            type=str,
            default="data/input/functions_definition.json",
        )

        parser.add_argument(
            "--input",
            type=str,
            default="data/input/function_calling_tests.json",
        )

        parser.add_argument(
            "--output",
            type=str,
            default="data/output/function_calling_results.json",
        )

        args = parser.parse_args()

        functions_path = Path(args.functions_definition)
        input_path = Path(args.input)
        output_path = Path(args.output)

        if not functions_path.is_file():
            raise FileNotFoundError(functions_path)

        if not input_path.is_file():
            raise FileNotFoundError(input_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        functions_definition = validate_function_definitions(functions_path)
        functions = []
        for fn in functions_definition:
            parameters = {}
            for param_name, param_def in fn.parameters.items():
                parameters[param_name] = param_def.type
            fn.parameters = parameters
            functions.append(fn)

        functions_lookup = {fn.name: fn for fn in functions}

        input_data = load_and_validate_prompts(input_path)

        llm = Small_LLM_Model()

        output_list = []
        for prompts_list in input_data:
            prompt = generate_prompt(
                functions_definition,
                prompts_list.prompt,
            )

            output = generate_output_from_prompt(llm, prompt)
            fn_name = output.split()[1] if output.startswith("Answer:") else output
            if fn_name not in [fn.name for fn in functions_definition]:
                raise ValueError(f"Invalid function name: {fn_name}")
            function = functions_lookup[fn_name]
            result = {
                "prompt": prompts_list.prompt,
                "name": fn_name,
                "parameters": function.parameters,
            }
            output_list.append(result)
            print(result)

        with output_path.open("w") as f:
            json.dump(output_list, f, indent=4)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()