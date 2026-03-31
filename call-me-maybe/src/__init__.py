from llm_sdk import Small_LLM_Model
import sys
import json
import argparse
from pathlib import Path
from typing import List
from .loader import load_and_validate_prompts, validate_function_definitions
from .models import FunctionDefinition
from .utils import extract_params_from_prompt, generate_output_from_prompt, generate_param_prompt, generate_prompt, generate_output_for_param, load_vocabulary





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
        vocab = load_vocabulary(llm)

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
            parameters = extract_params_from_prompt(
                llm,
                prompts_list.prompt,
                function,
                vocab
            )
            result = {
                "prompt": prompts_list.prompt,
                "name": fn_name,
                "parameters": parameters,
            }
            output_list.append(result)
            print(result)

        with output_path.open("w") as f:
            json.dump(output_list, f, indent=4)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()