import sys
import json
import argparse
from pathlib import Path
from typing import List
from .loader import load_and_validate_prompts, validate_function_definitions
from .models import FunctionDefinition
from .utils import JsonStructure



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
        with functions_path.open("r") as f:
            functions = json.load(f)
        # print(functions)
        input_data = load_and_validate_prompts(input_path)
        generate_output = []
        JsonStructure(
            generate_output,
            functions_definition,
            functions,
            input_data
            )

    
        with output_path.open("w") as f:
            json.dump(generate_output, f, indent=4)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()