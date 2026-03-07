from llm_sdk import Small_LLM_Model
import sys
import argparse
from pathlib import Path
from .loader import load_and_validate_prompts, validate_function_definitions
from typing import List
from .models import PromptInput, FunctionDefinition



def main() -> None:
    try:
        parser = argparse.ArgumentParser(description="Run the Small LLM Model with a prompt.")
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
        if not functions_path.is_file():
            raise FileNotFoundError(f"Functions definition file not found: {functions_path}")
        input_path = Path(args.input)
        if not input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()
        print("")
        print("")
        print("")
        print(f"Functions Definition Path: {functions_path}")
        print(f"Input Path: {input_path}")
        print(f"Output Path: {output_path}")

        print("")
        functions_definition = validate_function_definitions(functions_path)
        input_data = load_and_validate_prompts(input_path)
        print(f"Functions Definition: {functions_definition}")

        print("")
        
    except Exception as e:
        print(f"Error : {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()