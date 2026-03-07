import argparse
import sys
from pathlib import Path

from .loader import load_and_validate_functions, load_and_validate_prompts
from .pipeline import FunctionCallingPipeline


def main() -> None:
    """Entry point for the function calling system.

    Parses command-line arguments, loads input files,
    initializes the LLM pipeline, processes prompts,
    and saves results.
    """
    try:
        parser = argparse.ArgumentParser(
            description="Function calling system using "
            "constrained decoding with LLMs."
        )
        parser.add_argument(
            "--functions_definition",
            type=str,
            default="data/input/functions_definition.json",
            help="Path to the function definitions JSON file.",
        )
        parser.add_argument(
            "--input",
            type=str,
            default="data/input/function_calling_tests.json",
            help="Path to the input prompts JSON file.",
        )
        parser.add_argument(
            "--output",
            type=str,
            default="data/output/function_calling_results.json",
            help="Path to the output results JSON file.",
        )
        parser.add_argument(
            "--model",
            type=str,
            default="Qwen/Qwen3-0.6B",
            help="HuggingFace model name to use.",
        )

        args = parser.parse_args()

        functions_path = Path(args.functions_definition)
        input_path = Path(args.input)
        output_path = Path(args.output)

        print("=" * 60)
        print("  Call Me Maybe - Function Calling System")
        print("=" * 60)
        print(f"  Functions: {functions_path}")
        print(f"  Input:     {input_path}")
        print(f"  Output:    {output_path}")
        print(f"  Model:     {args.model}")
        print("=" * 60)

        print("\nLoading function definitions...")
        functions = load_and_validate_functions(functions_path)
        print(f"Loaded {len(functions)} function definitions:")
        for func in functions:
            params = ", ".join(
                f"{k}: {v.type}"
                for k, v in func.parameters.items()
            )
            print(f"  - {func.name}({params})")

        print("\nLoading input prompts...")
        prompts = load_and_validate_prompts(input_path)
        print(f"Loaded {len(prompts)} prompts")

        pipeline = FunctionCallingPipeline(
            function_definitions=functions,
            model_name=args.model,
        )
        pipeline.initialize()

        results = pipeline.process_all(prompts)

        pipeline.save_results(results, output_path)

        print("\nDone!")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Runtime error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
