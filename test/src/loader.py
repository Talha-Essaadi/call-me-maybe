import json
from pathlib import Path
from pydantic import ValidationError
from typing import List

from .models import PromptInput, FunctionDefinition


def load_and_validate_prompts(path: Path) -> List[PromptInput]:
    """Load and validate prompts from a JSON file.

    Args:
        path: Path to the JSON file containing prompts.

    Returns:
        A list of validated PromptInput objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON or structure.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in prompts file: {e}")

    if not isinstance(data, list):
        raise ValueError("Input file must contain a JSON array")

    prompts: List[PromptInput] = []
    for i, item in enumerate(data):
        try:
            prompts.append(PromptInput(**item))
        except (ValidationError, TypeError) as e:
            print(f"Warning: Skipping invalid prompt at index {i}: {e}")

    if not prompts:
        raise ValueError("No valid prompts found in input file")

    return prompts


def load_and_validate_functions(
    path: Path,
) -> List[FunctionDefinition]:
    """Load and validate function definitions from a JSON file.

    Args:
        path: Path to the JSON file containing function definitions.

    Returns:
        A list of validated FunctionDefinition objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON or structure.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in function definitions file: {e}"
        )

    if not isinstance(data, list):
        raise ValueError(
            "Function definitions file must contain a JSON array"
        )

    functions: List[FunctionDefinition] = []
    for i, item in enumerate(data):
        try:
            functions.append(FunctionDefinition(**item))
        except (ValidationError, TypeError) as e:
            print(
                f"Warning: Skipping invalid function "
                f"definition at index {i}: {e}"
            )

    if not functions:
        raise ValueError(
            "No valid function definitions found in input file"
        )

    return functions
