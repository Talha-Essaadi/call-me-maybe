import json
from pathlib import Path
from pydantic import ValidationError
from typing import Any, List
from .models import PromptInput, FunctionDefinition


def load_and_validate_prompts(path: Path) -> List[PromptInput]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    if not isinstance(data, list):
        raise ValueError("Input file must contain a JSON array")

    try:
        prompts = [PromptInput(**item) for item in data]
    except ValidationError as e:
        raise ValueError(f"Invalid prompt structure: {e}")

    return prompts


def validate_function_definitions(path: Path) -> List[FunctionDefinition]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    if not isinstance(data, list):
        raise ValueError("functions_definition.json must contain a JSON array")

    try:
        functions = [FunctionDefinition(**item) for item in data]
    except ValidationError as e:
        raise ValueError(f"Invalid function definition format:\n{e}")

    return functions