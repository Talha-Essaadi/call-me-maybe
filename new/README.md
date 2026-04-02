*This project has been created as part of the 42 curriculum by tessaadi*

# call me maybe
Introduction to function calling in LLMs

**Summary:** Do LLMs speak the language of computers? This project explores that question by converting natural-language prompts into structured, typed function calls.

**Version:** 1.2

---

## Description

`call me maybe` is a function-calling pipeline built around a small language model.  
Instead of answering user prompts directly in plain text, the program outputs machine-readable JSON describing:

- the function to call,
- the arguments to pass,
- and argument values with correct types.

The core challenge is **reliability**: small models are often unstable with JSON formatting. To address this, the project uses **constrained decoding** so generation remains valid and schema-compatible token by token.

---

## Table of Contents

1. [Foreword](#foreword)
2. [AI Instructions](#ai-instructions)
3. [Introduction](#introduction)
4. [Common Instructions](#common-instructions)
5. [Mandatory Part](#mandatory-part)
6. [Instructions](#instructions)
7. [Algorithm Explanation](#algorithm-explanation)
8. [Design Decisions](#design-decisions)
9. [Performance Analysis](#performance-analysis)
10. [Challenges Faced](#challenges-faced)
11. [Testing Strategy](#testing-strategy)
12. [Example Usage](#example-usage)
13. [Resources](#resources)
14. [Submission and Peer Review Notes](#submission-and-peer-review-notes)

---

## Foreword

Humans have long relied on structured formats to make information reliable, shareable, and actionable. Function calling in LLMs follows the same principle: move from ambiguous natural language to strict, machine-executable structure.

---

## AI Instructions

AI was used as a productivity assistant during development, with the following rules:

- Use AI to reduce repetitive work, not replace understanding.
- Verify all generated code and explanations.
- Review critical decisions with peers.
- Keep responsibility for every submitted line of code.

Good workflow: prompt AI for options, test locally, discuss with peers, refine.  
Bad workflow: copy-paste generated code without understanding.

---

## Introduction

### What is Function Calling?

Function calling translates a prompt such as:

> "What is the sum of 40 and 2?"

into structured output:

```json
{
        "prompt": "What is the sum of 40 and 2?",
        "name": "fn_add_numbers",
        "parameters": {
                "a": 40,
                "b": 2
        }
}
```

### Why is This Important?

It allows LLM systems to interact safely with tools, APIs, databases, and business logic using deterministic, typed payloads.

### The Challenge

Prompting alone is not enough for stable JSON with small models. The project therefore enforces structure during generation with constrained decoding.

---

## Common Instructions

### General Rules

- Python 3.10+
- `flake8` compliance
- `mypy` typing compliance
- graceful error handling
- robust resource management (context managers)
- pydantic-based validation

### Makefile Targets

- `install`
- `run`
- `debug`
- `clean`
- `lint`
- `lint-strict`

### Additional Guidelines

- include `.gitignore`
- use virtual environment via `uv`
- avoid hardcoded assumptions from sample data

---

## Mandatory Part

### Summary

Given prompts and function definitions, output JSON function calls with exact keys:

- `prompt`
- `name`
- `parameters`

### Input Files

- `data/input/function_calling_tests.json`
- `data/input/functions_definition.json`

### LLM Interaction

The pipeline relies on `llm_sdk/Small_LLM_Model` for token encoding, logits retrieval, and decoding support.

### Output Format

Generated file:

- `data/output/function_calling_results.json`

Validation requirements:

- valid JSON
- no extra keys
- all required parameters present
- parameter types match function schema

---

## Instructions

### Installation

```bash
make install
```

### Run

```bash
make run
```

or directly:

```bash
uv run python -m src \
        --functions_definition data/input/functions_definition.json \
        --input data/input/function_calling_tests.json \
        --output data/output/function_calling_results.json
```

### Debug

```bash
make debug
```

### Lint

```bash
make lint
```

---

## Algorithm Explanation

The constrained decoding strategy is schema-driven.

1. Build a prompt with available functions and the user request.
2. Generate the function name with token restrictions so only valid function-name continuations remain allowed.
3. For each parameter, constrain token selection by expected type:
         - `number` / `integer`: allow numeric-compatible tokens only.
         - `boolean`: allow tokens corresponding to `true`/`false`.
         - `string`: allow content tokens while preventing illegal structural closures.
4. At each step, mask invalid token logits with $-\infty$.
5. Select from valid tokens only.
6. Continuously validate partial JSON and finalize only valid objects.

This ensures the model cannot emit tokens that violate structural constraints.

---

## Design Decisions

- **Pydantic models:** strict validation for prompts and function definitions.
- **Typed pipeline:** `mypy`-friendly annotations across orchestration and decoding logic.
- **Separation of concerns:**
        - `src/loader.py` for I/O and validation,
        - `src/models.py` for schema,
        - `src/utils.py` for constrained generation,
        - `src/__init__.py` as entrypoint.
- **Fail-fast checks:** explicit errors for missing files and malformed input.

---

## Performance Analysis

This project targets:

- high function/argument accuracy,
- 100% parseable JSON output,
- stable behavior on malformed or edge-case inputs.

Actual throughput depends on hardware and model loading time. Reliability comes primarily from decoding constraints rather than model size.

---

## Challenges Faced

- Preventing invalid JSON while generating token-by-token.
- Handling type coercion safely for numbers and booleans.
- Keeping strict typing (`mypy`) in dynamic JSON-heavy logic.
- Managing partial outputs and stopping conditions without silent corruption.

---

## Testing Strategy

Validation includes:

- static checks: `flake8` + `mypy` through `make lint`,
- functional checks using provided input files,
- edge-case scenarios:
        - empty strings,
        - large numbers,
        - special characters,
        - ambiguous prompts,
        - multi-parameter functions,
        - boolean fields,
        - malformed/missing input files.

---

## Example Usage

```bash
uv run python -m src
```

with explicit paths:

```bash
uv run python -m src \
        --functions_definition data/input/functions_definition.json \
        --input data/input/function_calling_tests.json \
        --output data/output/function_calling_results.json
```

Expected output shape:

```json
[
        {
                "prompt": "What is the sum of 2 and 3?",
                "name": "fn_add_numbers",
                "parameters": {
                        "a": 2.0,
                        "b": 3.0
                }
        }
]
```

---

## Resources

### Technical References

- Python docs: https://docs.python.org/3/
- Pydantic docs: https://docs.pydantic.dev/
- `mypy` docs: https://mypy.readthedocs.io/
- `flake8` docs: https://flake8.pycqa.org/
- Hugging Face transformers docs: https://huggingface.co/docs/transformers/

### AI Usage Disclosure

AI tools were used for:

- drafting/refining documentation,
- suggesting type-hint improvements,
- lint/mypy troubleshooting guidance,
- wording and structure cleanup.

All generated content was reviewed, tested, and edited before integration.

---

## Submission and Peer Review Notes

Repository should contain:

- `src/`
- `llm_sdk/`
- `data/input/`
- `pyproject.toml`
- `uv.lock`
- `README.md`

`data/output/` is generated at runtime and should not be treated as source-of-truth code content.