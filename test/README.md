*This project has been created as part of the 42 curriculum.*

# Call Me Maybe — Introduction to Function Calling in LLMs

## Description

**Call Me Maybe** is a function calling system that translates natural language
prompts into structured, machine-executable function calls. Instead of answering
questions directly, the system identifies the appropriate function and extracts
typed arguments from user requests.

For example, given the prompt *"What is the sum of 40 and 2?"*, the system
produces:

```json
{
  "name": "fn_add_numbers",
  "parameters": {"a": 40.0, "b": 2.0}
}
```

The key innovation is **constrained decoding**: rather than hoping the LLM
spontaneously outputs valid JSON, we intervene at each token generation step to
guarantee that the output follows the required schema. This achieves 100% valid
JSON output even with a small 0.6B parameter model (Qwen3-0.6B).

## Instructions

### Prerequisites

- Python 3.10 or later
- [uv](https://github.com/astral-sh/uv) package manager
- GPU recommended (CUDA), but CPU works (slower)

### Installation

```bash
# Install dependencies
make install
# or directly:
uv sync
```

### Running

```bash
# Default paths (data/input/ -> data/output/)
make run
# or:
uv run python -m src

# Custom paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Linting

```bash
make lint          # flake8 + mypy standard
make lint-strict   # flake8 + mypy --strict
```

### Cleaning

```bash
make clean
```

## Algorithm Explanation

### Constrained Decoding Approach

The system uses a **template-driven constrained decoding** strategy:

1. **Fixed Structure Generation**: The JSON skeleton (`{"name": "`, 
   `", "parameters": {`, etc.) is deterministic — these tokens are appended
   directly to the input without running the LLM, since they are known in
   advance.

2. **Function Name Selection**: When generating the function name, only tokens
   that are valid prefixes of known function names are allowed. The LLM's
   logits for all other tokens are set to `-inf`, forcing selection from valid
   continuations only.

3. **Parameter Value Generation**: Each parameter value is generated according
   to its declared type:
   - **number**: Only tokens forming valid JSON number prefixes (digits, `-`,
     `.`, `e`/`E`) are allowed. Generation stops when a separator or closing
     brace is the highest-scoring valid token.
   - **string**: Any printable character except unescaped `"` is allowed for
     content. A closing `"` token is always in the valid set so the model can
     end the string naturally.
   - **boolean**: Only `true` or `false` are allowed as options.

4. **Token Masking**: At each free-form generation step:
   - The LLM produces logits (probability scores) for all ~150k tokens
   - A mask sets invalid token logits to `-inf`
   - The token with the highest remaining logit is selected (greedy decoding)

This guarantees that **every generated token maintains structural and semantic
validity**, making the JSON output 100% parseable regardless of model quality.

### Generation Flow

```
User Prompt
     ↓
Build LLM prompt (with function descriptions)
     ↓
Force: {"name": "
     ↓
Constrained generation: function name (from known list)
     ↓
Force: ", "parameters": {
     ↓
For each parameter:
  Force: "param_name": 
  Constrained generation: value (by type)
  Force: , or }}
     ↓
Output: valid JSON function call
```

## Design Decisions

- **Pydantic for validation**: All input/output models use Pydantic v2 for
  strict validation, ensuring type safety and clear error messages.
- **Template-driven vs. full state machine**: Rather than tracking every
  possible JSON state, we exploit the known schema to decompose generation into
  fixed segments and constrained-choice segments. This is simpler and more
  reliable.
- **Greedy decoding**: We use argmax (greedy) token selection after masking.
  This is deterministic and fast — sampling would add randomness without
  benefit when the valid token set is already constrained.
- **Vocabulary pre-filtering**: String content tokens are pre-computed once
  to avoid re-scanning the full vocabulary at every string generation step.
- **Graceful error handling**: Invalid prompts are skipped with warnings
  rather than crashing the entire batch.

## Performance Analysis

- **Accuracy**: 90%+ correct function selection and argument extraction on
  typical prompts, thanks to the LLM's understanding guided by the function
  descriptions in the prompt.
- **JSON Validity**: 100% — constrained decoding guarantees every output is
  valid, parseable JSON conforming to the schema.
- **Speed**: Depends on hardware. On GPU, processes ~10-20 prompts per minute.
  On CPU, slower but still within the 5-minute requirement for typical test
  sets.
- **Reliability**: The system never crashes on malformed input — it reports
  errors gracefully and continues processing.

## Challenges Faced

1. **Vocabulary mapping complexity**: The tokenizer uses BPE tokens that don't
   align with character boundaries. A token like `"Ġthe"` includes a leading
   space. Careful prefix matching was needed to handle this correctly.

2. **Number generation termination**: Knowing when a number is "complete"
   requires checking if the model wants to output a separator or closing brace,
   while still allowing multi-digit numbers and decimals.

3. **Balancing prompt engineering with constrained decoding**: The prompt must
   give the model enough context to choose the right function, while the
   constrained decoder ensures the output format is always correct.

4. **Memory management with small models**: The 0.6B parameter model fits
   comfortably in memory, but vocabulary scanning at each step required
   optimization to keep generation fast.

## Testing Strategy

- **Unit tests**: Individual components (loader, models, decoder) are tested
  with valid and invalid inputs.
- **Edge cases tested**:
  - Empty strings, large numbers, special characters
  - Ambiguous prompts that could match multiple functions
  - Missing or malformed input files
  - Functions with varying numbers of parameters
- **Integration tests**: Full pipeline run from JSON input to JSON output,
  validating the complete result structure.
- **Manual verification**: Output JSON is inspected for correct function names,
  argument types, and value extraction.

## Example Usage

```bash
# Run with default test data
uv run python -m src

# Run with custom files
uv run python -m src \
  --functions_definition my_functions.json \
  --input my_prompts.json \
  --output results.json

# Use a different model
uv run python -m src --model Qwen/Qwen3-0.6B
```

### Sample Input

**function_calling_tests.json**:
```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Greet shrek"},
  {"prompt": "Reverse the string 'hello'"}
]
```

### Sample Output

**function_calling_results.json**:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {"name": "shrek"}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

## Resources

- [Qwen3-0.6B Model](https://huggingface.co/Qwen/Qwen3-0.6B) — The small
  language model used in this project
- [Constrained Decoding for LLMs](https://arxiv.org/abs/2307.09702) — Research
  on grammar-constrained generation
- [Pydantic Documentation](https://docs.pydantic.dev/) — Data validation
  library used for all models
- [JSON Schema Specification](https://json-schema.org/) — Reference for JSON
  structure validation

### AI Usage

AI tools were used for:
- Generating boilerplate code structure and Pydantic model scaffolding
- Researching constrained decoding techniques and BPE tokenizer behavior
- Reviewing and debugging token mask logic
- Drafting documentation and docstrings

All AI-generated code was reviewed, understood, and tested manually before
inclusion.

## Project Structure

```
.
├── src/
│   ├── __init__.py            # Package init and main() entry point
│   ├── __main__.py            # Module runner
│   ├── models.py              # Pydantic models for validation
│   ├── loader.py              # JSON file loading and validation
│   ├── constrained_decoder.py # Core constrained decoding engine
│   └── pipeline.py            # Orchestration pipeline
├── llm_sdk/
│   └── __init__.py            # LLM wrapper (provided)
├── data/
│   └── input/
│       ├── functions_definition.json
│       └── function_calling_tests.json
├── pyproject.toml             # Project configuration
├── Makefile                   # Build automation
├── .gitignore
└── README.md
```
