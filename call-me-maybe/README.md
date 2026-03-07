# Knowledge: ✅ ❌

1. What is Function calling ?


# Project structure :
```
project/
│
├── src/
│   ├── __main__.py
│   ├── cli.py
│   ├── decoder.py
│   ├── schema.py
│   ├── generator.py
│   ├── io_handler.py
│
├── llm_sdk/   (copied)
├── data/input/
├── pyproject.toml
├── README.md
├── Makefile
```

# Explain:

1. I must build finite state machine FSM:
```
START
→ {
→ "prompt"
→ :
→ string
→ ,
→ "name"
→ :
→ one_of(valid_function_names)
→ ,
→ "parameters"
→ :
→ {
→ parameter_key
→ :
→ correct_type_value
→ ...
→ }
→ }
→ END
```

2. i must write the output to this path:
```sh
data/output/function_calling_results.json
```

3. test cases:
```
Very large numbers

Empty string

Special characters

Ambiguous prompts

Multiple parameters

Boolean parameters

Missing fields
```

4. this is the default:
```
logits → softmax → pick highest probability token
```
- this is what i should do:
```
logits → mask invalid tokens → softmax → pick valid token
```

# Questions:

1. what llm sdk mean ?

2. so i should customize the tokenizer of qwen3 to use specific vocabulary ?

3. explain this code :
```python
-float("inf")
```

4. ✅ what encoding mean ?


# rules:

1. All errors should be handled gracefully. Your program must never crash unexpectedly and must always provide clear error messages to the user.



2. By default, the program will read input files from the data/input/
directory and write output to the data/output/ directory. You
can optionally specify custom paths using the --input and --output
arguments. For example:

```sh
uv run python -m src
--functions_definition data/input/functions_definition.json
--input data/input/function_calling_tests.json
--output data/output/function_calls.json
```

# Steps:
1. after parsing Load and Validate JSON (Using Pydantic)

```sh
Parse CLI arguments
        ↓
Validate paths
        ↓
Load JSON files
        ↓
Validate using Pydantic
        ↓
Initialize LLM
        ↓
Load vocabulary
        ↓
For each prompt:
    Encode
    Constrained decode JSON
    Validate structure
        ↓
Collect results
        ↓
Write output JSON
```


# Architecture:
```sh
src/
 ├── main.py
 ├── loader.py
 ├── models.py
 └── decoder.py

models.py → Pydantic models

loader.py → JSON loading + validation

main.py → pipeline orchestration

decoder.py → constrained decoding logic
```