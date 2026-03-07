import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, ValidationError, Field
from llm_sdk import Small_LLM_Model  # نسخة محلية من الـ SDK

# # ----------------------------
# # 1. Pydantic Models
# # ----------------------------
# class FunctionParameter(BaseModel):
#     type: str

# class FunctionDefinition(BaseModel):
#     name: str
#     description: str
#     parameters: Dict[str, FunctionParameter]
#     returns: Dict[str, str]

# class PromptResult(BaseModel):
#     prompt: str
#     fn_name: str
#     args: Dict[str, Any]

# # ----------------------------
# # 2. CLI Arguments
# # ----------------------------
# parser = argparse.ArgumentParser()
# parser.add_argument("--input", type=str, default="data/input")
# parser.add_argument("--output", type=str, default="data/output/function_calling_results.json")
# args = parser.parse_args()

# input_dir = Path(args.input)
# output_file = Path(args.output)

# # ----------------------------
# # 3. Validate paths & load JSON
# # ----------------------------
# function_def_path = input_dir / "function_definitions.json"
# prompts_path = input_dir / "function_calling_tests.json"

# if not function_def_path.exists() or not prompts_path.exists():
#     raise FileNotFoundError("Required input JSON files are missing.")

# with open(function_def_path, "r") as f:
#     raw_functions = json.load(f)

# with open(prompts_path, "r") as f:
#     raw_prompts = json.load(f)

# ----------------------------
# 4. Validate JSON using Pydantic
# ----------------------------
# functions = [FunctionDefinition(**fd) for fd in raw_functions]

# ----------------------------
# 5. Initialize LLM
# ----------------------------
llm = Small_LLM_Model()
vocab_path = llm.get_path_to_vocab_file()
print(f"Vocabulary JSON path: {vocab_path}")
with open(vocab_path, "r") as f:
    vocabulary = json.load(f)

# ----------------------------
# 6. Processing Prompts
# ----------------------------
results: List[PromptResult] = []

for prompt in raw_prompts:
    try:
        # Encode prompt
        input_ids = llm.encode(prompt)

        # Constrained decode (pseudo-code, implement according to your decoding logic)
        decoded_json_str = llm.constrained_decode(
            input_ids=input_ids,
            vocabulary=vocabulary,
            allowed_functions=[f.name for f in functions]
        )

        # Validate JSON structure
        decoded_obj = json.loads(decoded_json_str)
        result = PromptResult(**decoded_obj)
        results.append(result)

    except (ValidationError, json.JSONDecodeError) as e:
        print(f"Error processing prompt '{prompt}': {e}")

# ----------------------------
# 7. Write output JSON
# ----------------------------
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, "w") as f:
    json.dump([r.dict() for r in results], f, indent=2)

print(f"Processed {len(results)} prompts. Results saved to {output_file}")