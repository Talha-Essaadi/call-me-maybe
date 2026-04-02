from llm_sdk import Small_LLM_Model
from typing import List
from .models import FunctionDefinition
import json


class JsonStructure():
    def __init__(self, output: List, functions_definition: FunctionDefinition, functions, prompts):
        self.output = output
        self.llm = Small_LLM_Model()
        self.prompts = prompts
        self.functions = {func.name: func for func in functions_definition}
        self.funcs_ids: list[list[int]] = [self.llm.encode(func.name)[0].tolist() for func in self.functions.values()]
        self.functions_data = functions
        self.generate_output()


    def generate_output(self):
        for user_request in self.prompts:
            prompt = self.generate_prompt(user_request, self.functions_data)
            name = '{"name": "'
            result= [prompt, name]
            func_name = self.generate_func_name(result)
            if func_name not in self.functions:
                raise ValueError(f"Function {func_name} not found in definitions.")
            func_data = [f for f in self.functions_data if f["name"] == func_name][0]
            result[0] = self.generate_prompt(user_request.prompt, func_data)
            result.append('", "parameters": ')
            self.generate_parameters(result, func_name)
            txt = "".join(result[1:])

            item = txt.strip()

            if not item.endswith("}}"):
                item += "}"

            try:
                item = json.loads(item)
                item = {
                "prompt": user_request.prompt,
                "name": func_name,
                "parameters": item["parameters"]
                }
                self.output.append(item)
                print("Generated output:", item)
            except json.JSONDecodeError:
                print("Skipped invalid JSON:", item)
            
            
    def generate_parameters(self, result, func_name):
        result.append('{')
        parameters = self.functions[func_name].parameters.items()
        for i, (param_name, param_def) in enumerate(parameters):
            result.append(f'"{param_name}": ')
            self.get_value(result, param_def.type, i + 1 == len(parameters))



    def get_value(self, result, param_type, end):
        value = ""
        while True:
            ids = self.llm.encode("".join(result) + value)[0].tolist()
            logits = self.llm.get_logits_from_input_ids(ids)
            next_token = self.constrained_decoding(logits, param_type, end)
            txt = self.llm.decode([next_token])
            # result.append(txt)
            print("#" * 50)
            data = "".join(result[1:] + [value] + [txt])
            print(data)
            print("#" * 50)
            print("token:", txt)
            print("param_type:", param_type)
            value = value.rstrip()
            if param_type == "string" and value.endswith('",'):
                print("breaking string")
                self.handle_value(result, value, param_type, "")
                break
            elif param_type != "string" and any(i == "," for i in txt):
                print("breaking")
                self.handle_value(result, value, param_type, txt)
                break
            try:
                tmp = txt.rstrip()
                if tmp.endswith("}}"):
                    txt = txt.rstrip()
                    print("ending with }}")
                    txt = txt[:-1]
                print("tmp:", txt)
                verify_json = "".join(result[result.index('", "parameters": ') + 1:] + [value] + [txt])
                print("verify_json:", verify_json)
                parsed = json.loads(verify_json)
                if isinstance(parsed, dict):
                    self.handle_value(result, value, param_type, txt)
                    break
            except json.JSONDecodeError:
                print("not a valid json yet")
                value = value + txt
                continue

    def handle_value(self, result, value, param_type, txt):
        if param_type == "number":
            token = float(value.strip())
        elif param_type == "boolean":
            token = value.strip().lower() == "true"
        else:
            token = value.strip()
        
        result.append(str(token) + txt)

    def constrained_decoding(self, logits, param_type, end):
        next_token = None
        if param_type == "number" or param_type == "integer":
            chars = [".", "-", ","]
            if end:
                chars = [".", "-", "}"]
            if param_type == "integer":
                chars.remove(".")

            while True:
                next_token = max(range(len(logits)), key=lambda i: logits[i])
                txt = self.llm.decode([next_token])
                if not txt.isdigit() and txt not in chars:
                    logits[next_token] = float("-inf")
                else:
                    break
        elif param_type == "string":
            chars = ["}"]
            if end:
                chars = [","]
            while True:
                next_token = max(range(len(logits)), key=lambda i: logits[i])
                txt = self.llm.decode([next_token])
                if txt in chars:
                    logits[next_token] = float("-inf")
                else:
                    break
        elif param_type == "boolean":
            allowed_tokens = set(self.llm.encode("true false"))
            while True:
                next_token = max(range(len(logits)), key=lambda i: logits[i])
                if next_token not in allowed_tokens:
                    logits[next_token] = float("-inf")
                else:
                    break
        return next_token


    def generate_func_name(self, result):
        func_logits = []
        while True:
            ids = self.llm.encode("".join(result))[0].tolist()
            logits = self.llm.get_logits_from_input_ids(ids)
            next_token = self.constrained_decoding_for_func_names(func_logits, logits)
            if not next_token:
                break

            result.append(self.llm.decode([next_token]))
            func_logits.append(next_token)
        return self.llm.decode(func_logits)


    def constrained_decoding_for_func_names(self, func_logits, logits):
        available_tokens = set()
        n = len(func_logits)
        for func_ids in self.funcs_ids:
            if len(func_ids) > n and func_ids[:n] == func_logits:
                for id_ in func_ids[n:]:
                    available_tokens.add(id_)

        if len(available_tokens) == 0:
            return None
        while True:
            next_token = max(range(len(logits)), key=lambda i: logits[i])
            if next_token not in available_tokens:
                logits[next_token] = float("-inf")
            else:
                break
        return next_token


    def generate_prompt(self, prompt, functions_data):
        return (
            f"Available functions:\n{functions_data}\n\n"
            f"User request: {prompt}\n\n"
            "Respond with a JSON object with keys 'name' and 'parameters'."
            )

