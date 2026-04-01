from llm_sdk import Small_LLM_Model
from typing import List
from .models import FunctionDefinition



class JsonStructure():
    def __init__(self, output: List, functions_definition: FunctionDefinition, prompts):
        self.output = output
        self.llm = Small_LLM_Model()
        self.prompts = prompts
        self.functions = {func.name: func for func in functions_definition}
        self.funcs_ids: list[list[int]] = [self.llm.encode(func.name)[0].tolist() for func in self.functions.values()]
        self.generate_output()


    def generate_output(self):
        for user_request in self.prompts:
            print("#User request:", user_request)
            prompt = self.generate_prompt(user_request)
            name = '{"name": "'
            result= [prompt, name]
            func_name = self.generate_func_name(result)
            if func_name not in self.functions:
                raise ValueError(f"Function {func_name} not found in definitions.")
            result.append('", "parameters": ')
            self.generate_parameters(result, func_name)
            print("".join(result[1:]))
        self.output = "".join(result[1:])

    def generate_parameters(self, result, func_name):
        result.append('{')
        parameters = self.functions[func_name].parameters.items()
        for i, (param_name, param_def) in enumerate(parameters):
            print("param_name", param_name)
            print("param_def", param_def)
            result.append(f'"{param_name}": ')
            self.get_value(result, param_def.type, i + 1 == len(parameters))



    def get_value(self, result, param_type, end):
        while True:
            ids = self.llm.encode("".join(result))[0].tolist()
            logits = self.llm.get_logits_from_input_ids(ids)
            next_token = self.constrained_decoding(logits, param_type, end)
            # next_token = logits.index(max(logits))
            txt = self.llm.decode([next_token])
            result.append(txt)
            if txt in ["}", ","] or "}" in txt:
                break
            print("###########################")
            # print("".join(result))
            print(txt)
            print("###########################")
            
        return "".join(result)

    def constrained_decoding(self, logits, param_type, end):
        print("param_type", param_type)
        next_token = None
        if param_type == "number":
            chars = [".", "-", ","]
            if end:
                chars = [".", "-", "}"]

            while True:
                next_token = logits.index(max(logits))
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
                next_token = logits.index(max(logits))
                txt = self.llm.decode([next_token])
                if txt in chars:
                    logits[next_token] = float("-inf")
                else:
                    break
        return next_token


    def generate_func_name(self, result):
        func_logits = []
        while True:
            ids = self.llm.encode("".join(result))[0].tolist()
            logits = self.llm.get_logits_from_input_ids(ids)
            logits = self.constrained_decoding_for_func_names(func_logits, logits)
            if not logits:
                break
            next_token = logits.index(max(logits))

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
        for i in range(len(logits)):
            if i not in available_tokens:
                logits[i] = float("-inf")
        return logits


    def generate_prompt(self, prompt):
        return (
            f"Available functions:\n{self.functions}\n\n"
            f"User request: {prompt}\n\n"
            "Respond with a JSON object with keys 'name' and 'parameters'."
            )

