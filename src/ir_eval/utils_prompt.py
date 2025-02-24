import re
import os
from typing import Union
from openai import OpenAI
from jinja2 import Template

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def _get_system_message(text):
    pattern = r"<\|im_start\|>system\s*(.*?)\s*<\|im_end\|>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None

def _get_user_message(text):
    pattern = r"<\|im_start\|>user\s*(.*?)\s*<\|im_end\|>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None

def load_prompt_text(prompt_template_path):
    with open(prompt_template_path) as file_:
        prompt_template_text = file_.read()
        return prompt_template_text
    

def preprocess_prompt(template: Union[str, Template], input_dict):
    if isinstance(template, str):
        template = Template(template) 
    prompt = template.render(input_dict)
    #print(prompt)
    system_message = _get_system_message(prompt)
    user_message = _get_user_message(prompt)
    return {"system": system_message, "user": user_message}

    
def eval_prompt(prompt_info_dict, model, temperature=0.0):
    system_message = prompt_info_dict['system']
    user_message = prompt_info_dict['user']
    messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    #print(messages)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    output = response.choices[0].message.content
    return output

def tag_parser(text, tag, add_start_tag=False):
    
    if add_start_tag:
        if f"<{tag}>" not in text:
            text = f"<{tag}>" + text
    
    pattern = fr"<{tag}>(.*?)</{tag}>"  # Dynamic tag in regex
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip().replace("\n","") for match in matches]