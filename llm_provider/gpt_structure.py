import json
import time
import os
import tiktoken
import openai
import numpy as np
import base64
import requests
openai_api_key=os.environ.get("OPENAI_API_KEY")
api_base = os.environ.get("OPENAI_API_BASE")

client = openai.OpenAI(
    api_key=openai_api_key, 
    base_url=api_base
    )

async_client = openai.AsyncOpenAI(
    api_key=openai_api_key,
    base_url=api_base
    )


def temp_sleep(seconds=0.1):
    time.sleep(seconds)


def GPT_4o_request(prompt, gpt_parameter):
    """
    Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
    server and returns the response.
    ARGS:
      prompt: a str prompt
      gpt_parameter: a python dictionary with the keys indicating the names of
                     the parameter and the values indicating the parameter
                     values.
    RETURNS:
      a str of GPT-4o-mini's response.
    """
    temp_sleep()
    try:
        if type(prompt) is dict:
            msg = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ]
        else:
            msg = [{"role": "user", "content": prompt}]
        completion = client.chat.completions.create(
            model=gpt_parameter["model"],
            messages=msg,
            max_tokens=gpt_parameter["max_tokens"],
            top_p=gpt_parameter["top_p"],
            frequency_penalty=gpt_parameter["frequency_penalty"],
            presence_penalty=gpt_parameter["presence_penalty"],
            temperature=gpt_parameter["temperature"],
        )
        return completion.choices[0].message.content
    except Exception as e:
        print("Exception: ", e)
        return "Error"

def text_embedding_request(prompt, 
                           model="text-embedding-ada-002",):
    return None

def text_embedding_request_v2(prompt, 
                           model="text-embedding-ada-002",):
    if type(prompt) is str:
        response = client.embeddings.create(
            model=model, 
            input=[prompt],
            )
        embedding = response.data[0].embedding
    elif type(prompt) is list:
        response = client.embeddings.create(
            model=model, 
            input=prompt,
            )
        embedding = [item.embedding for item in response.data]
    else:
        raise ValueError("The input must be a string or a list.")
    return embedding   
  
def safe_generate_response(
    prompt,
    gpt_parameter,
    repeat=5,
    fail_safe_response="error",
    func_validate=None,
    func_clean_up=None,
    verbose=False,
):
    if verbose:
        print(prompt)

    for i in range(repeat):
        curr_gpt_response = GPT_4o_request(prompt, gpt_parameter)
        if func_validate(curr_gpt_response, prompt=prompt):
            return func_clean_up(curr_gpt_response, prompt=prompt)
        if verbose:
            print("---- repeat count: ", i, curr_gpt_response)
            print(curr_gpt_response)
            print("~~~~")
    if curr_gpt_response == "Error":
        raise ValueError(
            "The GPT server is not responding. Please check your internet connection or the server status."
        )
    
    raise ValueError(fail_safe_response + f"The output is:\n {curr_gpt_response}")


def generate_prompt_role_play(curr_input, prompt_lib_file):
    """
    Takes in the current input (e.g. comment that you want to classifiy) and
    the path to a prompt file. The prompt file contains the raw str prompt that
    will be used, which contains the following substr: !<INPUT>! -- this
    function replaces this substr with the actual curr_input to produce the
    final promopt that will be sent to the GPT3 server.
    ARGS:
      curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                  INPUT, THIS CAN BE A LIST.)
      prompt_lib_file: the path to the promopt file.
    RETURNS:
      a str prompt that will be sent to OpenAI's GPT server.
    """
    if isinstance(curr_input, str):
        curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]
    prompt_lib_file = os.path.join(os.path.dirname(__file__), prompt_lib_file)

    f = open(prompt_lib_file, "r",encoding="utf-8")
    prompt = f.read()
    f.close()
    for count, i in enumerate(curr_input):
        prompt = prompt.replace(f"!<INPUT {count}>!", i)
    if "<commentblockmarker>###</commentblockmarker>" in prompt:
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
    return {"system": curr_input[0], "user": prompt.strip()}


def generate_prompt(curr_input, prompt_lib_file):
    """
    Takes in the current input (e.g. comment that you want to classifiy) and
    the path to a prompt file. The prompt file contains the raw str prompt that
    will be used, which contains the following substr: !<INPUT>! -- this
    function replaces this substr with the actual curr_input to produce the
    final promopt that will be sent to the GPT3 server.
    ARGS:
      curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                  INPUT, THIS CAN BE A LIST.)
      prompt_lib_file: the path to the promopt file.
    RETURNS:
      a str prompt that will be sent to OpenAI's GPT server.
    """
    if type(curr_input) is type("string"):
        curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]
    prompt_lib_file = os.path.join(os.path.dirname(__file__), prompt_lib_file)

    f = open(prompt_lib_file, "r", encoding="utf-8")
    prompt = f.read()
    f.close()
    for count, i in enumerate(curr_input):
        prompt = prompt.replace(f"!<INPUT {count}>!", i)
    if "<commentblockmarker>###</commentblockmarker>" in prompt:
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
    return prompt.strip()

    
def tokens_check_v0(text: str, model = "gpt-4o-mini"):
    model_to_encoding = {
        "gpt-4o-mini":"cl100k_base",
        "gpt-4o":"cl100k_base",
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
        "davinci": "p50k_base",
        "curie": "p50k_base",
        "babbage": "p50k_base",
        "ada": "p50k_base",
    }

    encoding_name = model_to_encoding[model]
    encoding = tiktoken.get_encoding(encoding_name)

    tokens = encoding.encode(text)
    return len(tokens), tokens

def tokens_check(text):
    if isinstance(text, str):
        text=text.strip().replace(",","").replace(".","")
        tokens=text.split(" ")
    elif isinstance(text, list):
        tokens=text
    else:
        raise ValueError("The input must be a string or a list.")
    return len(tokens), tokens

def format_set_as_table(data_set):
    data_list = list(data_set)
    lines = []
    for i in range(0, len(data_list), 15):
        line = "\t".join(str(item) for item in data_list[i:i+15])
        lines.append(line)
    table_str = "\n".join(lines)
    return table_str
  
def print_run_prompts(
    prompt_template=None,
    player_id=None,
    prompt=None,
    output=None,
):
    print(f"=== {prompt_template}")
    print("~~~ persona    ---------------------------------------------------")
    print(player_id, "\n")
    print("~~~ prompt     ---------------------------------------------------")
    print(prompt, "\n")
    print("~~~ output    ----------------------------------------------------")
    print(output, "\n")
    print("=== END ==========================================================")
    print("\n\n\n")
    

if __name__=="__main__":
    pass
    
    
    

    
    
    
    
