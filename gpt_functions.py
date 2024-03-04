"""
This is just a more updated version of gpt_utils.py that I have been using in other projects. I don't want to update the old usages so I am keeping it for
compat but this one is more robust and should be used going forward (and honestly this should have been spun off into its own actual library at this point).
"""
import hashlib
import json
import os
import re
import tempfile
import time
from io import StringIO

import clevercsv
import openai
import pandas as pd
import pyperclip

g_model = "gpt-3.5-turbo"
# g_model = "gpt-3.5-turbo-16k"
# g_model = "gpt-3.5-turbo-0301"
# g_model = "gpt-4"
total_cost = 0

seed = ""


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def set_seed(new_seed):
    global seed
    seed = new_seed


def get_digest(input_string):
    global seed
    return hashlib.sha256(f"{input_string}{seed}".encode('utf-8')).hexdigest()


def print_cost():
    print(f"${total_cost:0.4f}")


def reset_cost():
    global total_cost
    total_cost = 0


def get_cost():
    return total_cost


def get_models():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    models = openai.Model.list()
    print(models)
    pass


def backup_response(prompt_string, messages, prompt_tokens, response_tokens, total_tokens, total_prompts):
    digest = get_digest(str(prompt_string))
    with open(f"gpt_caches/{digest}.json", "w", encoding="utf-8") as fp:
        json.dump({
            "messages": messages,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_prompts": total_prompts,
            "total_tokens": total_tokens
        }, fp, indent=2)


def load_response(prompts):
    digest = get_digest(str(prompts))
    if os.path.exists(f"gpt_caches/{digest}.json"):
        with open(f"gpt_caches/{digest}.json", "r", encoding="utf-8") as fp:
            response = json.load(fp)
        return response, digest
    return None, digest


def load_response_digest(digest):
    if os.path.exists(f"gpt_caches/{digest}.json"):
        with open(f"gpt_caches/{digest}.json", "r", encoding="utf-8") as fp:
            response = json.load(fp)
        return response, digest
    return None, digest


def get_proper_digest(model=g_model, system_prompt=None, user_prompts=()):
    if type(user_prompts) == str:
        user_prompts = [user_prompts]

    prompt_string = f"{model}-{system_prompt}-{user_prompts}"
    digest = get_digest(prompt_string)
    return digest, prompt_string


def calculate_cost(model, in_tokens, out_tokens=0):
    # given a model and a number of tokens, calculate the estimated cost in USD
    # gpt4: 0.03 per 1k tokens input, 0.06 per 1k tokens output
    # gpt3: 0.0015 per 1k tokens, 0.002 per 1k tokens output
    model_pricing = {
        "gpt-3.5-turbo": {"in": 0.0015, "out": 0.002},
        "gpt-3.5-turbo-16k": {"in": 0.003, "out": 0.004},
        "gpt-3.5-turbo-0301": {"in": 0.0015, "out": 0.002},
        "gpt-4": {"in": 0.03, "out": 0.06},
    }
    cost = in_tokens * model_pricing[model]["in"] + out_tokens * model_pricing[model]["out"]
    return cost/1000


def query_gpt(user_prompts=(), system_prompt=None, model=g_model, note=None, no_print=False, fake=False, bypass_cache=False):
    global total_cost
    # I also want to be able to use this with just a single string
    if type(user_prompts) == str:
        user_prompts = [user_prompts]

    # notes are just so I know which query is running at a glance
    digest, prompt_string = get_proper_digest(model, system_prompt, user_prompts)
    response, digest = load_response_digest(digest)
    note_text = f"starting with '{user_prompts[0][:40]}...'" if note is None else f"({note})"
    query_desc = f"{digest[:20]}..., {len(user_prompts): >3} prompt(s), {note_text} -> {model}"

    # load a cached response if it exists, and if we are not bypassing it
    if response is not None and not bypass_cache:
        prompt_cost = calculate_cost(model, response["prompt_tokens"], response["response_tokens"])
        total_cost += prompt_cost
        if not no_print:
            print(
                f'Found existing: {query_desc} | used {response["total_prompts"]} prompts and {response["total_tokens"]} tokens for ${prompt_cost:02.6f}',
                flush=True
            )
        return response["messages"], response["messages"][-1]

    if not no_print:
        print(f"Generating new: {query_desc} ", end="", flush=True)

    # load openai key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # stats
    prompt_tokens = 0
    response_tokens = 0
    total_tokens = 0
    total_prompts = 0

    # build the initial messages object
    messages = [{"role": "system", "content": system_prompt}] if system_prompt is not None else []

    timer_start = time.time()
    completion = None
    for prompt in user_prompts:
        if not no_print:
            print(total_prompts, end=",")
        # append the next prompt to the messages
        messages.append({"role": "user", "content": prompt})
        if fake:
            pyperclip.copy(prompt)
            breakpoint()
        # contact openai and get a response
        completion = try_gpt(
            openai.ChatCompletion.create,
            model=model,
            messages=messages,
        )
        # extract the response from the completion
        response = completion["choices"][-1]["message"]
        # append the response to the messages
        messages.append(response)
        # add to stats
        prompt_tokens += completion.usage.prompt_tokens
        response_tokens += completion.usage.completion_tokens
        total_prompts += 1
        total_tokens += completion.get("usage", {}).get("total_tokens", 0)
        pass
    timer_end = time.time()

    prompt_cost = calculate_cost(model, prompt_tokens, response_tokens)
    total_cost += prompt_cost
    if not no_print:
        print(f" | used {total_prompts} prompts and {total_tokens} tokens for ${prompt_cost:06.06f}, {timer_end - timer_start:05.02f}s", flush=True)
    backup_response(prompt_string, messages, prompt_tokens, response_tokens, total_tokens, total_prompts)
    return messages, messages[-1]


def get_cached(user_prompts=(), system_prompt=None, model=g_model):
    # I also want to be able to use this with just a single string
    if type(user_prompts) == str:
        user_prompts = [user_prompts]

    # load a cached response if it exists
    digest, prompt_string = get_proper_digest(model, system_prompt, user_prompts)
    response, digest = load_response_digest(digest)

    if response is not None:
        return response["messages"], response["messages"][-1], digest

    return None, None, digest


# automatic retries because I am tired of openai timing out.
# todo: pass a pre bound function rather than the function and its parameters
def try_gpt(call, *args, **kwargs):
    total_attempts = 10
    output = None
    success = False
    err = None
    for attempt_num in range(total_attempts):
        try:
            time.sleep(1)
            output = call(*args, **kwargs)
            success = True
            break
        except (openai.error.ServiceUnavailableError, openai.error.APIError, openai.error.Timeout) as err:
            print(f"\n---===>>> Could not connect, waiting 10 seconds and trying again... ({attempt_num+1}/{total_attempts}) <<<===---")
            time.sleep(10)
            continue
    if not success:
        raise err
    return output


def extract_code_block(text):
    code_block = re.search(r'```(.*?)```', text, re.DOTALL)
    return code_block.group(1).strip() if code_block else text


def get_role_messages(messages, role="assistant"):
    return [message for message in messages if message["role"] == role]


def get_last_content(messages):
    return messages[-1]["content"]


def get_all_content(messages):
    return [message["content"] for message in messages]


def csv_to_df(text, headers):
    # to keep consistent between windows/linux we remove carriage return chars
    text = text.replace("\r", "")

    # sometimes gpt includes the header text even though I do not want it to, so here we test to see if it did to inform pd.read_csv
    includes_header = set([cell.strip(' "\'') for cell in text.split("\n")[0].split(",")]) == set(headers)

    # remove trailing commas, GPT seems to love to sometimes include those.
    text = "\n".join([line.strip(",") for i, line in enumerate(text.split("\n"))])

    # replace escaped " characters with fancy quote characters because it helps
    # todo: undo this in the df
    text = text.replace(r"\"", "“")


    # create the df, but drop any row that is all nan.
    try:
        df = pd.read_csv(
            StringIO(text),
            encoding="utf-8",
            header=0 if includes_header else None,
            names=headers,
            quotechar='"',
            sep=',',
            skipinitialspace=True
        ).dropna(how='all')
    except pd.errors.ParserError as err:
        # replace [quote comma space quote] sequence with something that is not used, then replace all the other commas with something unused
        unused1 = "[[[[[[[[["
        unused_comma = "，"
        text = text.replace("\", \"", unused1)
        text = text.replace(",", unused_comma)
        text = text.replace(unused1, "\", \"")
        df = pd.read_csv(
            StringIO(text),
            encoding="utf-8",
            header=0 if includes_header else None,
            names=headers,
            quotechar='"',
            sep=',',
            skipinitialspace=True
        ).dropna(how='all')

    # todo: undo the weird stuff I did earlier

    # sometimes gpt numbers the entries even though I tell it not to.
    # To combat that I look in the first column and try to strip out any
    # df[headers[0]] = df[headers[0]].apply(lambda x: re.sub(r'^\s*\d+\.', '', x).strip())

    # sometimes gpt uses a dash for blanks, these should be removed too
    df = df.replace("-", "")

    df = df.fillna("")
    return df


if __name__ == '__main__':
    get_models()
    pass
