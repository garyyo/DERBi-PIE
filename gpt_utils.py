import hashlib
import json
import os
import re
import time
from io import StringIO

import numpy as np
import openai
import pandas as pd
import pyperclip


"""
author - anton vinogradov

a collection of useful functions used to query gpt. Most are self explanatory, but note that query_gpt and query_gpt_fake are more or less interchangeable, but
that the fake version requires the user to paste into chatgpt and copy its response. I also make use of breakpoint() for the fake version as the output for gpt
often raises edge cases. Its literally oops all edge cases, you have been warned.

Also I use a directory called gpt_caches to store the responses for gpt. This is so I can stop execution at any point and resume at a later date. These are
cached on the prompt digest, so any change in prompt no matter how minor will make a new query. These caches are not included in the repo though, only the
placeholder file.

Godspeed.
"""


def get_digest(input_string):
    return hashlib.sha256(input_string.encode('utf-8')).hexdigest()


def get_models():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    models = openai.Model.list()
    breakpoint()
    pass


def backup_response(prompt, response):
    digest = get_digest(prompt)
    with open(f"gpt_chaches/{digest}.json", "w", encoding="utf-8") as fp:
        json.dump(response, fp, indent=2)


def load_response(prompt):
    digest = get_digest(prompt)
    if os.path.exists(f"gpt_chaches/{digest}.json"):
        with open(f"gpt_chaches/{digest}.json", "r", encoding="utf-8") as fp:
            response = json.load(fp)
        return response, digest
    return None, digest


def delete_response(prompt):
    digest = get_digest(prompt)
    if os.path.exists(f"gpt_chaches/{digest}.json"):
        os.remove(f"gpt_chaches/{digest}.json")
    return


def query_gpt(prompt):
    completion, digest = load_response(prompt)
    if completion is None:
        print("existing not found - NEW QUERY", digest[:20])
        openai.api_key = os.getenv("OPENAI_API_KEY")
        model = "gpt-3.5-turbo"
        # model = "gpt-4"
        completion = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        backup_response(prompt, completion)
        time.sleep(1)
    else:
        print("existing found - REUSING", digest[:20])

    return completion


def query_gpt_fake(prompt):
    completion, digest = load_response(prompt)
    if completion is None:
        print("existing not found - NEW QUERY", digest[:20])
        pyperclip.copy(prompt)
        breakpoint()
        text = pyperclip.paste()
        completion = {"choices": [{"message": {"content": text}}]}
        backup_response(prompt, completion)
    else:
        print("existing found - REUSING", digest[:20])
    return completion


def process_gpt(completion, headers):
    text = completion["choices"][0]["message"]["content"]
    # to keep consistent between windows/linux we remove carriage return chars
    text = text.replace("\r", "")

    # sometimes gpt likes to include a """ and that's not allowed
    text = text.replace('"'*3, '"'*2).replace("'"*3, "'"*2)

    # sometimes gpt includes the header text even though I do not want it to, so here we test to see if it did to inform pd.read_csv
    includes_header = set([cell.strip(' "\'') for cell in text.split("\n")[0].split(",")]) == set(headers)

    # remove trailing commas, GPT seems to love to sometimes include those.
    text = "\n".join([line.strip(",") for i, line in enumerate(text.split("\n"))])
    df = pd.read_csv(
        StringIO(text),
        encoding="utf-8",
        header=0 if includes_header else None,
        names=headers,
        quotechar='"',
        sep=',',
        skipinitialspace=True
    )

    # sometimes gpt numbers the entries even though I tell it not to.
    # To combat that I look in the first column and try to strip out any
    # df[headers[0]] = df[headers[0]].apply(lambda x: re.sub(r'^\s*\d+\.', '', x).strip())

    # sometimes gpt uses a dash for blanks, these should be removed too
    df = df.replace("-", "")

    df = df.fillna("")
    return df


def get_text_digest(completion):
    return get_digest(completion["choices"][0]["message"]["content"])


if __name__ == '__main__':
    get_models()
    pass
