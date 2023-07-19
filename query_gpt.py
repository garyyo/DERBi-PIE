import hashlib
import json
import os
from io import StringIO

import numpy as np
import openai
import pandas as pd
import pyperclip


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
        json.dump(response, fp)


def load_response(prompt):
    digest = get_digest(prompt)
    if os.path.exists(f"gpt_chaches/{digest}.json"):
        with open(f"gpt_chaches/{digest}.json", "r", encoding="utf-8") as fp:
            response = json.load(fp)
        return response
    return None


def query_gpt(prompt):
    completion = load_response(prompt)
    if completion is None:
        print("new query")
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
    return completion


def query_gpt_fake(prompt):
    completion = load_response(prompt)
    if completion is None:
        print("existing not found - NEW QUERY")
        pyperclip.copy(prompt)
        breakpoint()
        text = pyperclip.paste()
        completion = {"choices": [{"message": {"content": text}}]}
        backup_response(prompt, completion)
    else:
        print("existing found - REUSING")
    return completion


def process_gpt(completion):
    text = completion["choices"][0]["message"]["content"]
    text = text.replace("|", "")
    text = "\n".join([line.strip(",") for i, line in enumerate(text.replace("\r", "").split("\n"))])
    # number the lines
    # text = "\n".join([f"{i},{line}" for i, line in enumerate(text.split("\n"))])
    df = pd.read_csv(StringIO(text), encoding="utf-8", header=None, names=["language", "reflex", "meaning", "notes"], quotechar='"', sep=',', skipinitialspace=True)
    return df


if __name__ == '__main__':
    get_models()
    pass
