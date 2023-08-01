import json
import re
from collections import Counter
from unidecode import unidecode

import pandas as pd
import pyperclip
from io import StringIO


"""
author - anton vinogradov

The only purpose of this file is to process the scraped pokorny site: https://indo-european.info/pokorny-etymological-dictionary/

Since that is not being used as a good source of pokorny data, this script should not be run and only exists to document what has been done. Seriously, don't 
use this.

Godspeed.
"""


def process_pos(processed_pokorny, pos_file):
    corrections = {
        "adv": "adverb",
        "adj": "adjective",
        "interjections": "interjection",
        "prep": "preposition",
    }
    with open(pos_file, "r") as fp:
        lines = fp.readlines()
    index_to_pos = {
        int(line.split(":")[0])-1:
            [corrections.get(pos.strip(), pos.strip()) for pos in sorted(line.split(":")[-1].lower().strip().replace(",", ";").replace("/", ";").split(";"))]
        for line in lines
    }
    # for each entry, get the new pos, if a new one doesn't exist use the old one, if an old one doesn't exist make it blank
    pos_pokorny = [{**entry, "pos": index_to_pos.get(i, entry.get("pos", []))} for i, entry in enumerate(processed_pokorny)]
    return pos_pokorny


a = {
    "adv": "adverb",
    "adj": "adjective",
    "interjections": "interjection",
    "prep": "preposition",
    "number": "numeral",
}


def remove_non_english_chars(text, allowed_chars=""):
    if "ə" not in allowed_chars:
        text = text.replace("ə", "e")
    cleaned_text = unidecode(text)
    cleaned_text = ''.join([c for c in cleaned_text if c.isalpha() or c in allowed_chars])
    return cleaned_text


def output_pos(processed_pokorny):
    newline = '\n'
    text_lines = [
        f"{i+1} - {'; '.join(entry['root']).replace(newline, ' ')}" \
        f"\n\tEnglish Meaning:\t{'; '.join(entry['meaning'])}" \
        f"\n\t German Meaning:\t{'; '.join(entry['other_meaning']['german'])}" \
        f"\n\t     Tagged POS:\t{'; '.join(entry['pos'])}\n\n"
        for i, entry in enumerate(processed_pokorny)
    ]
    with open("data_pokorny/pokorny_pos_output.txt", "w", encoding="utf-8") as fp:
        fp.writelines(text_lines)
    pass


# the prompt used
"""
I am going to give you a numbered set of word definitions in german. Tell me the part of speech that the word is in english, based solely on the definition. 
If there are multiple that you think are reasonable, give all that are applicable. Give your answer in the form "number: part of speech". 
Do not include any other text, do not include the meaning either.
"""


def clipboard(processed_pokorny):
    definitions = [f'{i + 1}: {"; ".join(entry["other_meaning"]["german"])}' for i, entry in enumerate(processed_pokorny) if "; ".join(entry["meaning"]) != ""]
    for i in range(0, len(definitions), 100):
        pyperclip.copy("\n".join(definitions[i:i+100]))
        breakpoint()


def compare_pos(pokorny_1, pokorny_2):
    pos_list_1 = [entry["pos"] for i, entry in enumerate(pokorny_1)]
    pos_list_2 = [entry["pos"] for i, entry in enumerate(pokorny_2)]
    pos_diff = pd.DataFrame([
        {
            "index": i,
            1: p1,
            2: p2,
        }
        for i, (p1, p2) in enumerate(zip(pos_list_1, pos_list_2)) if set(p1) != set(p2) and len(p1) > 0 and len(p2) > 0
    ])
    return pos_diff


def merge_pos(pokorny_1, pokorny_2):
    return [{**entry1, "pos": sorted(set(entry1["pos"] + entry2["pos"]))}for entry1, entry2 in zip(pokorny_1, pokorny_2)]


def generate_ids(pokorny):
    # get all the proto-ids, which really just removes the weird chars (even though they display just fine in the browser, might be difficult to paste or something)
    stripped_roots = [remove_non_english_chars(entry["root"][0]) for entry in pokorny]
    stripped_meanings = [remove_non_english_chars(sorted(entry["meaning"])[0].replace(" ", "-"), allowed_chars="-") if len(entry["meaning"]) else "unknown" for entry in pokorny]
    proto_id = [f"{root}_{meaning}" for root, meaning in zip(stripped_roots, stripped_meanings)]
    # put a number at the end so any duplicates are removed, and add them to the entries
    processed_pokorny = [{**entry, "entry_id": f"{entry_id}_{proto_id[:i].count(entry_id)+1}"} for i, (entry_id, entry) in enumerate(zip(proto_id, pokorny))]
    return processed_pokorny


def common():
    with open("data_pokorny/pokorny_scraped.json", "r", encoding="utf-8") as fp:
        pokorny = json.load(fp)

    # for each entry in the raw pokorny, we need to build out entries that are well laid out
    common_pokorny = [
        {
            "root": [root.strip() for roots in entry["root"] for root in roots.split(",")],
            "meaning": [meaning.strip() for meanings in entry["English meaning"] for meaning in meanings.split(",")],
            "other_meaning": {"german": [meaning.strip() for meanings in entry["German meaning"] for meaning in meanings.replace("`", "'").replace("'", "").split(",")]},
        }
        for entry in pokorny
    ]

    # we prompt chatgpt here with:
    # 'I am going to give you a numbered set of word definitions. Tell me the part of speech that the word is, based solely on the definition. If there are multiple that you think are reasonable, give all that are applicable. Give your answer in the form "number: part of speech". Do not include any other text, do not include the meaning either.'
    # then pasting in what clipboard gives you and putting the chat output in pokorny_pos.txt
    # clipboard(processed_pokorny)

    # the output of gpt is not deterministic afaik, so processing the output may require changing the processing func.
    # common_pokorny = process_pos(common_pokorny, "data_pokorny/gpt_pos.txt")
    # common_pokorny = process_pos(common_pokorny, "data_pokorny/gpt_german_pos.txt")

    # we merge the automated ones, but overwrite with the human one (only on entries that the human corrected).
    common_pokorny = merge_pos(process_pos(common_pokorny, "data_pokorny/gpt_pos.txt"), process_pos(common_pokorny, "data_pokorny/gpt_german_pos.txt"))
    common_pokorny = process_pos(common_pokorny, "data_pokorny/human_pos.txt")

    # compare two pos tagged pokorny lists
    # compare_pos(process_pos(common_pokorny, "data_pokorny/human_pos.txt"), process_pos(common_pokorny, "data_pokorny/gpt_german_pos.txt"))

    # output to file for double-checking by human
    # output_pos(processed_pokorny)

    # we also have a human double-checking and correcting those tags, which takes precedence
    # processed_pokorny = process_pos(processed_pokorny, "data_pokorny/human_pos.txt")

    # generate IDs for each. There are just human-readable ways to refer to the word that can be put into a url and other things.
    common_pokorny = generate_ids(common_pokorny)

    # output the processed to json, so we can import into a db or something
    with open("data_pokorny/pokorny_processed.json", "w", encoding="utf-8") as fp:
        json.dump(common_pokorny, fp)
    # breakpoint()
    pass


def extract_material_forms(material_line_splits):
    extracted_forms = []
    for split in material_line_splits:
        # we often get empty lines at the end, this just ensures that we skip those
        if len(split.strip()) == 0:
            continue

        # most forms are italicized, but not all
        forms = list(re.findall(r'\\\\(.*?)\\\\', split))

        if len(forms) != 0:
            # most definitions follow the form
            definition = list(re.findall(r'\\\\.*?\\\\(.*)', split))[0].strip()
        else:
            # if its empty assume we failed and that this needs some further looking at
            breakpoint()

        # add onto the entry
        extracted_forms.append({
            "forms": forms,
            "definition": definition,
        })
    return extracted_forms


def pokorny_table():
    with open("data_pokorny/pokorny_scraped.json", "r", encoding="utf-8") as fp:
        pokorny = json.load(fp)

    # todo: process the material field
    #  formatting stuff:
    #   \\words\\ = italicized
    #   __words__ = underlined
    #  breakdown of a single line:
    #   1. mentions the language (presumably), tends to look like ["Ai.", "gr.", "lat.", ...]
    #   2. a generally italicized (but not always) word (presumably called a "reconstruction" or "language form")
    #    - there may be multiple words here
    #   3. the definition
    #   4. semicolon ";" marking the end of definitions
    #   5. 2-4 may repeat.
    #  extracted fields:
    #   - language: string
    #   - forms object
    #     - forms: list of forms (strings)
    #     - definition: string

    for entry in pokorny:
        processed_materials = []
        for material_line in entry["Material"]:
            # split up each ; separated entry for this specific langauge
            material_line_splits = material_line.split(";")

            # separate off the langauge, and edit the first entry to remove it
            language = material_line_splits[0].split(" ")[0]
            material_line_splits[0] = " ".join(material_line_splits[0].split(" ")[1:])

            extracted_forms = extract_material_forms(material_line_splits)
            processed_materials.append({
                "language": language,
                "forms": extracted_forms
            })

        # print(entry["root"])
        # print()
        # for line in entry["Material"]:
        #     print(line.replace("\\\\", "\\"))
        # print()
        #
        # breakpoint()
        pass


def get_base_letters(alt_letters):
    unique_letters = sorted(set([remove_non_english_chars(letter)[0].lower() for letter in alt_letters]))
    base_letters = {
        unique_letter:
            [letter for letter in a if unique_letter == remove_non_english_chars(letter)[0].lower() and letter != unique_letter]
        for unique_letter in unique_letters
    }
    return base_letters


if __name__ == '__main__':
    # print(get_base_letters(["p", "pʰ", "b", "bʰ", "t", "tʰ", "d", "dʰ", "ḱ", "ḱʰ", "ǵ", "ǵʰ", "k", "kʰ", "g", "gʰ", "kʷ", "kʷʰ", "gʷ", "gʷʰ", "T", "K", "k(')", "ǵ(ʰ)", "g(')ʰ", "g(')", "k(ʷ)", "g(ʷ)ʰ", "g(ʷ)", "h₁", "h₂", "h₃", "H, hₓ", "s", "z", "F", "w, u̯", "y, i̯", "G", "r", "l", "L", "m", "n", "N", "r̥", "l̥", "m̥", "n̥", "i", "u", "a", "e", "o", "ā", "ē", "ō", "á", "é", "ó", "ā́", "ḗ", "ṓ", "í", "ī", "ī́", "ú", "ū", "ū́", "ə"]))
    pokorny_table()
    # common()
    pass
