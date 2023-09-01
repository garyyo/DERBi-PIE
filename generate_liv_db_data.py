import json
import regex as re

import pandas as pd
from tqdm import tqdm

import translate_meaning_gpt
from gpt_utils import *


def process_language_form_and_translation(line):
    entry = {"lang": {"abbr": "", "questionable": False}, "reflex": "", "meaning": None, "notes": []}
    # Todo: the whole thing needs to be extracted from the gpt corrected csv
    return entry


def get_reflex_entries(group):
    entries = []
    for i, row in group.iterrows():
        if row.category == "root":
            continue
        # temp call
        process_language_form_and_translation(row.language_form_and_translation)
        language = {
            'language_name': 'NYI',
            'language_abbr': 'NYI',
            'langauge_aka': '',
            'family_name': 'NYI',
            'family_name_override': '',
            # todo: I think that this is incorrect but I need to put this data somewhere
            'sub_family_name': row.branch,
            'is_sub_branch': False,
            'is_sub_sub_branch': False
        }
        entry = {
            # language/family
            "language": language,
            # reflex itself (may contain multiple), store as list
            "reflexes": [row.language_form_and_translation],
            # part of speech
            "pos": "NYI",
            # the meaning (gloss)
            "gloss": "NYI",
            # source (static cuz we only have one place we are taking it from).
            "source": {
                "text_sources": [{'code': 'LIV', 'display': 'Helmut Rix, Martin Kümmel et al.: Lexikon der indogermanischen Verben (1998)'}],
                'db_sources': [{'code': 'TOL', 'display': 'LIV as compiled by Thomas Olander)'}]
            },
        }
        entries.append(entry)
    return entries


def create_lineup_csv():
    with open("data_pokorny/table_pokorny.json", "r") as fp:
        pokorny_json = json.load(fp)
    liv_data_path = "data_liv/Olander_LIV_-_resultat_af_script.csv"

    root_match_up = pd.DataFrame([{"root": entry["root"], "searchables": entry["searchable_roots"], "canonical_id": "", "liv": ""} for entry in pokorny_json])
    root_match_up.to_csv("data_liv/pokorny_liv_root_matchup.csv")

    df = pd.read_csv(liv_data_path)
    df["iew_num"] = df["iew_reference"].str.extract(r"IEW (\d*)")
    df["iew_num"] = pd.to_numeric(df["iew_num"], downcast='integer')
    df[["lemma", "iew_num", "iew_reference"]].drop_duplicates().sort_values(by="iew_num").to_csv("data_liv/all_liv_roots.csv", index=False)

    breakpoint()


def load_olander():
    liv_data_path = "data_liv/Olander_LIV_-_resultat_af_script.csv"
    df = pd.read_csv(liv_data_path)

    # fixme: this behavior might be confusing, check here if something is strange.
    # if the english_meanings.txt doesnt already exist
    if not os.path.exists("data_liv/english_meanings.txt"):
        # write out the translations to file for auto translating
        with open("data_liv/german_meanings.txt", "w", encoding='utf8') as fp:
            fp.writelines([
                # remove any citation in form {num} and then strip all these possible quotation marks ’‘
                re.sub(r'\{\d+\}', '', line).strip(' ’‘') + "\n"
                for line in df.translation.to_list()
            ])
        translate_meaning_gpt.main("data_liv/german_meanings.txt", "data_liv/english_meanings.txt")

    # load the english
    with open("data_liv/english_meanings.txt", "r", encoding='utf8') as fp:
        english_meanings = [line.strip(" \n") for line in fp.readlines()]
        df["english_meaning"] = english_meanings
    return df


def main():
    df = load_olander()

    root_groups = df.groupby("lemma")
    liv_entries = []
    for root, group in tqdm(list(root_groups), ncols=150):
        gloss = group[group.category == "root"].iloc[0].english_meaning,

        # todo: make this actually match the entry_id that exists in the common version. I have no clue how to do this.
        entry_id = f'{root}_{gloss}'

        reflex_entries = get_reflex_entries(group)

        entry = {
            "entry_id": entry_id,
            "root": root,
            "meaning": gloss,
            "reflexes": reflex_entries,
            # todo: part of speech
            "pos": ["NYI"],
            "semantic": "NYI",
            # todo: figure out how to make roots actually searchable from the root that we have.
            "searchable_roots": root
        }
        liv_entries.append(entry)

    with open("data_liv/table_liv.json", "w", encoding="utf-8") as fp:
        json.dump(liv_entries, fp, indent=2)


def gpt_fix_fields():
    df = load_olander()
    # get rid of all rows where category is root
    df = df[df.category != "root"]
    df["notes"] = df["notes"].fillna("")

    # iterate through all the rows
    df_list = []
    for i, (_, row) in enumerate(df.iterrows()):
        headers = ["language", "reflex", "questionable", "meaning", "notes"]
        headers_str = str(headers)[1:-1]
        prompt_ex = (
            "Here are some examples and their output,:\n"
            "Information: [gr. ἔφαγον ‘aß (auf)’{3}\n"
            f"{headers_str}\n"
            "'gr.', 'ἔφαγον', 'false', 'aß (auf)', '3'\n"
            "\nInformation: ved. Konj. bhájati ‘teilt zu’, bhájate ‘bekommt Anteil’\n"
            f"{headers_str}\n"
            "'ved.', 'bhájati', 'false', 'teilt zu', 'None'\n"
            "'ved.', 'bhájate', 'false', 'bekommt Anteil', 'None'\n"
            "\nInformation: alit. bė́gmi, [lit. bė́gu, (bė́gti) ‘laufen, fliehen’\n"
            f"{headers_str}\n"
            "'alit.', 'bė́gmi', 'false', 'laufen, fliehen', 'None'\n"
            "'lit.', 'bė́gu', 'false', 'laufen, fliehen', 'None'\n"
            "\nInformation: ?[arm. Inj. Med. bekanem ‘breche’{3}\n"
            f"{headers_str}\n"
            "'arm.', 'bekanem', 'true', 'breche', '3'\n"
        )
        # make a prompt for gpt to separate out all the parts that seems to be stuffed into the language_form_and_translation
        prompt = (
            f'"{row.language_form_and_translation}" is information (in german) about the reflexes for the PIE root "{row.lemma}" described as "{row.english_meaning}". '
            f'It should be about a {row.category} in {row.branch}. '
            f'I need to extract the language abbreviation, the reflex (there may be multiple), whether it is questionable, the german meaning, and any note markers. '
            f'I want this in comma separated format with the following headers: {headers_str}. '
            f'\n\n{prompt_ex}\n'
            f'Give these in CSV format, without any other text, placed in a code block. '
            f'Remember to quote every field to preserve commas and to put each reflex on its own line. '
            f'Also make sure you only put the reflex in the reflex column, without information  \n\n'
            f'Information: "{row.language_form_and_translation}"\n'
            f'{headers_str}'
        )
        # row.notes is in form "\{int\} note text \{int\} note text", I need to turn this into a dict from {"int": "note text"} using regex
        notes = re.findall(r'{(\d+)}\s+(.*?)(?=\s*{(\d+)}|\s*$)', row.notes)
        notes = {note[0]: note[1] for note in notes}

        # print(f"~~~~~~~~~~~~~~~{i+1}/{len(df)}~~~~~~~~~~~~~~~")
        # print("Original:", row.language_form_and_translation)
        for _ in range(3):
            try:
                response = query_gpt(prompt)
                csv_text = extract_code_block(response["choices"][0]["message"]["content"])
                # print("CSV'd:", *[f"\n\t{text}" for text in csv_text.split("\n")])
                response_df = process_gpt(csv_text, headers, quote_char="'")
                # print(response_df)
                response_df["root"] = row.lemma
                response_df["branch"] = row.branch
                response_df["category"] = row.category
                response_df["original_text"] = row.language_form_and_translation
                response_df["expanded_notes"] = [f"{note}: {notes[note]}" if note in notes else note for note in response_df.notes]
                df_list.append(response_df)
                break
            except pd.errors.ParserError as e:
                breakpoint()
                print(e)
                delete_response(prompt)
                continue
    # merge all the dfs in the list, ignoring the index
    combined_df = pd.concat(df_list, ignore_index=True)
    # reorder the columns
    combined_df = combined_df[["root", "category", "branch", "language", "questionable", "reflex", "meaning", "notes", "expanded_notes", "original_text"]]
    # save to file
    combined_df.to_csv("data_liv/liv_gpt.csv")
    # breakpoint()
    pass


if __name__ == '__main__':
    # gpt_fix_fields()
    # create_lineup_csv()
    main()
    pass
