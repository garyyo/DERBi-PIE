import json

import pandas as pd
from tqdm import tqdm


def get_reflex_entries(group):
    entries = []
    for i, row in group.iterrows():
        if row.category == "root":
            continue
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
                "text_sources": [[{'code': 'LIV', 'display': 'Helmut Rix, Martin Kümmel et al.: Lexikon der indogermanischen Verben (1998)'}]],
                'db_sources': [{'code': 'TOL', 'display': 'LIV as compiled by Thomas Olander)'}]
            },
        }
        entries.append(entry)
    return entries


def main():
    liv_data_path = "data_liv/Olander_LIV_-_resultat_af_script.csv"
    df = pd.read_csv(liv_data_path)
    root_groups = df.groupby("lemma")

    liv_entries = []
    for root, group in tqdm(list(root_groups), ncols=150):
        gloss = group[group.category == "root"].iloc[0].translation,

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


if __name__ == '__main__':
    main()
    pass
