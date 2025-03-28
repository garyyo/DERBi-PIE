import json
import re
import Levenshtein
import pandas as pd
from nltk.stem import WordNetLemmatizer
import csv

from pyconcepticon import Concepticon

from generate_pokorny_db_data import remove_html_tags_from_text

lemmatizer = WordNetLemmatizer()
relevant_concept_set_keys = [
    "id",
    "definition",
    "gloss",
    "semanticfield",
]


def split_excluding_parentheses(s):
    # Regular expression to match commas outside parentheses
    return


def add_concepticon_data():
    api = Concepticon('concepticon-concepticon-data-e4dd288')

    concept_glosses = {concept_id: concept.gloss for concept_id, concept in api.conceptsets.items()}
    concepticon_glosses = list(v.lower() for v in concept_glosses.values())
    gloss_to_concept = {v: k for k, v in concept_glosses.items()}
    concepticon_glosses_lemmatized = {lemmatizer.lemmatize(word): word for word in concepticon_glosses}

    with open("data_common/table_common.json", "r") as fp:
        common_table = json.load(fp)

    manually_linked_df = pd.read_csv("data_common/concepticon_now_linked.csv", encoding="utf-8")

    concepticon_glosses_set = set(concepticon_glosses)
    for entry in common_table:
        common_table_glosses = set(remove_html_tags_from_text(word).lower().strip(" './-{}()[]\\/!@#$%^&*=.,;:\"") for word in entry["meaning"].split(" "))

        match = concepticon_glosses_set.intersection(common_table_glosses)

        # if we have a manual override do that instead
        if entry["root"] in manually_linked_df.Root.values:
            row = manually_linked_df[manually_linked_df.Root == entry["root"]].iloc[0]
            if not pd.isna(row["Suggested Match"]):
                match = {word.strip().lower() for word in re.split(r',(?=(?:[^()]*\([^()]*\))*[^()]*$)', row["Suggested Match"])}

        # If we do not find a match then we try again on a lemmatized version of the word
        if not match:
            match = set(concepticon_glosses_lemmatized.keys()).intersection({lemmatizer.lemmatize(gloss) for gloss in common_table_glosses})
            match = {concepticon_glosses_lemmatized[word] for word in match}

        # connect the concepticon entries with ours. The above search is not optimal so this may need to be corrected at a later date.
        entry["concepticon"] = [{
            key: api.conceptsets[
                gloss_to_concept[concept.upper()]
            ].__dict__[key]
            for key in relevant_concept_set_keys
        } for concept in match]

    with open("data_common/table_common.json", "w") as fp:
        json.dump(common_table, fp)

    pass


def closest_word(target: str, words: iter):
    closest = min(words, key=lambda word: Levenshtein.distance(target, word))
    return closest if Levenshtein.distance(target, closest) <= 2 else None


def main():
    # anton: dont use this.
    api = Concepticon('concepticon-concepticon-data-e4dd288')
    # concept_lists = [concept_list for concept_list in api.conceptlists.values() if concept_list.target_language not in ["chinese", "japanese"]]

    concept_glosses = {concept_id: concept.gloss for concept_id, concept in api.conceptsets.items()}
    concepticon_glosses = list(v.lower() for v in concept_glosses.values())
    gloss_to_concept = {v: k for k, v in concept_glosses.items()}
    concepticon_glosses_lemmatized = {lemmatizer.lemmatize(word): word for word in concepticon_glosses}

    # with open("tempt/misc.json", "w") as fp:
    #     json.dump(concepticon_glosses, fp)

    # with open("tempt/misc.json", "r") as fp:
    #     concepticon_glosses = set(json.load(fp))

    with open("data_common/table_common.json", "r") as fp:
        common_table = json.load(fp)

    counter = 0
    used_concepts = set()
    concepticon_glosses_set = set(concepticon_glosses)
    unmatched = []
    for entry in common_table:
        common_table_glosses = set(
            re.sub(
                '[\.\/\-\{\}\(\)\[\]\\\/\!\@\#\$\%\^\&\*\=\.\,\;\:\"\?]',
                " ",
                remove_html_tags_from_text(word).lower()
            ).strip()
            for word in entry["meaning"].split(" ")
        )
        match = concepticon_glosses_set.intersection(common_table_glosses)
        # we try again on the lemmatized version
        if not match:
            match = set(concepticon_glosses_lemmatized.keys()).intersection({lemmatizer.lemmatize(gloss) for gloss in common_table_glosses})
            match = {concepticon_glosses_lemmatized[word] for word in match}

        # connect the concepticon entries with ours. The above search is not optimal so this may need to be corrected at a later date.
        entry["concepticon"] = [{
            key: api.conceptsets[
                gloss_to_concept[concept.upper()]
            ].__dict__[key]
            for key in relevant_concept_set_keys
        } for concept in match]
        if match:
            # print(entry["root"], entry["meaning"], "->", match)
            used_concepts = match.union(used_concepts)
            counter += 1
        else:
            potential_partial_matches = [closest_word(gloss, concepticon_glosses_set) for gloss in common_table_glosses]
            potential_partial_matches = [match for match in potential_partial_matches if match is not None]
            unmatched.append([entry, [potential_partial_matches]])

    print(f"{len(unmatched)}/{len(common_table)} = {len(unmatched)/len(common_table) * 100:.02f}%")

    with open("data_common/concepticon_unlinked.csv", mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Root', 'Meaning', 'Closest Concepticon Match (unreliable)'])

        for entry in unmatched:
            first_part, second_part = entry
            root = first_part['root']
            meaning = first_part['meaning']
            words = [word.upper() for sublist in second_part for word in sublist] if isinstance(second_part, list) else []
            writer.writerow([root, meaning, ', '.join(words)])

    # breakpoint()

    pass


# do not run this manually to get the additional data added to common, instead run generate_common_db_data.py which should run the necessary portion of this script.
if __name__ == '__main__':
    # add_concepticon_data()
    main()
