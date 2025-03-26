import json

from pyconcepticon import Concepticon


from generate_pokorny_db_data import remove_html_tags_from_text


relevant_concept_set_keys = [
    "id",
    "definition",
    "gloss",
    "semanticfield",
]


def add_concepticon_data():
    api = Concepticon('concepticon-concepticon-data-e4dd288')

    concept_glosses = {concept_id: concept.gloss for concept_id, concept in api.conceptsets.items()}
    concepticon_glosses = list(v.lower() for v in concept_glosses.values())
    gloss_to_concept = {v: k for k, v in concept_glosses.items()}

    with open("data_common/table_common.json", "r") as fp:
        common_table = json.load(fp)

    concepticon_glosses_set = set(concepticon_glosses)
    for entry in common_table:
        common_table_glosses = set(remove_html_tags_from_text(word).lower().strip(" './-{}()[]\\/!@#$%^&*=.,;:\"") for word in entry["meaning"].split(" "))
        match = concepticon_glosses_set.intersection(common_table_glosses)

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


def main():
    # anton: dont use this.
    api = Concepticon('concepticon-concepticon-data-e4dd288')
    # concept_lists = [concept_list for concept_list in api.conceptlists.values() if concept_list.target_language not in ["chinese", "japanese"]]

    concept_glosses = {concept_id: concept.gloss for concept_id, concept in api.conceptsets.items()}
    concepticon_glosses = list(v.lower() for v in concept_glosses.values())
    gloss_to_concept = {v: k for k, v in concept_glosses.items()}

    # with open("tempt/misc.json", "w") as fp:
    #     json.dump(concepticon_glosses, fp)

    # with open("tempt/misc.json", "r") as fp:
    #     concepticon_glosses = set(json.load(fp))

    with open("data_common/table_common.json", "r") as fp:
        common_table = json.load(fp)

    counter = 0
    used_concepts = set()
    concepticon_glosses_set = set(concepticon_glosses)
    for entry in common_table:
        common_table_glosses = set(remove_html_tags_from_text(word).lower().strip(" './-{}()[]\\/!@#$%^&*=.,;:\"") for word in entry["meaning"].split(" "))
        match = concepticon_glosses_set.intersection(common_table_glosses)

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
            not_matched = entry
    breakpoint()
    with open("data_common/table_common.json", "w") as fp:
        json.dump(common_table, fp)


if __name__ == '__main__':
    main()
