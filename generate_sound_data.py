import json
import re

import pandas as pd


"""
author - anton vinogradov

The purpose of this script is to generate every individual sound that is used in the DERBi PIE website. These are distinct from characters as they may contain
multiple characters, which considering that regex works on a character level usually, is necessary. These are generated from one csv file (see the main() 
function for which one) and thus to add on more, that csv should be edited, not this code. 
"""


def get_input_to_sounds(inventory):
    input_to_sounds = []
    for i, row in inventory.iterrows():
        sounds = [row["user_input"]]
        if "," in row["sounds"]:
            sounds = row["sounds"].split(", ")

        entry = {
            "user_input": row["user_input"],
            "sounds": sounds,
        }
        # breakpoint()
        input_to_sounds.append(entry)
    return input_to_sounds


def select_pos(features, group):
    return features[features[group] != "0"]["Sound"].to_list()


def select_neg(features, group):
    return features[group] != "1"


def main():
    features = pd.read_csv("sound_processing/sound_features.csv", dtype=str).fillna("0")
    # inventory = pd.read_csv("sound_processing/sound_inventory.csv")
    # input_to_sounds = get_input_to_sounds(inventory)

    groups_start = 2
    groups_end = 27
    letters_start = -11

    # for possible grouping, we need to find which sounds correspond to that group
    # all the positive features
    plus_groups = {f"[+{group}]": select_pos(features, group) for group in features.columns[groups_start:groups_end]}
    # negative features
    minus_groups = {f"[-{group}]": features[(features[group] != "1")]["Sound"].to_list() for group in features.columns[groups_start:groups_end]}
    # cover letters (capital letters that stand for a few different sounds)
    cover_groups = {group: select_pos(features, group) for group in features.columns[letters_start:]}
    # the rest, mostly things that use parens that are expanded out, but also the interchangeable pairs
    paren_alt_groups = {group: select_pos(features, group) for group in features.columns[groups_end:letters_start]}
    # corresponds to the - (dash) symbol, so needs to be separated out
    dash_group = features["Sound"].to_list()

    all_groups = {**plus_groups, **minus_groups, **cover_groups, **paren_alt_groups}
    all_regex = {group_name: make_regex(group_sounds) for group_name, group_sounds in all_groups.items()}
    # dash group is weird as it technically matches 1 or more sounds
    all_regex["-"] = f"{make_regex(dash_group)}+"
    # hashtag is for a word boundary
    # fixme: but the db is not set up for this very well as roots contains parens, which probably need to be expanded out for searching.
    all_regex["#"] = "(?:\s|^)"

    with open("sound_processing/regex.json", "w", encoding="utf-8") as fp:
        json.dump(all_regex, fp, indent=2)

    pass


def make_regex(sound_list):
    return f'(?:{"|".join([re.escape(sound) for sound in sound_list])})'


if __name__ == '__main__':
    main()
    pass
