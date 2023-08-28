import json

import numpy as np
import pandas as pd


def main():
    # open the pokorny and liv files
    pokorny_filename = "data_pokorny/table_pokorny.json"
    liv_filename = "data_liv/table_liv.json"
    with open(pokorny_filename, 'r') as fp:
        pokorny_data_list = json.load(fp)
    with open(liv_filename, 'r') as fp:
        liv_data_list = json.load(fp)
    # pokorny and liv need to be redictionaried into key: entry
    pokorny_data = {entry["root"]: entry for entry in pokorny_data_list}
    liv_data = {entry["root"]: entry for entry in liv_data_list}
    assert len(pokorny_data) == len(pokorny_data_list)
    assert len(liv_data) == len(liv_data_list)

    # open the match-up csv
    match_df = pd.read_csv("data_common/matchup.csv")
    # fill the ["liv: cross-reference"] with "" instead of NaN
    match_df["liv: cross-reference"] = match_df["liv: cross-reference"].fillna("")

    # for everything in the match-up:
    # 1. create a new common entry, with a list of objects under the name "dictionary":
    # 2. add the pokorny root to that entry in "pokorny_entries" of that list
    # 3. if there is a liv, add the liv root to that entry in "liv_entries" of that list
    # 3. keep track of which liv roots you have used.
    #   If there are any left over at the end add them as an entry, with that root in the liv_entries, but nothing in the pokorny_entries
    used_liv_roots = set()
    common_data = []
    liv_to_pokorny = {}
    counter = 0
    for index, row in match_df.iterrows():
        pokorny_root = row["root"]
        liv_root = row["liv: cross-reference"]
        liv_roots = [root.strip() for root in liv_root.split(",")]
        # find the pokorny entry in the pokorny data
        # pokorny_data_entry = pokorny_data[pokorny_root]
        # liv_data_entry = liv_data.get(liv_root, None)

        new_entry = {
            "root": pokorny_root,
            "dictionary": [
                {"pokorny_entries": [pokorny_root]},
                {"liv_entries": liv_roots if liv_root else []}
            ],
            "numerical_id": counter
        }
        counter += 1
        # add it to the list
        common_data.append(new_entry)
        for root in liv_roots:
            used_liv_roots.add(root)
            liv_to_pokorny[root] = pokorny_root

    # remove '' from the set of used liv roots and find the
    used_liv_roots.remove('')
    # set math
    unused_liv = set(liv_data.keys()) - used_liv_roots
    # add the remaining liv roots to the common data
    for root in unused_liv:
        new_entry = {
            "root": root,
            "dictionary": [
                {"pokorny_entries": []},
                {"liv_entries": [root]}
            ],
            "numerical_id": counter
        }
        counter += 1
        common_data.append(new_entry)
    # reorganize the common data into a dictionary by pokorny id
    common_data_dict = {entry["root"]: entry for entry in common_data}

    # now go back through the pokorny and liv and add a "common_entries" to each
    for entry in pokorny_data.values():
        root = entry["root"]
        common_entry = common_data_dict[root]
        entry["common_id"] = common_entry["numerical_id"]

    for entry in liv_data.values():
        root = entry["root"]
        if root in liv_to_pokorny:
            root = liv_to_pokorny[root]
        common_entry = common_data_dict[root]
        entry["common_id"] = common_entry["numerical_id"]

    print("")
    # save the common data
    with open("data_common/table_common.json", 'w') as fp:
        json.dump(common_data, fp, indent=4)
    # save the pokorny data
    with open("data_pokorny/table_pokorny.json", 'w') as fp:
        json.dump(pokorny_data_list, fp, indent=4)
    # save the liv data
    with open("data_liv/table_liv.json", 'w') as fp:
        json.dump(liv_data_list, fp, indent=4)
    pass


if __name__ == '__main__':
    main()
    pass
