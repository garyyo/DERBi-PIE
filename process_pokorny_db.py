import glob
import json
import os.path
import re
from collections import defaultdict

import pandas as pd
from process_pokorny import remove_non_english_chars


def remove_html_tags_from_text(text):
    clean_text = re.sub(r'<.*?>', '', text)
    return clean_text


def common():
    dfs = {os.path.splitext(os.path.basename(df_file))[0]: pd.read_pickle(df_file) for df_file in glob.glob("data_pokorny/table_dumps/*.df")}

    # create entries for each word
    master_entries = []
    for i, row in dfs["lex_etyma"].iterrows():
        reflex_ids = dfs['lex_etyma_reflex'][dfs['lex_etyma_reflex']['etyma_id'] == row["id"]]["reflex_id"].tolist()
        reflex_pos = dfs['lex_reflex_part_of_speech'][dfs['lex_reflex_part_of_speech'].id.isin(reflex_ids)].text.tolist()
        reflex_pos = sorted(set(reflex_pos))

        roots = row["entry"].strip("\n\t ").replace("<p>", "").replace("</p>", "")
        gloss = row["gloss"].strip("\n\t ").replace("<p>", "").replace("</p>", "")

        filtered_root = remove_non_english_chars(remove_html_tags_from_text(roots))
        filtered_gloss = remove_non_english_chars(remove_html_tags_from_text(gloss), " ").replace(" ", "_").strip("() <>\\/.[]{}")

        entry_id = f'{filtered_root}_{filtered_gloss}'

        entry = {
            "entry_id": entry_id,
            "root": roots,
            "meaning": gloss,
            "pos": reflex_pos,
        }
        master_entries.append(entry)

    with open("data_pokorny/pokorny_db_processed.json", "w") as fp:
        json.dump(master_entries, fp)

    pass


def main():
    dfs = {os.path.splitext(os.path.basename(df_file))[0]: pd.read_pickle(df_file) for df_file in glob.glob("data_pokorny/table_dumps/*.df")}

    # lex_etyma: the main entries
    # lex_etyma_cross_reference: entries that link to other entries because they are related
    # lex_etyma_reflex: links from etyma to reflex
    # lex_etyma_semantic_field: links from etyma to semantic (lex_semantic_field)
    # lex_language: language entries
    # lex_language_family: not entirely sure, but contains the language families?
    # lex_language_sub_family: language subfamilies, seems like lex_language has a ref to these
    # lex_lexicon: idk
    # lex_part_of_speech: lists all the POS codes (the linguist readable) and their regular human-readable counterparts
    # lex_reflex: the reflexes, aka the derivatives(?)
    # lex_reflex_part_of_speech: reflex to pos codes. seems that the codes still need decoding
    # lex_reflex_source: links between reflexes and the source of the reflex info (links to lex_source)
    # lex_semantic_category: overarching semantic categories
    # lex_semantic_field: more narrow semantic categories, links to the categories
    # lex_source: where information (in the reflex) is from.

    # for each entry in the lex etyma:
    #   calculate the basic info (roots, meaning, id)
    #   find links to cross-references, make note of these entries (link to our ids in second pass)
    #   find links to reflexes
    #       a single entry can have multiple reflexes
    #       a single reflex can be part of multiple entries (but only if the entries are already cross-referenced already)

    # the actual entries
    pokorny_entries = []
    # for storing the entries in a way that can be easily cross_referenced
    lrc_to_pokorny_id = {}
    # for linking cross-references on a second pass
    entry_to_entry = defaultdict(set)
    for i, row in dfs["lex_etyma"].iterrows():
        lrc_id = row['id']

        # link to reflexes
        reflex_ids = dfs['lex_etyma_reflex'][dfs['lex_etyma_reflex']['etyma_id'] == lrc_id]["reflex_id"].tolist()
        reflex_pos = dfs['lex_reflex_part_of_speech'][dfs['lex_reflex_part_of_speech'].id.isin(reflex_ids)].text.tolist()
        reflex_pos = sorted(set(reflex_pos))

        # basic info
        roots = row["entry"].strip("\n\t ").replace("<p>", "").replace("</p>", "")
        gloss = row["gloss"].strip("\n\t ").replace("<p>", "").replace("</p>", "")

        # making the id
        filtered_root = remove_non_english_chars(remove_html_tags_from_text(roots))
        filtered_gloss = remove_non_english_chars(remove_html_tags_from_text(gloss), " ").replace(" ", "_").strip("() <>\\/.[]{}")
        entry_id = f'{filtered_root}_{filtered_gloss}'

        # check for link backs
        cross = set(dfs['lex_etyma_cross_reference'][(dfs['lex_etyma_cross_reference'].from_etyma_id == lrc_id) | (dfs['lex_etyma_cross_reference'].to_etyma_id == lrc_id)][["from_etyma_id", "to_etyma_id"]].to_numpy().flatten().tolist())
        # cross = cross.union([row['id']])
        cross = tuple(sorted(cross))

        for cross_id in cross:
            entry_to_entry[lrc_id].add(cross_id)

        entry = {
            "lrc_id": lrc_id,
            "entry_id": entry_id,
            "root": roots,
            "meaning": gloss,
            "pos": reflex_pos,
            "semantic": []
        }
        pokorny_entries.append(entry)
        lrc_to_pokorny_id[lrc_id] = entry_id

    # second pass to link cross-references
    pokorny_entries_new = []
    for i, entry in enumerate(pokorny_entries):
        # extract the old id (and remove it)
        lrc_id = entry["lrc_id"]
        del entry["lrc_id"]

        entry["cross"] = []
        for cross_id in entry_to_entry[lrc_id]:
            entry["cross"].append(lrc_to_pokorny_id[cross_id])

        pokorny_entries_new.append(entry)

    breakpoint()
    pass


if __name__ == '__main__':
    main()
    pass
