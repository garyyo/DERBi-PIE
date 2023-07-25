import glob
import json
import os.path
import re
from collections import defaultdict
from io import StringIO

import pandas as pd
import pyperclip
from tqdm import tqdm

from process_pokorny import remove_non_english_chars
from query_gpt import query_gpt, process_gpt, query_gpt_fake, get_digest, get_text_digest


def remove_html_tags_from_text(text):
    clean_text = re.sub(r'<.*?>', '', text)
    return clean_text


def find_html_tags(text):
    pattern = r"<([a-zA-Z0-9]+)[^>]*>"
    matches = re.findall(pattern, text)
    return matches


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


def get_reflex_language(dfs, language_id):
    language_series = dfs["lex_language"][dfs["lex_language"].id == language_id].iloc[0]
    sub_family_series = dfs["lex_language_sub_family"][dfs["lex_language_sub_family"].id == language_series.sub_family_id].iloc[0]
    family_series = dfs["lex_language_family"][dfs["lex_language_family"].id == sub_family_series.family_id].iloc[0]

    # subfamily seems to include a lot of HTML tags, which indicate whether it is a subbranch or sub-subbranch
    found_tags = find_html_tags(sub_family_series["name"])
    sub_family_name = remove_html_tags_from_text(sub_family_series["name"]).strip()

    # sometimes there is no subfamily, but there is a family, so this assert does not hold true. This is required since they do things in a weird way.
    # assert sub_family_name != ""

    family_name = family_series["name"].strip()
    family_name_override = ""
    if language_series.override_family is not None:
        family_name_override = language_series.override_family

    return {
        "language_name": language_series["name"].strip(),
        "language_abbr": language_series["abbr"].strip(),
        "langauge_aka": language_series.aka.strip(),
        "family_name": family_name.strip(),
        "family_name_override": family_name_override.strip(),
        "sub_family_name": sub_family_name.strip(),
        # saving whether it is a sub or sub_sub, or neither.
        "is_sub_branch": "b" in found_tags,
        "is_sub_sub_branch": "i" in found_tags,
        # todo: figure out if we actually care about custom sort, or if we care about order (I don't think so)
    }


def get_reflex_pos(dfs, reflex_id):
    pos_code_to_meaning = dict(dfs['lex_part_of_speech'][["code", "display"]].to_numpy().tolist())
    reflex_pos_series = dfs["lex_reflex_part_of_speech"][dfs["lex_reflex_part_of_speech"].reflex_id == reflex_id].iloc[0]
    pos_code = reflex_pos_series.text
    pos_code_meaning = [{"code": code, "meaning": pos_code_to_meaning[code]} for code in pos_code.split(".")]

    # we store the code, and the decoded words for the code. I can either list out what they mean or probably include like a popover on mouseover or something.
    return pos_code_meaning


def get_reflex_source(dfs, reflex_id):
    source_ids = dfs["lex_reflex_source"][dfs["lex_reflex_source"].reflex_id == reflex_id].source_id.to_list()
    return {
        "text_sources": [[dfs["lex_source"][dfs["lex_source"].id == source_id].iloc[0][["code", "display"]].to_dict() for source_id in source_ids]],
        # we always credit LRC for the db source since we are processing their DB dump.
        "db_sources": [dfs["lex_source"][dfs["lex_source"].code == "LRC"].iloc[0][["code", "display"]].to_dict()]
    }


def get_reflex_entries(dfs, lrc_id):
    reflex_ids = dfs['lex_etyma_reflex'][dfs['lex_etyma_reflex']['etyma_id'] == lrc_id]["reflex_id"].tolist()
    reflex_df = dfs["lex_reflex"][dfs["lex_reflex"].id.isin(reflex_ids)]

    entries = []
    for i, reflex_row in reflex_df.iterrows():
        reflex_json = json.loads(reflex_row["entries"])

        entry = {
            # language/family
            "language": get_reflex_language(dfs, reflex_row.language_id),
            # reflex itself (may contain multiple), store as list
            "reflexes": [val["text"] for val in reflex_json],
            # part of speech
            "pos": get_reflex_pos(dfs, reflex_row.id),
            # the meaning (gloss)
            "gloss": reflex_row["gloss"],
            # source (generally IEW, but we will credit LRC as db source)
            "source": get_reflex_source(dfs, reflex_row.id),
        }

        # I do not get why reflex_row["entries"] is json, and why it's a list, so I have an assert in case something changes
        # assumption 1: the only field in each of these is "text"
        assert all([sorted(val.keys()) == ["text"] for val in reflex_json])

        entries.append(entry)

    return entries


def figure_out_language_overrides(dfs):
    # find all overrides, that are actually used in a reflex.
    override_df = dfs["lex_language"][~dfs["lex_language"].override_family.isna() & dfs["lex_language"].id.isin(dfs['lex_reflex'].language_id.unique().tolist())]
    override_to_lang = defaultdict(set)
    for i, row in override_df.iterrows():
        sub_family_series = dfs["lex_language_sub_family"][dfs["lex_language_sub_family"].id == row.sub_family_id].iloc[0]
        family_series = dfs["lex_language_family"][dfs["lex_language_family"].id == sub_family_series.family_id].iloc[0]

        language_name = row["abbr"]
        sub_name = remove_html_tags_from_text(sub_family_series["name"])
        family_name = family_series["name"]
        override_family_name = row.override_family

        override_to_lang[(sub_name, family_name, override_family_name)].add(language_name)

    # print out which language families are overriden and a truncated list of languages
    print(f"{'sub_family':<10}\t{'family_name':<10}\t{'override_family':<10}\t{'overridden languages'}")
    for (sub_name, family_name, override_family_name), languages in override_to_lang.items():
        str_languages = str(list(languages))
        len_cutoff = 8
        if len(languages) > len_cutoff:
            str_languages = str(sorted(languages)[:len_cutoff])[:-1] + ", ...]"
        print(f"{sub_name:<10}\t{family_name:<10}\t{override_family_name:<10}\t{str_languages}")

    breakpoint()


def main():
    dfs = {os.path.splitext(os.path.basename(df_file))[0]: pd.read_pickle(df_file) for df_file in glob.glob("data_pokorny/table_dumps/*.df")}
    # figure_out_language_overrides(dfs)

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
    pokorny_by_id = {}
    # for storing the entries in a way that can be easily cross_referenced
    lrc_to_pokorny_id = {}
    # for linking cross-references on a second pass
    entry_to_entry = defaultdict(set)
    for i, row in tqdm(list(dfs["lex_etyma"].iterrows()), ncols=150):
        lrc_id = row['id']

        # link to reflexes
        reflex_entries = get_reflex_entries(dfs, lrc_id)

        # basic info
        roots = row["entry"].strip("\n\t ").replace("<p>", "").replace("</p>", "")
        gloss = row["gloss"].strip("\n\t ").replace("<p>", "").replace("</p>", "")

        # making the id
        filtered_root = remove_non_english_chars(remove_html_tags_from_text(roots))
        filtered_gloss = remove_non_english_chars(remove_html_tags_from_text(gloss), " ").replace(" ", "_").strip("() <>\\/.[]{}")
        entry_id = f'{filtered_root}_{filtered_gloss}'

        # check for cross-references
        to_refs = dfs['lex_etyma_cross_reference'][dfs['lex_etyma_cross_reference'].to_etyma_id == lrc_id].from_etyma_id.tolist()
        from_refs = dfs['lex_etyma_cross_reference'][dfs['lex_etyma_cross_reference'].from_etyma_id == lrc_id].to_etyma_id.tolist()
        cross = sorted(set(to_refs + from_refs))
        entry_to_entry[lrc_id] = cross

        entry = {
            "entry_id": entry_id,
            "root": roots,
            "meaning": gloss,
            "reflexes": reflex_entries,
            "pos": [],
            "semantic": [],
            # this id is deleted on the second pass
            "lrc_id": lrc_id,
        }
        pokorny_entries.append(entry)
        pokorny_by_id[entry_id] = entry
        lrc_to_pokorny_id[lrc_id] = entry_id

    # breakpoint()

    # second pass to link cross-references
    pokorny_entries_new = []
    for i, entry in enumerate(pokorny_entries):
        # extract the old id (and remove it)
        lrc_id = entry["lrc_id"]
        del entry["lrc_id"]

        # cross-references should not link to their own entry.
        entry["cross"] = sorted([
            {"id": lrc_to_pokorny_id[cross_id], "display": pokorny_by_id[lrc_to_pokorny_id[cross_id]]["root"]}
            for cross_id in entry_to_entry[lrc_id]
            if lrc_to_pokorny_id[cross_id] != entry["entry_id"]
        ], key=lambda v: v['display'])

        pokorny_entries_new.append(entry)

    with open("data_pokorny/table_pokorny.json", "w", encoding="utf-8") as fp:
        json.dump(pokorny_entries_new, fp)
    # breakpoint()
    pass


if __name__ == '__main__':
    main()
    pass
