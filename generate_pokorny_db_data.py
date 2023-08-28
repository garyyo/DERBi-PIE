import glob
import json
import os.path
import re
from collections import defaultdict

import pandas as pd
import pyperclip
from tqdm import tqdm

from generate_pokorny_scraped_data_OLD import remove_non_english_chars


"""
author - anton vinogradov

This script is a collection of methods used to generate 2 files for the database: table_pokorny.json and table_common.json.
Currently the common table (these are called tables but are actually "collections" when in MongoDB) is mostly redundant, but is there to serve as a common point
of reference between all specialized tables. 
"""


def load_abbreviations_df():
    abbreviation_data = pd.read_csv("data_pokorny/abbreviations.csv").dropna(axis=1, how='all').fillna("")
    # remove parens
    parens_rows = abbreviation_data[abbreviation_data['German abbreviation'].str.contains(r"[()]")].copy()
    parens_rows['German abbreviation'] = parens_rows['German abbreviation'].str.replace(r'[()]', '', regex=True)
    parens_rows['anton notes'] = "() removed"
    # removes parens + inside
    parens_rows2 = abbreviation_data[abbreviation_data['German abbreviation'].str.contains(r"[()]")].copy()
    parens_rows2['German abbreviation'] = parens_rows2['German abbreviation'].apply(lambda x: re.sub(r'\([^()]*\)', '', x))
    parens_rows2['anton notes'] = "(...) removed"
    abbreviation_data = pd.concat([abbreviation_data, parens_rows, parens_rows2], ignore_index=True)
    return abbreviation_data


def remove_html_tags_from_text(text):
    clean_text = re.sub(r'<.*?>', '', text)
    return clean_text


def find_html_tags(text):
    pattern = r"<([a-zA-Z0-9]+)[^>]*>"
    matches = re.findall(pattern, text)
    return matches


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
        "text_sources": [dfs["lex_source"][dfs["lex_source"].id == source_id].iloc[0][["code", "display"]].to_dict() for source_id in source_ids],
        # we always credit LRC for the db source since we are processing their DB dump.
        "db_sources": [dfs["lex_source"][dfs["lex_source"].code == "LRC"].iloc[0][["code", "display"]].to_dict()]
    }


def get_semantic(dfs, lrc_id):
    try:
        semantic_id = dfs["lex_etyma_semantic_field"][dfs["lex_etyma_semantic_field"].etyma_id == lrc_id].iloc[0].semantic_field_id
        semantic_row = dfs["lex_semantic_field"][dfs["lex_semantic_field"].id == semantic_id].iloc[0]
        category_row = dfs["lex_semantic_category"][dfs["lex_semantic_category"].id == semantic_row.semantic_category_id].iloc[0]
        return [item.strip(", ") for item in semantic_row.text.split(",") + [category_row.text]]
    except IndexError:
        return []


iffy_languages = set()
not_found_languages = set()


def recover_gpt_reflexes(dfs, lrc_id):
    global iffy_languages
    global not_found_languages
    # get the root
    root = dfs["lex_etyma"][dfs["lex_etyma"].id == lrc_id].iloc[0]["entry"]
    # cross-reference the root to the recovered reflexes
    reflex_df = dfs["missing_forms"][dfs["missing_forms"].root == root]
    # loop through them to get all the entries
    entries = []
    for i, row in reflex_df.iterrows():
        # fixme: I am not sure about these, but I want to keep working
        language_override = {
            "Modern Persian": "New Persian",
            "Old Church Slavic": "Old Slavic",
        }
        override_language_name = row.language in language_override
        row.language = language_override.get(row.language, row.language)

        # try to get the language from the lex_language, and if it doesn't exist try to find the closest match.
        # maybe the first letter of each word is not capitalized?
        language = None
        if row.language.lower() in dfs["lex_language"]["name"].str.lower().to_list():
            language_series = dfs["lex_language"][dfs["lex_language"]["name"].str.lower() == row.language.lower()]
            language = get_reflex_language(dfs, language_series.iloc[0].id)

        # if all else fails try each separate
        if language is None:
            # try splitting on some things (dash, space), and if one of those matches (starting from the end) then use that
            split_language = [lang.strip() for lang in re.split(r"[- ]", row.language)]
            found_languages = [
                dfs["lex_language"][dfs["lex_language"]["name"].str.lower() == language.lower()]
                for language in split_language
                if language.lower() in dfs["lex_language"]["name"].str.lower().to_list()
            ]
            # check if language_series (a dataframe) is not empty
            if len(found_languages) > 0:
                language_series = found_languages[-1]
                language = get_reflex_language(dfs, language_series.iloc[0].id)
                iffy_languages.add(row.language)
                override_language_name = True
            pass

        # manual overrides for things I know are wrong but cannot fix
        # todo: FIX THESE
        if language is None:
            not_found_languages.add(row.language)
            language = {'language_name': row.language, 'language_abbr': row.language, 'langauge_aka': '', 'family_name': 'MISSING', 'family_name_override': '', 'sub_family_name': 'MISSING', 'is_sub_branch': False, 'is_sub_sub_branch': False}

        if override_language_name:
            language["language_name"] = row.language
            language["language_abbr"] = row.language
        entry = {
            # language/family
            "language": language,
            # reflex itself (may contain multiple), store as list
            "reflexes": [row.reflex],
            # part of speech
            "pos": [{'code': 'TBD', 'meaning': "To Be Determined"}],
            # the meaning (gloss), we use the translation if it is available, and default to the original german otherwise
            "gloss": row["meaning"] if row["F translated"] == "" else row["F translated"],
            # source (generally IEW, but we will credit UKY and GPT as db source?)
            # todo: figure out who to actually credit.
            "source": {
                # just statically set it I guess.
                "text_sources": [{'code': 'IEW', 'display': 'Julius Pokorny: Indogermanisches etymologisches Wörterbuch (1959)'}],
                # I guess we credit gpt and uky?
                "db_sources": [{'code': 'UKY', 'display': 'Dr. Andrew Byrd, University of Kentucky (fact checking GPT\'s output)'}, {'code': 'GPT', 'display': 'GPT-4 (organizing text from Pokorny)'}]
            },
        }
        entries.append(entry)
    return entries


def get_reflex_entries(dfs, lrc_id):
    reflex_ids = dfs['lex_etyma_reflex'][dfs['lex_etyma_reflex']['etyma_id'] == lrc_id]["reflex_id"].tolist()
    reflex_df = dfs["lex_reflex"][dfs["lex_reflex"].id.isin(reflex_ids)]
    if len(reflex_df) == 0:
        return recover_gpt_reflexes(dfs, lrc_id)

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


# exploratory function :)
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


def expand_parenthesized_root(word):
    # Step 1: Extract all parenthesized substrings and store them in replacements list
    replacements = re.findall("\((.*?)\)", word)

    # Step 2: Create a template string by replacing all parenthesized parts with curly braces "{}"
    replace_str = re.sub("(\(.*?\))", "{}", word)

    len_replacements = len(replacements)

    # Step 3: Generate all combinations using binary counting
    binary_strings = [format(i, f'0{len_replacements}b') for i in range(1 << len_replacements)]
    combinations = [
        replace_str.format(*["" if bit == "0" else paren_part for bit, paren_part in zip(binary_string, replacements)])
        for binary_string in binary_strings
    ]

    return combinations


def expand_hyphenated_root(word):
    if "-" not in word:
        return [word]
    split_hyphen = word.split("-")
    return ["".join(split_hyphen[:i+1]) for i in range(len(split_hyphen))]


def root_conditioning(roots):
    roots = roots.replace("<sup>u̯</sup>", "ʷ").replace("h", "ʰ").replace("k̑", "ḱ").replace("g̑", "ǵ")
    return roots


def root_troubling_stop_words(roots):
    roots.replace(" or ", " ").replace(" it ", " ").replace(" on ", " ").replace(" to ", " ").replace(" of ", " ")
    return roots


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
    dfs["missing_forms"] = get_missing_forms()

    # the actual entries
    pokorny_entries = []
    common_entries = []
    pokorny_by_id = {}
    # for storing the entries in a way that can be easily cross_referenced
    lrc_to_pokorny_id = {}
    # for linking cross-references on a second pass
    entry_to_entry = defaultdict(set)
    for i, row in tqdm(list(dfs["lex_etyma"].iterrows()), ncols=150):
        lrc_id = row['id']

        # link to reflexes
        reflex_entries = get_reflex_entries(dfs, lrc_id)

        # get semantic info
        semantic = get_semantic(dfs, lrc_id)

        # basic info
        # I have to replace the "or" and other short words carefully otherwise it might catch some real roots
        roots = row["entry"].strip("\n\t ").replace("<p>", "").replace("</p>", "")
        # condition the roots (some very specific patterns need to be replaced)
        roots = root_conditioning(roots)
        # remove the troubling stop words
        search_roots = root_troubling_stop_words(roots)
        # turn into an actual list (while removing html and striping some chars)
        search_roots = [remove_html_tags_from_text(root).strip(" ,-") for root in re.split("(,|\s|:|\n)", search_roots)]
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

        remove_words = ['Balto-Slavic', 'Celtic', 'Indo', 'Indo-Iranian', 'Iranian', 'ablative', 'accurately', 'adj.', 'also', 'and', 'base', 'based', 'better',
                        'before', 'broken', 'case', 'chiefly', 'compare', 'etc.', 'extended', 'extension', 'fem.', 'from', 'form', 'genitive', 'genitive-ablative', 'grade',
                        'heavy', 'heavy-base', 'lengthened', 'locative', 'masc.', 'more', 'n-stem', 'nasalized', 'nasals', 'oblique', 'occasionally', 'particle', 'plural',
                        'possibly', 'presumably', 'probably', 'reduced', 'reduplicated', 'reduplication', 'root', 'simplified', 'suffixes', 'the', 'thematic',
                        'weak', 'which', 'with', '(masc.)', '(fem.)', "(from"]
        split_roots = [
            root
            for condensed_root in search_roots
            if condensed_root.strip() not in remove_words
            for expanded_root in expand_parenthesized_root(condensed_root)
            for root in expand_hyphenated_root(expanded_root)
            if root != ""
            and not root.strip(". -").isnumeric()
            and root.strip() not in remove_words
        ]

        entry = {
            "entry_id": entry_id,
            "root": roots,
            "meaning": gloss,
            "reflexes": reflex_entries,
            # todo: part of speech
            "pos": [],
            "semantic": semantic,
            # this id is deleted on the second pass
            "lrc_id": lrc_id,
            "searchable_roots": " ".join(split_roots)
        }
        pokorny_entries.append(entry)
        pokorny_by_id[entry_id] = entry
        lrc_to_pokorny_id[lrc_id] = entry_id

        common_entry = {
            "entry_id": entry_id,
            "root": roots,
            "meaning": gloss,
            "pos": None,
        }
        common_entries.append(common_entry)

    # todo: get rid of these when they are no longer needed
    global not_found_languages
    global iffy_languages
    not_found_languages = sorted(not_found_languages)
    iffy_languages = sorted(iffy_languages)
    print(f"{not_found_languages=}")
    print(f"{iffy_languages=}")

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
        json.dump(pokorny_entries_new, fp, indent=2)

    with open("data_pokorny/table_common.json", "w") as fp:
        json.dump(common_entries, fp)
    pass


def get_missing_forms():
    abbr_df = load_abbreviations_df()
    # every field is to be interpreted as a string
    dfs = [
        pd.read_csv(file, dtype=str).fillna("")
        for file in glob.glob("data_pokorny/gpt_corrections/*.csv")
    ]
    # drop the first column of each
    dfs = [df.drop(columns=df.columns[0]) for df in dfs]

    # some of these are similar but need to be "realigned"
    # which just means that the columns need to be shifted, root onwards, over by 1
    for i in range(0, 2):
        old_cols = dfs[i].columns
        dfs[i]["web_root"] = ""
        # rearrange the columns
        dfs[i] = dfs[i][["web_root"] + list(old_cols)]

    # copy the column names from -1 to 0 and 1
    dfs[1].columns = dfs[-1].columns
    dfs[0].columns = dfs[-1].columns

    # concat
    df = pd.concat(dfs, ignore_index=True)
    # build a dictionary translating from df.abbr to df.language, using the abbr_df["German abbreviation"]: abbr_df["English"] for each unique abbr
    not_found = set()
    abbreviation_to_english = {}
    for abbr in df.abbr.unique():
        abbr_df_row = abbr_df[abbr_df["German abbreviation"] == abbr]
        if abbr in ["", "note"]:
            continue
        if len(abbr_df_row) == 0:
            # try all lower?
            abbr_df_row = abbr_df[abbr_df["German abbreviation"] == abbr.lower()]
            pass
        # try splitting on space
        # todo: this might be a bad idea. better to handle them individually
        if len(abbr_df_row) == 0:
            for split_abbr in abbr.split(" "):
                abbr_df_row = abbr_df[abbr_df["German abbreviation"] == split_abbr]
                if len(abbr_df_row) > 0:
                    break
        # if it doesn't have a . at the end, add it and try again
        if len(abbr_df_row) == 0 and not abbr.endswith("."):
            abbr_df_row = abbr_df[abbr_df["German abbreviation"] == abbr + "."]
        # if still nothing, skip
        if len(abbr_df_row) == 0:
            not_found.add(abbr)
            continue
        abbr_df_row = abbr_df_row.iloc[0]
        if abbr not in abbreviation_to_english:
            abbreviation_to_english[abbr] = abbr_df_row["English"]
        else:
            if abbreviation_to_english[abbr] != abbr_df_row["English"]:
                breakpoint()

    # apply the abbreviation_to_english to the df.abbr column and store in the language column
    df["language"] = df["abbr"].apply(lambda x: abbreviation_to_english.get(x, "Language not found"))
    return df


if __name__ == '__main__':
    main()
    pass
