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


def find_missing_pokorny():
    dfs = {os.path.splitext(os.path.basename(df_file))[0]: pd.read_pickle(df_file) for df_file in glob.glob("data_pokorny/table_dumps/*.df")}

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

    # etyma_to_reflex = dfs['lex_etyma_reflex'].groupby("etyma_id")["reflex_id"].apply(list).to_dict()

    # figure out what entries do not have reflexes
    missing_ids = sorted(set(dfs["lex_etyma"].id.unique()) - set(dfs['lex_etyma_reflex'].etyma_id.unique()))
    missing_entries = dfs['lex_etyma'][dfs['lex_etyma'].id.isin(missing_ids)]
    complete_entries = dfs['lex_etyma'][~dfs['lex_etyma'].id.isin(missing_ids)]

    with open("data_pokorny/pokorny_scraped.json", "r", encoding="utf-8") as fp:
        scraped_pokorny = json.load(fp)

    # build a prompt for gpt for each
    output_dfs = []
    output_digests = set()
    for counter, (i, row) in enumerate(missing_entries.iterrows()):
        # get some information
        material = "\n".join(scraped_pokorny[i]['Material']).replace("`", "'")
        # reflexes = get_reflex_entries(dfs, row.id)
        remove_html_tags_from_text(row.entry).strip("\n")
        root = remove_html_tags_from_text(row.entry).strip("\n")
        gloss = remove_html_tags_from_text(row.gloss)

        abbreviations_used = augment_material(material, abbreviation_data)
        # continue

        # form prompt

        if i < 317:
            prompt = format_gpt_prompt_old(root, gloss, material)
        else:
            prompt = format_gpt_prompt(root, gloss, material, abbreviations_used)

        # ask gpt
        print(f"{counter+1:5}|{root:40}", end=" | ")
        if material.strip() == "":
            print("missing material - SKIPPING")
            continue
        response = query_gpt_fake(prompt)

        # check if we made a mistake and double pasted or something
        output_digest = get_text_digest(response)
        if output_digest in output_digests:
            breakpoint()
        output_digests.add(output_digest)

        df = process_gpt(response)
        df["root"] = root
        output_dfs.append(df)
    combined_df = pd.concat(output_dfs)
    combined_df = combined_df[['root', 'abbr', 'reflex', 'meaning', 'notes']]
    combined_df.to_csv("sample.csv")
    pass


def decap(word):
    return word[:1].lower() + word[1:]


def decap_isin(word, sequence):
    return sequence.find(word) != -1 or sequence.find(decap(word)) != -1


def augment_material(material, abbreviation_data):
    # found_matches2 = abbreviation_data[abbreviation_data['German abbreviation'].apply(lambda x: decap_isin(x.strip(), material.replace("\\", "")))]
    found_matches = abbreviation_data[abbreviation_data['German abbreviation'].apply(lambda x: x.strip().lower()).isin(re.split(r'[, \n()]', material.replace("\\", "").lower()))]

    return found_matches


def format_gpt_prompt(root, gloss, material, abbreviations_used):
    german_abbr = "\n".join([f'- {row["German abbreviation"]}: {row["German"]}' for i, row in abbreviations_used.iterrows()])
    prompt = f'I will give you a mostly german block of text for the Proto-Indo-European reconstructed root "{root}" meaning "{gloss}". '
    prompt += 'The text will generally contain a language abbreviation, a reflex, and a german meaning. '
    prompt += 'The language abbreviation is in German, so if you recognize the text as other german text it is likely notes for that entry. ' \
              f'Here is a list of abbreviations and their German meaning that might be helpful.\n{german_abbr}\n'
    prompt += 'A reflex is an attested word from which a root in the Proto-Indo-European is reconstructed from. ' \
              'You may find that multiple reflexes exist per line for a given language abbreviation, you should treat these are separate entries. '
    prompt += 'Some of the reflexes are already surrounded by backslashes, which is used to indicate italic text, but others are not and you cannot rely on this. '
    prompt += f'If the german meaning is missing fill that field with the meaning of the root (in this case "{gloss}"). '
    prompt += 'In comma separated format, give me the language abbreviation, the reflex, the extracted german meaning, and whatever notes you may have found in the text for that entry. '
    prompt += 'All of these are listed in the text block. '
    prompt += ' \n \n'
    prompt += re.sub(r'\\\\+', r'\\', material)
    prompt += ' \n \n'
    prompt += 'Remember to give these in CSV format, without any other text. ' \
              'The CSV should have columns for the language abbreviation, the reflex, the untranslated german meaning, and the notes for that entry. ' \
              'Put quotes around each field to preserve commas and remove any black slashes you find. '
    # prompt += 'Here is an example'
    return prompt


def format_gpt_prompt_old(root, gloss, material):
    prompt = f'I will give you a mostly german block of text for the Proto-Indo-European reconstructed root "{root}" meaning "{gloss}". '
    prompt += 'The text will generally contain a language abbreviation, a reflex, and a german meaning. '
    prompt += 'The language abbreviation is in German, so if you recognize the text as other german text it is likely notes for that entry. '
    prompt += 'A reflex is an attested word from which a root in the Proto-Indo-European is reconstructed from. ' \
              'You may find that multiple reflexes exist per line for a given language abbreviation, you should treat these are separate entries. '
    prompt += 'Some of the reflexes are already surrounded by double backslashes, which is used to indicate italic text, but others are not and you cannot rely on this. '
    prompt += f'If the german meaning is missing fill that field with the meaning of the root (in this case "{gloss}"). '
    prompt += 'In comma separated format, give me the language abbreviation, the reflex, the extracted german meaning, and whatever notes you may have found in the text for that entry. '
    prompt += 'All of these are listed in the text block. '
    prompt += ' \n \n'
    prompt += material
    prompt += ' \n \n'
    prompt += 'Remember to give these in CSV format, without any other text. ' \
              'The CSV should have columns for the language abbreviation, the reflex, the untranslated german meaning, and the notes for that entry. ' \
              'Put quotes around each field to preserve commas and remove any black slashes you find. '
    # prompt += 'Here is an example'
    return prompt


def format_gpt_prompt2(root, gloss, material):
    prompt = ""
    prompt += f'I am going to give you a block of text from a german dictionary on Proto-Indo-European. ' \
              f'This text is about the root "{root}" which in Proto-Indo-European means "{gloss}" and contains information about the words used to reconstruct this root (called reflexes). ' \
              f'This information includes: an abbreviation of the language the reflex is from, the reflexes in that language, sometimes the meanings of those reflexes (other times the meaning is the same as the root meaning and thus is not included), and some notes and comments (such as the author stating that they are not sure, citing their sources, and other miscellaneous comments). ' \
              f'There is generally only a single language abbreviation per line, but each line may have multiple reflexes (some of which may share a meaning). ' \
              f'To help with this I have added backslashes around some of them, but not all of them. ' \
              f'If a word is not in german it is likely to be a reflex. ' \
              f'I need this information in a more organized format, in CSV format. ' \
              f'Here is the header of the CSV containing the column names, please include it in your response, along with the information for each column:\n' \
              f'\n' \
              f'"Language Abbreviation","Reflex","German Meaning","Notes"\n' \
              f'\n' \
              f'Please extract this information for me. Remember to quote every field to preserve commas and to put each reflex on its own line. Here is the text:\n' \
              f'\n' \
              f'{material}'

    return prompt


if __name__ == '__main__':
    # main()
    find_missing_pokorny()
    pass
