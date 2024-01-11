import glob
import json
import os
import re
import sys
import time
import warnings

import openai
import pandas as pd
import pyperclip

import gpt_functions
from generate_pokorny_db_data import remove_html_tags_from_text, load_abbreviations_df
from gpt_utils import process_gpt, query_gpt_fake, get_text_digest, get_digest


"""
author - anton vinogradov
This script serves two purposes:
1. identify all the missing pokorny forms that cannot be processed via gpt
2. query gpt with the ones that can

Querying gpt is unfortunately not possible automatically as I did not have gpt4 access when I created this script.
Despite this I have a work around, but it requires the user to manually go to chatgpt and paste in the prompt (the script will load your clipboard for you) and
manually copy GPT's output (the script will read the clipboard for you too). Ideally this script will never be rerun, but if this is being used as reference by
someone other than me, I have left comments where I can. Ask ChatGPT to explain the rest of the code, it was written with its help after all.  

If it is rerun, run it in a debugger. I make use of breakpoint() to pause execution to allow the user to paste and wait on GPT. Running this code often leads
to edge cases so having a debugger at the ready helped with that, and if you run this code yourself expect that an error will occur and that you will need to
write more code to handle that edge case. 
"""


# region prompts

def format_gpt_prompt_new2(root, gloss, material, abbreviations_used, headers):
    headers_str = str(headers)[1:-1]
    headers_str = headers_str.replace("'", '"')
    aug_prompt = re.sub(r'\\\\+', r'\\', material)
    aug_prompt = re.sub(r'__+', r'_', aug_prompt)
    german_abbr = "\n".join([f'- {row["German abbreviation"]}: {row["German"]}' for i, row in abbreviations_used.iterrows()])
    prompt = f'I will give you a mostly german block of text for the Proto-Indo-European reconstructed root "{root}" meaning "{gloss}". '
    prompt += 'The text will generally contain a language abbreviation, a reflex, and a german meaning. '
    prompt += 'The language abbreviation is in German, so if you recognize the text as other german text it is likely notes for that entry. ' \
              'You may find that multiple languages listed for a single set of reflexes, when you see this you should split these into separate lines. ' \
              'Each line should only have a single language. ' \
              f'Here is a list of abbreviations and their German meaning that might be helpful.\n{german_abbr}\n'
    prompt += 'A reflex is an attested word from which a root in the Proto-Indo-European is reconstructed from. ' \
              'You may find that multiple reflexes exist per line for a given language abbreviation, you should treat these are separate entries. ' \
              'You will never find a reflex that matches an abbreviation.'
    prompt += 'Some of the reflexes are already surrounded by backslashes, which is used to indicate italic text, but others are not and you cannot rely on this. '
    prompt += f'If the german meaning is missing fill that field with the meaning of the root (in this case "{gloss}"). '
    prompt += 'In comma separated format, give me the language abbreviation, the reflex, the extracted german meaning, and whatever notes you may have found in the text for that entry. '
    prompt += 'All of these are listed in the text block. '
    prompt += ' \n \n'
    prompt += aug_prompt
    prompt += ' \n \n'
    prompt += (
        'Remember to give these in CSV format, without any other text. '
        'The CSV should have columns for the language abbreviation, the reflex, the untranslated german meaning, and the notes for that entry. '
        'Put quotes around each field to preserve commas and remove any black slashes you find. '
        'Put the CSV in a codeblock marked with ``` so I can read it easily. '
    )
    prompt += f'Here are the headers, continue on from here: \n{headers_str}'
    return prompt

def format_gpt_prompt_new(root, gloss, material, abbreviations_used, headers):
    headers_str = str(headers)[1:-1]
    headers_str = headers_str.replace("'", '"')
    aug_prompt = re.sub(r'\\\\+', r'\\', material)
    aug_prompt = re.sub(r'__+', r'_', aug_prompt)
    german_abbr = "\n".join([f'- {row["German abbreviation"]}: {row["German"]}' for i, row in abbreviations_used.iterrows()])
    prompt = f'I will give you a mostly german block of text for the Proto-Indo-European reconstructed root "{root}" meaning "{gloss}". '
    prompt += 'The text will generally contain a language abbreviation, a reflex, and a german meaning. '
    prompt += 'The language abbreviation is in German, so if you recognize the text as other german text it is likely notes for that entry. ' \
              'You may find that multiple languages listed for a single set of reflexes, when you see this you should split these into separate lines. ' \
              'Each line should only have a single language. ' \
              f'Here is a list of abbreviations and their German meaning that might be helpful.\n{german_abbr}\n'
    prompt += 'A reflex is an attested word from which a root in the Proto-Indo-European is reconstructed from. ' \
              'You may find that multiple reflexes exist per line for a given language abbreviation, you should treat these are separate entries. ' \
              'You will never find a reflex that matches an abbreviation.'
    prompt += 'Some of the reflexes are already surrounded by backslashes, which is used to indicate italic text, but others are not and you cannot rely on this. '
    prompt += f'If the german meaning is missing fill that field with the meaning of the root (in this case "{gloss}"). '
    prompt += 'In comma separated format, give me the language abbreviation, the reflex, the extracted german meaning, and whatever notes you may have found in the text for that entry. '
    prompt += 'All of these are listed in the text block. '
    prompt += ' \n \n'
    prompt += aug_prompt
    prompt += ' \n \n'
    prompt += 'Remember to give these in CSV format, without any other text. ' \
              'The CSV should have columns for the language abbreviation, the reflex, the untranslated german meaning, and the notes for that entry. ' \
              'Put quotes around each field to preserve commas and remove any black slashes you find. '
    prompt += f'Here are the headers, continue on from here: \n{headers_str}'
    return prompt


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

# endregion


def augment_material(material, abbreviation_data):
    found_matches = abbreviation_data[abbreviation_data['German abbreviation'].apply(lambda x: x.strip().lower()).isin(re.split(r'[, \n()]', material.replace("\\", "").lower()))]

    return found_matches


def find_missing_pokorny():
    dfs = {os.path.splitext(os.path.basename(df_file))[0]: pd.read_pickle(df_file) for df_file in glob.glob("data_pokorny/table_dumps/*.df")}

    abbreviation_data = load_abbreviations_df()

    # etyma_to_reflex = dfs['lex_etyma_reflex'].groupby("etyma_id")["reflex_id"].apply(list).to_dict()

    # figure out what entries do not have reflexes
    missing_ids = sorted(set(dfs["lex_etyma"].id.unique()) - set(dfs['lex_etyma_reflex'].etyma_id.unique()))
    missing_entries = dfs['lex_etyma'][dfs['lex_etyma'].id.isin(missing_ids)]
    complete_entries = dfs['lex_etyma'][~dfs['lex_etyma'].id.isin(missing_ids)]

    headers = ['abbr', 'reflex', 'meaning', 'notes']

    with open("data_pokorny/pokorny_scraped.json", "r", encoding="utf-8") as fp:
        scraped_pokorny = json.load(fp)

    # build a prompt for gpt for each
    output_dfs = []
    output_digests = set()

    # some are out of order, so I have a dict to manually realign between UTexas and the pokorny site
    reorder_dict = {}
    manual_skips = [2007]
    for counter, (i, row) in enumerate(missing_entries.iterrows()):
        old_i = i
        if old_i in manual_skips:
            continue

        # Attempting to solve the misaligned entries issue by adding more and more offsets when I see them.
        if counter >= 239:
            i = old_i - 3
        if counter >= 429:
            i = old_i - 2
        if counter >= 430:
            i = old_i - 4
        if counter >= 500:
            i = old_i - 5

        if counter >= 606:
            i = old_i - 6

        if counter >= 663:
            i = old_i - 1

        if old_i in reorder_dict:
            i = reorder_dict[old_i]

        # get some information
        material = "\n".join(scraped_pokorny[i]['Material']).replace("`", "'")
        web_root = scraped_pokorny[i]["root"]
        texas_root = remove_html_tags_from_text(row.entry).strip("\n")
        gloss = remove_html_tags_from_text(row.gloss)

        abbreviations_used = augment_material(material, abbreviation_data)

        # form prompt (which has changed over time and I do not want to invalidate the caches on work already done)
        if i < 317:
            prompt = format_gpt_prompt_old(texas_root, gloss, material)
        elif i < 766:
            prompt = format_gpt_prompt(texas_root, gloss, material, abbreviations_used)
        else:
            prompt = format_gpt_prompt_new(texas_root, gloss, material, abbreviations_used, headers)

        # ask gpt
        print(f"{counter+1:5}|{texas_root:20}", end=" | ")
        if material.strip() == "":
            print("missing material - SKIPPING")
            continue

        response = query_gpt_fake(prompt)

        # check if we made a mistake and double pasted or something
        output_digest = get_text_digest(response)
        if output_digest in output_digests:
            print("I got the same data twice which is not expected. ",
                  "If you are expecting this then you must update the source code, as this is currently not allowed. ",
                  "If this was a mistake you must find and delete the most recent cache in the gpt_caches/ folder. ",
                  f"Look for {get_digest(prompt)}",
                  "The program will exit once you (c)ontinue from here.")
            breakpoint()
            exit(-1)
        output_digests.add(output_digest)

        df = process_gpt(response, headers)
        df["root"] = texas_root
        df["web_root"] = web_root[-1]
        output_dfs.append(df)
    # combined_df = pd.concat(output_dfs)
    combined_df = pd.concat(output_dfs)
    combined_df = combined_df[['web_root', 'root'] + headers]
    combined_df.to_csv("data_pokorny/recovered_pokorny_reflexes.csv")
    pass


def find_unrecoverable_pokorny():
    dfs = {os.path.splitext(os.path.basename(df_file))[0]: pd.read_pickle(df_file) for df_file in glob.glob("data_pokorny/table_dumps/*.df")}
    missing_ids = sorted(set(dfs["lex_etyma"].id.unique()) - set(dfs['lex_etyma_reflex'].etyma_id.unique()))
    missing_entries = dfs['lex_etyma'][dfs['lex_etyma'].id.isin(missing_ids)]

    with open("data_pokorny/pokorny_scraped.json", "r", encoding="utf-8") as fp:
        scraped_pokorny = json.load(fp)

    reorder_dict = {}
    manual_skips = [2007]

    unrecoverable_entries = []

    for counter, (i, row) in enumerate(missing_entries.iterrows()):
        old_i = i
        if old_i in manual_skips:
            continue

        if counter >= 239:
            i = old_i - 3
        if counter >= 429:
            i = old_i - 2
        if counter >= 430:
            i = old_i - 4
        if counter >= 500:
            i = old_i - 5

        if counter >= 606:
            i = old_i - 6

        if counter >= 663:
            i = old_i - 1

        if old_i in reorder_dict:
            i = reorder_dict[old_i]

        material = "\n".join(scraped_pokorny[i]['Material']).replace("`", "'")

        if material.strip() == "":
            unrecoverable_entries.append((row.entry, row.gloss))
    print(unrecoverable_entries)
    json_unrecoverable = dict(unrecoverable_entries)

    with open("data_pokorny/unrecoverable_pokorny.json", "w") as fp:
        json.dump(json_unrecoverable, fp, indent=2)
    pass


def find_non_gpt_pokorny():
    pokorny_filename = "data_pokorny/table_pokorny.json"
    with open(pokorny_filename, 'r') as fp:
        pokorny_data_list = json.load(fp)
    allow = ["GPT"]
    deny = []
    filtered = []
    for entry in pokorny_data_list:
        pass_test = True
        root = entry["root"]
        sources = []
        for reflex in entry["reflexes"]:
            for source in (reflex["source"]["text_sources"] + reflex["source"]["db_sources"]):
                sources.append(source['code'])
        in_allow = set(allow).issubset(set(sources))
        in_deny = set(deny).issubset(set(sources)) if len(deny) > 0 else False
        if not in_allow or in_deny:
            pass_test = False
        if pass_test:
            filtered.append(root)
    # print(f"{len(filtered)}/{2222} (= {(len(filtered)/2222)*100:0.2f}%)")
    return filtered


def do_remaining_pokorny():
    dfs = {os.path.splitext(os.path.basename(df_file))[0]: pd.read_pickle(df_file) for df_file in glob.glob("data_pokorny/table_dumps/*.df")}

    abbreviation_data = load_abbreviations_df()

    # etyma_to_reflex = dfs['lex_etyma_reflex'].groupby("etyma_id")["reflex_id"].apply(list).to_dict()

    line_up_df = get_web_lrc_lineup_corrected()

    # figure out what entries do not have reflexes
    missing_ids = sorted(set(dfs["lex_etyma"].id.unique()) - set(dfs['lex_etyma_reflex'].etyma_id.unique()))
    missing_entries = dfs['lex_etyma'][dfs['lex_etyma'].id.isin(missing_ids)]
    complete_entries = dfs['lex_etyma'][~dfs['lex_etyma'].id.isin(missing_ids)]

    headers = ['abbr', 'reflex', 'meaning', 'notes']

    with open("data_pokorny/pokorny_scraped.json", "r", encoding="utf-8") as fp:
        scraped_pokorny = json.load(fp)

    # turn scraped pokorny into a dictionary
    scraped_pokorny_dict = {
        ", ".join(entry["root"]): entry
        for entry in scraped_pokorny
    }

    # build a prompt for gpt for each
    output_dfs = []
    output_digests = set()

    # some are out of order, so I have a dict to manually realign between UTexas and the pokorny site
    reorder_dict = {}
    manual_skips = [2007]
    prompt_lengths = []
    how_many_processed = 0
    how_many_valid = 0
    num_gpt35 = 0
    num_gpt4 = 0
    num_gpt4_errored = 0
    num_errored = 0
    num_too_long = 0
    parse_errored_entries = []

    # break out the entries to be processed
    gpt4_char_threshold = 4500
    entries_to_be_processed = list(enumerate(complete_entries.iterrows()))
    # entries_to_be_processed = list(enumerate(complete_entries.iterrows()))[:21]

    for counter, (i, row) in entries_to_be_processed:
        # get some information
        texas_root = remove_html_tags_from_text(row.entry).strip("\n")

        web_key = get_web_from_lrc(line_up_df, texas_root)
        if web_key is None:
            # todo: we are going to have to figure out what to do with these extras, and the missing ones
            # breakpoint()
            continue
        web_entry = scraped_pokorny_dict.get(web_key, None)
        if web_entry is None:
            # todo: we are going to have to figure out what to do with these extras, and the missing ones
            # breakpoint()
            continue

        web_root = web_entry["root"]
        gloss = remove_html_tags_from_text(row.gloss)

        material = "\n".join(web_entry['Material']).replace("`", "'")
        abbreviations_used = augment_material(material, abbreviation_data)

        # we cant process things if there are no materials to process
        if material.strip() == "":
            print("missing material - SKIPPING")
            df = pd.DataFrame([
                {
                    'abbr': "missing material",
                    'reflex': "missing material",
                    'meaning': "missing material",
                    'notes': "missing material"
                }
            ])
            df["root"] = texas_root
            df["web_root"] = web_root[-1]
            output_dfs.append(df)
            continue

        # form prompt (which has changed over time and I do not want to invalidate the caches on work already done)
        prompt = format_gpt_prompt_new2(texas_root, gloss, material, abbreviations_used, headers)

        # ask gpt
        # print(f"{counter+1:5}|{texas_root:20}", end=" | ")
        # continue

        how_many_valid += 1

        # count the lengths of the prompts to estimate costs
        prompt_lengths.append(len(prompt))
        # continue

        # print(f"prompt length: {len(prompt)}")
        # modulate the model based on the length of the prompt, anything over 3000 characters gets sent to gpt4.
        if len(prompt) > 15000:
            num_too_long += 1
            df = pd.DataFrame([
                {
                    'abbr': "too long",
                    'reflex': "too long",
                    'meaning': "too long",
                    'notes': "too long"
                }
            ])
            df["root"] = texas_root
            df["web_root"] = web_root[-1]
            output_dfs.append(df)
            continue

        model = "gpt-4" if len(prompt) > gpt4_char_threshold else "gpt-3.5-turbo"

        if model == "gpt-4":
            num_gpt4 += 1
        if model == "gpt-3.5-turbo":
            num_gpt35 += 1

        # attempt to do a thing, if it doesn't work then move on this time
        attempt_success = True
        for attempt in range(2):
            sys.stdout.flush()
            messages, _ = gpt_functions.query_gpt(
                [prompt],
                note=f"{counter}/{len(entries_to_be_processed)} {texas_root}{' '*15}"[:15],
                model=model,
            )
            try:
                content = gpt_functions.get_last_content(messages)
                response = gpt_functions.extract_code_block(content)

                # check if the first line is not valid csv and exclude it if it's not. It is likely to be marking what language the code block is.
                if "," not in response.split("\n")[0]:
                    response = "\n".join(response.split("\n")[1:])

                df = gpt_functions.csv_to_df(response, headers)
                break
            except openai.error.RateLimitError as err:
                time.sleep(60)
                continue
            except (pd.errors.ParserError, ) as err:
                num_errored += 1
                if model == "gpt-4":
                    num_gpt4_errored += 1

                # warnings.warn(f"{err}")
                # digest, _ = gpt_functions.get_proper_digest(model, user_prompts=[prompt])

                # file = f"gpt_caches/{digest}.json"
                # manual_fix(file, headers, prompt)
                # breakpoint()
                df = pd.DataFrame([
                    {
                        'abbr': "errored",
                        'reflex': "errored",
                        'meaning': "errored",
                        'notes': "errored"
                    }
                ])
                break
                # os.remove(file)
                # continue
                # osCommandString = f"\"C:\\Program Files (x86)\\Notepad++\\notepad++.exe\" {file}"
                # os.system(osCommandString)
                # # give a warning message and continue
                # warnings.warn(f"Failed to parse response, investigate/delete the file and rerun. \n check: {digest}.json", stacklevel=2)
                # parse_errored_entries.append(digest)
                # num_errored += 1
                # attempt_success = False
                # break

        if not attempt_success:
            continue

        df["root"] = texas_root
        df["web_root"] = web_root[-1]
        output_dfs.append(df)

        how_many_processed += 1

        stopped = False
        if stopped:
            break
    # combined_df = pd.concat(output_dfs)
    how_many_gpt4 = len([length for length in prompt_lengths if length > gpt4_char_threshold])
    how_many_gpt35 = len(prompt_lengths) - how_many_gpt4
    completion_percentage = counter / len(complete_entries)
    print(
        f"{how_many_processed=}",
        f"{num_errored=}",
        f"{num_too_long=}",
        f"{how_many_valid=}",
        f"{how_many_gpt35=}",
        f"{how_many_gpt4=}",
        f"{num_gpt4_errored=}",
        f"percentage gpt4: {(how_many_gpt4/len(prompt_lengths))*100:03.2f}%",
        f"percentage completed: {completion_percentage*100:03.2f}%",
        f"current cost: ${gpt_functions.get_cost():0.4f}",
        f"projected cost: ${gpt_functions.get_cost()/completion_percentage:0.4f}",
        sep="\n"
    )
    gpt_functions.print_cost()
    if len(output_dfs) > 0:
        for i, sub in enumerate(split_list(output_dfs)):
            combined_df = pd.concat(sub)
            combined_df = combined_df[['web_root', 'root'] + headers]
            combined_df.to_csv(f"data_pokorny/additional_pokorny/additional_pokorny_reflexes_{i}.csv")
    else:
        print("No output")
    # gpt_functions.print_cost()
    breakpoint()


def split_list(input_list, num_outputs=10):
    sublist_size = len(input_list) // num_outputs

    return [
        input_list[i * sublist_size: (i+1) * sublist_size if i != num_outputs-1 else len(input_list)]
        for i in range(num_outputs)
    ]


def manual_fix(file, headers, prompt):
    with open(file, "r") as fp:
        r = json.load(fp)
    pyperclip.copy(prompt)
    breakpoint()
    content = pyperclip.paste()
    response = gpt_functions.extract_code_block(content)
    # try it again to validate
    if "," not in response.split("\n")[0]:
        response = "\n".join(response.split("\n")[1:])

    df = gpt_functions.csv_to_df(response, headers)

    # if it passes validation then we save it
    r["messages"][-1]["content"] = response
    with open(file, "w") as fp:
        json.dump(r, fp)
    return df


def create_web_lrc_lineup():
    dfs = {os.path.splitext(os.path.basename(df_file))[0]: pd.read_pickle(df_file) for df_file in glob.glob("data_pokorny/table_dumps/*.df")}
    missing_ids = sorted(set(dfs["lex_etyma"].id.unique()) - set(dfs['lex_etyma_reflex'].etyma_id.unique()))
    all_entries = dfs['lex_etyma']

    with open("data_pokorny/pokorny_scraped.json", "r", encoding="utf-8") as fp:
        scraped_pokorny = json.load(fp)

    entries_to_be_processed = list(enumerate(all_entries.iterrows()))
    web_to_texas = []
    for counter, (i, row) in entries_to_be_processed:
        if i in range(len(scraped_pokorny)):
            web_root = scraped_pokorny[i]["root"]
        else:
            web_root = []
        texas_root = remove_html_tags_from_text(row.entry).strip("\n")
        web_to_texas.append({
            "web_root": ", ".join(web_root),
            "texas_root": texas_root,
        })
    df = pd.DataFrame(web_to_texas)
    df.to_csv("data_pokorny/web_to_texas.csv")
    # breakpoint()


def get_web_lrc_lineup_corrected():
    # dfs = {os.path.splitext(os.path.basename(df_file))[0]: pd.read_pickle(df_file) for df_file in glob.glob("data_pokorny/table_dumps/*.df")}
    # all_entries = dfs['lex_etyma']
    #
    # with open("data_pokorny/pokorny_scraped.json", "r", encoding="utf-8") as fp:
    #     scraped_pokorny = json.load(fp)

    line_up_df = pd.read_csv("data_common/web_to_texas_corrected.csv", index_col=0).fillna("MISSING")
    return line_up_df


def get_web_from_lrc(line_up_df, lrc_root):
    found_root = line_up_df[line_up_df.texas_root == lrc_root].iloc[0].web_root
    if found_root == "MISSING":
        return None
    return found_root


def get_lrc_from_web(line_up_df, web_root):
    found_root = line_up_df[line_up_df.web_root == ", ".join(web_root)].iloc[0].texas_root
    if found_root == "MISSING":
        return None
    return found_root


if __name__ == '__main__':
    # get_web_lrc_lineup_corrected()
    # create_web_lrc_lineup()
    do_remaining_pokorny()
    # find_unrecoverable_pokorny()
    # find_missing_pokorny()
    pass
