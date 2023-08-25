import time
from collections import defaultdict

import openai

from gpt_utils import query_gpt_fake, query_gpt, delete_response

"""
author - anton vinogradov

Translates text from 'data_pokorny/german_meanings.txt' into english and places it into 'data_pokorny/english_meanings.txt', completely mirroring the lines in
the german one. It also filters for exact duplicates as to cut down on queries and removes empty lines (but adds them back when making the english txt file)
though this is partially overlapping in functionality with the caching system. If there is an issue and GPT gives an answer that is outside of the format that
the prompt asks for, you can manually change the cache json. Sort by date modified :)

The ideal usage is to copy the column that you want translated, and paste into german_meanings.txt, then when it is done copy the data in english_meanings.txt 
and paste over the column you still have selected in google sheets or excel or whatever. This usage is why it reconstructs the exact columns in the english, 
even if some are empty.
"""


def main(in_file="data_pokorny/german_meanings.txt", out_file="data_pokorny/english_meanings.txt"):
    # Open the file and read all lines into a list, stripping whitespace
    with open(in_file, "r", encoding="utf-8") as fp:
        lines = [line.strip() for line in fp.readlines()]

    line_to_index = defaultdict(list)
    for i, line in enumerate(lines):
        line_to_index[line].append(i)

    # Create a set, remove empty strings, and sort the entries to turn it back into a list with a consistent ordering
    entries_set = set(lines)
    entries_set.discard('')  # Remove empty strings
    sorted_entries = sorted(entries_set, key=lambda x: line_to_index[x])

    entry_match = [""] * len(lines)
    for i, german_meaning in enumerate(sorted_entries):

        output, prompt = try_gpt_translate(german_meaning, sorted_entries, i)
        message_content = output["choices"][0]["message"]["content"]
        try:
            english_meaning = message_content.split("|")[1].strip(' "')
        except Exception as err:
            print("Got exception, removing last attempt and trying a second time", err)
            delete_response(prompt)
            # try a second time, but if this fails we just let it crash.
            output, prompt = try_gpt_translate(german_meaning, sorted_entries, i)
            message_content = output["choices"][0]["message"]["content"]
            english_meaning = message_content.split("|")[1].strip(' "')
            # if you want to try a third time just rerun the script, it will remove the bad response and query once more
            pass

        print("\t", english_meaning)
        for line_num in line_to_index[german_meaning]:
            entry_match[line_num] = english_meaning.strip(" \n")
        # time.sleep(1)

    with open(out_file, "w", encoding="utf-8") as fp:
        fp.write("\n".join(entry_match))


def try_gpt_translate(german_meaning, sorted_entries, i):
    prompt = f'Translate to English from German: "{german_meaning}"\n' \
             f'Reply with a single line in the following format: Original Text | English translation\n' \
             f'Only translate the german, if you see text in any other language (such as greek or english), leave that in its original form and translate only the words around it. If there is no german, or you cannot translate it, just repeat the original text followed by \"[UNKNOWN]\". ' \
             f"For example: \"Machek (Slavia 16, 174) nimmt als ursprüngl. Bedeutung 'mager' an, das er somit zu ai. \kṣudhyati\ 'hungert', \kṣōdh-uka-\ 'hungrig' stellen möchte.\" translates to \"Machek (Slavia 16, 174) nimmt als ursprüngl. Bedeutung 'mager' an, das er somit zu ai. \kṣudhyati\ 'hungert', \kṣōdh-uka-\ 'hungrig' stellen möchte. | Machek (Slavia 16, 174) assumes 'mager' as the original meaning, which he thus relates to Sanskrit \kṣudhyati\ 'hungers', \kṣōdh-uka-\ 'hungry'.\"\n" \
             f"For example: \"aus *sulak'\" translates to \"aus *sulak' | from *sulak'\"\n" \
             f"For example: \"*k̂u̯oinā\" translates to \"*k̂u̯oinā | *k̂u̯oinā [UNKNOWN]\", note here how I need the original repeated as well as the \"[UNKNOWN]\" marker.\n" \
             f"For example: \"refl.\" translates to \"refl. | refl. [UNKNOWN]\", note here how I need the original repeated as well as the \"[UNKNOWN]\" marker.\n" \
             f"For example: \"*skuftu-; m.\" translates to \"*skuftu-; m. | *skuftu-; masculine.\"\n" \
             f"Remember to stick to the format of \"Original Text | Translation\", the \"|\" character is necessary."
    print(f"{i+1:>3}/{len(sorted_entries)} | {german_meaning:<60}"[:60], end=" | ")
    # only reattempt 10 times as to not be mean to openai
    output = None
    for _ in range(10):
        try:
            # output = query_gpt_fake(prompt)
            output = query_gpt(prompt)
            break
        except (openai.error.ServiceUnavailableError, openai.error.APIError, openai.error.Timeout) as err:
            print("---===>>> Could not connect, waiting 15 seconds and trying again... <<<===---")
            time.sleep(15)
            continue
    assert output is not None
    return output, prompt


if __name__ == '__main__':
    main()
    pass
