import time
from collections import defaultdict

import openai

from gpt_utils import query_gpt_fake, query_gpt


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


def main():
    # Open the file and read all lines into a list, stripping whitespace
    with open("data_pokorny/german_meanings.txt", "r", encoding="utf-8") as fp:
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
        prompt = f'Translate to English from German: "{german_meaning}"\n' \
                 f'Reply with a single line in the following format: Original Text | English translation'
        print(f"{i:>3}/{len(sorted_entries)} | {german_meaning:<60}"[:60], end=" | ")

        # only reattempt 50 times as to not be mean to openai
        output = None
        for _ in range(10):
            try:
                # output = query_gpt_fake(prompt)
                output = query_gpt(prompt)
                break
            except openai.error.ServiceUnavailableError as err:
                print("---===>>> Could not connect, waiting 15 seconds and trying again...")
                time.sleep(15)
                continue

        assert output is not None

        english_meaning = output["choices"][0]["message"]["content"].split("|")[1].strip(' "')
        print("\t", english_meaning)
        for line_num in line_to_index[german_meaning]:
            entry_match[line_num] = english_meaning
        # time.sleep(1)

    with open("data_pokorny/english_meanings.txt", "w", encoding="utf-8") as fp:
        fp.write("\n".join(entry_match))


if __name__ == '__main__':
    main()
    pass
