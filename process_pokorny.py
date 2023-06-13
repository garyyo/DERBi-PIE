import json
from collections import Counter

import pyperclip


def process_pos(processed_pokorny, pos_file):
    with open(pos_file, "r") as fp:
        lines = fp.readlines()
    index_to_pos = {int(line.split(":")[0])-1: line.split(":")[-1].strip().replace(",", ";").split(";") for line in lines}
    pos_pokorny = [{**entry, "pos": index_to_pos.get(i, [])} for i, entry in enumerate(processed_pokorny)]
    return pos_pokorny


def output_pos(processed_pokorny):
    newline = '\n'
    text_lines = [
        f"{i+1} - {'; '.join(entry['root']).replace(newline, ' ')}" \
        f"\n\tEnglish Meaning:\t{'; '.join(entry['meaning'])}" \
        f"\n\t German Meaning:\t{'; '.join(entry['other_meaning']['german'])}" \
        f"\n\t     Tagged POS:\t{'; '.join(entry['pos'])}\n\n"
        for i, entry in enumerate(processed_pokorny)
    ]
    with open("data_pokorny/pokorny_pos_output.txt", "w", encoding="utf-8") as fp:
        fp.writelines(text_lines)
    pass


def clipboard(processed_pokorny):
    definitions = [f'{i + 1}: {"; ".join(entry["meaning"])}' for i, entry in enumerate(processed_pokorny) if "; ".join(entry["meaning"]) != ""]
    for i in range(0, len(definitions), 100):
        pyperclip.copy("\n".join(definitions[i:i+100]))
        breakpoint()


def main():
    with open("data_pokorny/pokorny_scraped.json", "r", encoding="utf-8") as fp:
        pokorny = json.load(fp)

    # for each entry in the raw pokorny, we need to build out entries that are well laid out
    processed_pokorny = [
        {
            "root": [root for roots in entry["root"] for root in roots.split(",")],
            "meaning": [meaning.strip() for meanings in entry["English meaning"] for meaning in meanings.split(",")],
            "other_meaning": {"german": [meaning.strip() for meanings in entry["German meaning"] for meaning in meanings.replace("`", "'").split(",")]},
        }
        for entry in pokorny
    ]

    # we prompt chatgpt here with:
    # 'I am going to give you a numbered set of word definitions. Tell me the part of speech that the word is, based solely on the definition. If there are multiple that you think are reasonable, give all that are applicable. Give your answer in the form "number: part of speech". Do not include any other text, do not include the meaning either.'
    # then pasting in what clipboard gives you and putting the chat output in pokorny_pos.txt
    # clipboard(processed_pokorny)

    # the output of gpt is not deterministic afaik, so processing the output may require changing the processing func.
    processed_pokorny = process_pos(processed_pokorny, "data_pokorny/gpt_pos.txt")

    # output to file for double-checking by human
    output_pos(processed_pokorny)

    # we also have a human double-checking and correcting those tags, which takes precedence
    # processed_pokorny = process_pos(processed_pokorny, "data_pokorny/human_pos.txt")

    # generate IDs for each. There are just human-readable ways to refer to the word that can be put into a url

    # output the processed to json, so we can import into a db or something
    with open("data_pokorny/pokorny_processed.json", "w", encoding="utf-8") as fp:
        json.dump(processed_pokorny, fp)
    # breakpoint()
    pass


if __name__ == '__main__':
    main()
    pass
