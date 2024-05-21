import glob
import json
import os.path
import shutil

import pandas as pd
import regex as re

from cltk.data.fetch import FetchCorpus


def extract_text(data):
    result = []
    for key in sorted(data, key=int):  # Ensure the keys are processed in numerical order
        if isinstance(data[key], dict):
            # If the value is a dictionary, recurse into it
            result.extend(extract_text(data[key]))
        elif isinstance(data[key], str):
            # If the value is a string, add it to the result list
            result.append(data[key])
    return result


def process_corpus_perseus():
    # 1. download a big set of latin texts
    corpus_downloader = FetchCorpus(language="lat")
    corpus_downloader.import_corpus("lat_text_perseus")

    # 2. turn those into data we can use
    # figure out where the data is stored
    if os.getenv("CLTK_DATA"):
        cltk_data_dir = os.getenv("CLTK_DATA")
    else:
        cltk_data_dir = os.path.expanduser("~/cltk_data/lat/text/lat_text_perseus/cltk_json")

    # luckily it provides it in nice json format
    latin_jsons = glob.glob(f"{cltk_data_dir}/*latin.json")

    # loop through it and extract all of it
    latin_texts = []
    for json_path in latin_jsons:
        with open(json_path, "r", encoding="utf-8") as fp:
            latin_obj = json.load(fp)

        # skip if we pick up a non-latin text
        if latin_obj["language"] != "latin":
            breakpoint()
            continue

        # extract all the text out
        text_list = extract_text(latin_obj["text"])

        # clean the text just a bit (remove leading and following spaces and newlines)
        latin_text = "\n".join(text.strip("\n ") for text in text_list)
        latin_texts.append(latin_text)

    return latin_texts


def process_corpus_tesserae():
    # 1. download a big set of latin texts
    corpus_downloader = FetchCorpus(language="lat")
    corpus_downloader.import_corpus("lat_text_tesserae")

    # 2. turn those into data we can use
    # figure out where the data is stored
    if os.getenv("CLTK_DATA"):
        cltk_data_dir = os.getenv("CLTK_DATA")
    else:
        cltk_data_dir = os.path.expanduser("~/cltk_data/lat/text/lat_text_tesserae/texts")

    # the .tess file format seems to be just text
    latin_tess_files = glob.glob(f"{cltk_data_dir}/*.tess")

    latin_texts = []
    for tess_file in latin_tess_files:
        with open(tess_file, 'r', encoding='utf-8') as fp:
            latin_texts.append("\n".join(
                re.sub(r'^<[^>]+>', '', line).strip("\n ")
                for line in fp.readlines()
            ))

    return latin_texts


def download_corpus():
    # since there is significant overlap between these, I will take the biggest one
    # contains ~2.9e7 characters
    # latin_perseus_texts = process_corpus_perseus()
    # contains ~4.6e7 characters
    latin_tesserae_texts = process_corpus_tesserae()

    with open("prealigned/latin_corpus3.txt", "w", encoding="utf-8") as fp:
        fp.writelines(latin_tesserae_texts)

    # split them up into smaller files, might help the lemmatization process
    # clear out the directory beforehand
    directory = "prealigned/latin_corpus_paragraphs3"
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    # save it
    pad_amount = len(str(len(latin_tesserae_texts)))
    for i, split in enumerate(latin_tesserae_texts):
        with open(f"{directory}/{i:0{pad_amount}}.txt", "w", encoding="utf-8") as fp:
            fp.write(split.strip())
    pass


def main():
    # df_old = pd.read_csv("prealigned/lemmatized_bak.csv")
    # df_new = pd.read_csv("prealigned/latin_corpus2.csv", encoding="ANSI")

    download_corpus()
    breakpoint()


if __name__ == '__main__':
    main()
    pass
