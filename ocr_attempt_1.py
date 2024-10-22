"""
author - anton vinogradov

DERBi PIE. Not sure what it means, but it's a good name.

The purpose of this project is to scan in and digitize PIE dictionaries.
These can be in several languages, but we start with a german one.

The final site needs a couple of features:
- search
    - search for roots
    - search by words (a word is generated by the roots?) (partial match?)
    - search by meaning
    - search by word class
- view entries laid out in a nice way
    - ????
- likely will be written with nodejs
- something else?

Before that we need to:
- OCR scan PIE dictionaries
    - the OCR stuff likely will need to be post processed somehow, preferably automatically
- come up with a suitable schema for organizing this data (may ask mom for help)
    - we expect the data to be large, allegedly GB of text
- come up with a set of queries for the DB (this can be changed later, but we should think up a couple starter ones in the meantime)
-

Concepts:
- front end: This is the portion of the code that is rendered on the user's computer. mostly about looks
- back end: This is the portion of the code that processes user's requests, assembles the code that is sent over to the user, and holds database stuff.
    - users cant directly query the DB, that would be unsafe obviously
- tesseract: Open Source OCR software. will likely need to be customized or something for the sake of our purposes.
- DERBi PIE: this entire project, but can also be used to refer to the website.
"""
import glob
import json
from collections import defaultdict
from pprint import pprint
import pytesseract
import matplotlib.pyplot as plt
import pypdf

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
import dacite
from pdf2image import convert_from_path

from ocr_page_classes import *

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# print(pytesseract.get_languages())
poppler_path = "poppler/Library/bin"

size_multiplier = 3
text_color = (0, 0, 0)
regular_font = ImageFont.truetype("fonts/noto_serif/NotoSerif-Regular.ttf", 9 * size_multiplier)
small_font = ImageFont.truetype("fonts/noto_serif/NotoSerif-Regular.ttf", 7 * size_multiplier)
title_font = ImageFont.truetype("fonts/noto_serif/NotoSerif-Regular.ttf", 11 * size_multiplier)


def combine_json():
    ocr_full = []
    for file in glob.glob("data_liv/data/*"):
        with open(file, "r", encoding="utf8") as fp:
            ocr_full += json.load(fp)["responses"]

    # sort by page number since glob reads them out of order
    ocr_full = sorted(ocr_full, key=lambda x: x['context']["pageNumber"])

    # save to disk to avoid doing that again
    output_file = "data_liv/LIV_ocr.npy"
    np.save(output_file, {"ocr_full": ocr_full}, allow_pickle=True)

    return ocr_full, output_file


def load_combined_json(file="indodoc/LIV_ocr.npy"):
    ocr_full = np.load(file, allow_pickle=True).item()["ocr_full"]
    return ocr_full, file


def flatten_pages(ocr_full, page_start=70, page_end=707, page_limit=-1):
    use_pages = list(range(page_start, page_end, 1))
    use_pages = use_pages[0:page_limit]
    # go from page to page attempting to find the "header" of a new entry. then process that entry
    pages = [page for page in ocr_full if page["context"]["pageNumber"] in use_pages]
    # flatten the structure a little bit
    pages = [
        {
            "page_num": page["context"]["pageNumber"],
            "languages": page["fullTextAnnotation"]["pages"][0]["property"]["detectedLanguages"],
            "width": page["fullTextAnnotation"]["pages"][0]["width"],
            "height": page["fullTextAnnotation"]["pages"][0]["height"],
            "confidence": page["fullTextAnnotation"]["pages"][0]["confidence"],
            "blocks": page["fullTextAnnotation"]["pages"][0]["blocks"]
        }
        for page in pages
    ]
    return pages


def get_text_pos(page, word):
    x, y = word.boundingBox[0]
    return page.width * x * size_multiplier, page.height * y * size_multiplier


def get_font(block):
    # this is a bad way to do it, but it's a start
    # proper way is to just classify all the text
    first_word = "".join([symbol.text for symbol in block.paragraphs[0].words[0].symbols])

    if first_word == "Lemmata":
        font = title_font
    elif first_word in [str(i) for i in range(15)]:
        font = small_font
    else:
        font = regular_font

    return font


def get_text_color(word):
    return get_text_color_conf(word.confidence)


def get_text_color_conf(confidence):
    # squared to make the errors stand out more
    return 0, int((1 - confidence ** 2) * 255), 0


def draw_ocr_tess():
    tess_size_multiplier = 4
    tess_font = ImageFont.truetype("fonts/noto_serif/NotoSerif-Regular.ttf", 9 * size_multiplier * tess_size_multiplier)
    # take what we want from LIV and convert it into images
    pages = convert_from_path("data_liv/LIV2_subset.pdf", poppler_path=poppler_path, dpi=200 * tess_size_multiplier)

    for page_num, page in enumerate(pages):
        # OCR the page
        df = pytesseract.image_to_data(page, output_type="data.frame", lang="deu")

        # start drawing the image
        image = Image.new('RGB', (page.width, page.height), color='white')
        draw = ImageDraw.Draw(image)

        for i, row in df.iterrows():
            # skip all the entries that don't have text, I am not currently sure what they are.
            if type(row.text) != str:
                continue
            draw.text((row.left, row.top), row.text, fill=get_text_color_conf(row.conf), font=tess_font)

        image.save(f"output_pages/recreated_{page_num}.png")
        break


def draw_ocr_json():
    images = convert_from_path("data_liv/LIV2_subset.pdf", poppler_path=poppler_path, dpi=400)

    # ocr_full, output_file = combine_json()
    ocr_full, output_file = load_combined_json()

    # we need to manually figure out which pages are the ones we need and which are the ones that can be ignored
    # to start I will make some assumptions
    pages = flatten_pages(ocr_full, page_limit=20)

    # this is used to doublecheck the data is as I think it is.
    # pprint(extract_structure(pages))

    # turn it into dataclasses (mainly for validation, but also its more convenient)
    data_pages = [dacite.from_dict(data=page, data_class=OCRPage) for page in pages]
    offset_to_word = defaultdict(set)
    for page in data_pages:
        image = Image.new('RGB', (page.width * size_multiplier, page.height * size_multiplier), color='white')
        draw = ImageDraw.Draw(image)
        for block in page.blocks:
            # for each block: try to detect what that block is first. then we can set the font size and later classify it for db purposes
            font = get_font(block)

            # todo: experimental bs area
            # first_word = "".join([symbol.text for symbol in block.paragraphs[0].words[0].symbols])
            # offset_to_word[block.boundingBox[0].x].add(first_word)

            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    text = word.get_text()

                    # if any word gets a really low confidence, maybe try something different
                    if word.confidence < 0.3:
                        #
                        image = images[page.page_num - 70]

                        (x1, y1), (x2, y2) = paragraph.boundingBox[0], paragraph.boundingBox[2]
                        crop = image.crop((x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height))
                        plt.imshow(crop)
                        plt.show()
                        breakpoint()
                    # draw things here
                    draw.text(get_text_pos(page, word), text, fill=get_text_color_conf(word.confidence), font=font)
                # breakpoint()

        image.save(f"output_pages/recreated_{page.page_num}.png")


def main():
    # reader = pypdf.PdfReader("indodoc/LIV2_subset.pdf")
    # breakpoint()
    # draw_ocr_json()
    draw_ocr_tess()
    pass


if __name__ == '__main__':
    main()
    pass
