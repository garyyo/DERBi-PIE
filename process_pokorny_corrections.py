import glob
import json
import os
from copy import copy

import pandas as pd
from openpyxl import load_workbook
from openpyxl.cell.rich_text import CellRichText, TextBlock

import gpt_functions
from find_missing_pokorny_gpt import get_web_lrc_lineup_corrected, get_web_from_lrc, augment_material, format_gpt_prompt_new2
from generate_pokorny_db_data import load_abbreviations_df, remove_html_tags_from_text

corrections_needed = {
    "Additional Pokorny Forms 0.xlsx": ["", ""],
    "Additional Pokorny Forms 1.xlsx": ["", ""],
    "Additional Pokorny Forms 2.xlsx": ["dheu-4, dheu̯ə- (vermutlich: dhu̯ē-, vgl. die Erw. dhu̯ē-k-, dhu̯ē̆-s-)", ""],
}


def mutate_font(font, **kwargs):
    # some common longhand
    translation_dict = {
        "bold": "b",
        "underline": "u",
        "italic": "i",
    }
    new_font = copy(font)
    for key, value in kwargs.items():
        new_font.__dict__[translation_dict.get(key, key)] = value
    return new_font


# todo: make sure I use forward slashes as I have assumed in the code here.
def un_italicize_cell(cell):
    # if the entire cell is italicized then we only put italics around the whole thing.
    if cell.font.italic:
        # weird roundabout way to set the italics to false but apparently its required
        cell.font = mutate_font(cell.font, italic=False)

        if isinstance(cell.value, CellRichText):
            for block_count, text_block in enumerate(cell.value):
                # make sure it is de-italicized
                if isinstance(text_block, TextBlock):
                    if text_block.font.italic:
                        text_block.font = mutate_font(text_block.font, italic=False)
                # add in my italics marker "/" to the start and end
                if block_count == 0:
                    if isinstance(text_block, TextBlock):
                        text_block.text = f"/{text_block.text}"
                    else:
                        cell.value[block_count] = f"/{text_block}"
                if block_count == (len(cell.value) - 1):
                    if isinstance(text_block, TextBlock):
                        text_block.text = f"{text_block.text}/"
                    else:
                        cell.value[block_count] = f"{text_block}/"
        elif isinstance(cell.value, str):
            cell.value = f"/{cell.value}/"
        elif cell.value is not None:
            breakpoint()
    # otherwise we do it a bunch of times.
    elif isinstance(cell.value, CellRichText):
        for text_block in cell.value:
            # ensure it's a text block, and not a plain string, and that it is italicized
            if isinstance(text_block, TextBlock) and text_block.font.italic:
                # mark it and remove the italics
                text_block.font = mutate_font(text_block.font, italic=False)
                text_block.text = f"/{text_block.text}/"


def fix_italics(filepath='data_pokorny/additional_pokorny_corrections/Additional Pokorny Forms 0.xlsx'):
    # Load the workbook
    workbook = load_workbook(filepath, rich_text=True)

    # Assume you're working with the first sheet, change as needed
    sheet = workbook.active

    # Iterate through all cells in the sheet
    for row in sheet.iter_rows():
        for cell in row:
            un_italicize_cell(cell)

    # Close the workbook when done
    path, ext = os.path.splitext(filepath)
    # workbook.save(filepath)
    new_filepath = f"{path}_italic_fix{ext}"
    workbook.save(new_filepath)
    workbook.close()
    return new_filepath


def extract_bad_entries(filepath):
    workbook = load_workbook(filepath, rich_text=True)
    sheet = workbook.active
    rerun_entry_ids = set()
    delete_entry_ids = set()
    normal_colors = ["00000000", "FFFFFFFF"]
    error_color = "FFFF0000"
    redo_color = "FFFF9900"

    for row in sheet.iter_rows():
        for cell in row:
            if cell.fill.start_color.rgb not in normal_colors + [error_color, redo_color]:
                # if we find a color that has never been seen
                print(f"New cell color '{cell.fill.start_color.rgb}' in cell {cell}, with value '{cell.value}'")
                breakpoint()
            if cell.fill.start_color.rgb == error_color:
                # here are entries that we just need to mark for deletion
                delete_entry_ids.add((row[1].value, row[2].value))
                break
            if cell.fill.start_color.rgb == redo_color:
                # this row needs to be skipped and marked as missing
                rerun_entry_ids.add((row[1].value, row[2].value))
                break
    if (None, None) in rerun_entry_ids:
        rerun_entry_ids.remove((None, None))
    return rerun_entry_ids, delete_entry_ids


def redo_pokorny(rerun_entry_ids):
    # we have the entries that we need to do, but we need all the other information about them
    dfs = {os.path.splitext(os.path.basename(df_file))[0]: pd.read_pickle(df_file) for df_file in glob.glob("data_pokorny/table_dumps/*.df")}
    abbreviation_data = load_abbreviations_df()
    line_up_df = get_web_lrc_lineup_corrected()

    redo_entries = dfs['lex_etyma'][dfs['lex_etyma'].entry.isin([id[1] for id in rerun_entry_ids])]

    headers = ['abbr', 'reflex', 'meaning', 'notes']

    with open("data_pokorny/pokorny_scraped_old.json", "r", encoding="utf-8") as fp:
        scraped_pokorny = json.load(fp)

    # turn scraped pokorny into a dictionary
    scraped_pokorny_dict = {
        ", ".join(entry["root"]): entry
        for entry in scraped_pokorny
    }

    gpt4_char_threshold = 4500
    entries_to_be_processed = list(enumerate(redo_entries.iterrows()))

    header_color = "\033[92m"
    end_color = "\033[0m"

    for counter, (i, row) in entries_to_be_processed:
        texas_root = remove_html_tags_from_text(row.entry).strip("\n")

        web_key = get_web_from_lrc(line_up_df, texas_root)
        web_entry = scraped_pokorny_dict.get(web_key, None)

        web_root = web_entry["root"]
        gloss = remove_html_tags_from_text(row.gloss)

        material = "\n".join(web_entry['Material']).replace("`", "'")
        abbreviations_used = augment_material(material, abbreviation_data)

        prompt = format_gpt_prompt_new2(texas_root, gloss, material, abbreviations_used, headers)

        model = "gpt-4" if len(prompt) > gpt4_char_threshold else "gpt-3.5-turbo"

        # check if it's valid before trying to grab the cached response
        if len(prompt) > 15000 or material.strip() == "":
            print(f"{header_color}---=== Never Run for {web_root} ===---{end_color}")
        else:
            messages, _ = gpt_functions.get_cached([prompt], model=model)

            if messages is not None:
                print(f"{header_color}---=== Found Cached for {web_root} ===---{end_color}")
                print(messages[-1]["content"])
                print(f"for {web_root}")
            else:
                print(f"{header_color}---=== Cached Missing for {web_root}! ===---{end_color}")
        breakpoint()
    pass


def main():
    filepath = 'data_pokorny/additional_pokorny_corrections/Additional Pokorny Forms 0.xlsx'
    # filepath = fix_italics(filepath)
    # rerun_entry_ids, delete_entry_ids = extract_bad_entries(filepath)
    rerun_entry_ids = {('bhares- : bhores-', 'bhares-, bhores-'), ('ant-s', 'ant-s'), ('aĝ-', 'ag̑-'), ('aleq-', 'aleq-'), ('ā̆p-2', '2. ā̆p-'), ('ar(ə)-', 'ar(ə)-'), ('au̯(e)-9, au̯ed-, au̯er-', '9. au̯(e)-, au̯ed-, au̯er-'), ('ā̆l-3', '3. ā̆l-'), ('ā̆s-, davon azd-, azg(h)-', 'ā̆s-, based on it azd-, azg(h)-'), ('bhā̆u-1 : bhū̆-', '1. bhā̆u- : bhū̆-'), ('bhedh-2', '2. bhedh-'), ('au̯(e)-10, au̯ē(o)-, u̯ē-', '10. au̯(e)-, au̯ē(i)-, u̯ē-'), ('al-5', '5. al-'), ('bheid-', 'bheid-'), ('bhardhā', 'bhardhā'), ('bhel-3, bhlē-', '3. bhel-, bhlē-'), ('b(e)u-2, bh(e)ū̆-', '2. b(e)u-, bh(e)ū̆-'), ('al-1, ol-', '1. al-, ol-'), ('ak̂-, ok̂-', '2. ak̑-, ok̑-'), ('aig-3', '3. aig-'), ('ar-1^^*^^, themat. (a)re-, schwere Basis arə-, rē- und i-Basis (a)rī̆-, rēi-', '1. ar-, thematic (a)re-, heavy-base arə-, rē-, and i-base (a)rī̆-, rēi-')}
    # print out what needs to be rerun?
    redo_pokorny(rerun_entry_ids)
    breakpoint()
    pass


if __name__ == '__main__':
    main()
