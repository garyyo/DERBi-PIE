import json
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
import pyperclip
from tqdm import tqdm

from docx import Document
from docx.document import Document as DocType
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph
import regex as re
import os
import pickle
import copy

import gpt_functions

# collecting stats
num_weird_headings = 0
num_bad_descendants = 0
num_gpt_skipped = 0
num_parts = {
    "box1": 0,
    "box2": 0,
    "box3": 0,
    "box4": 0,
    "box5": 0,
    "box6": 0,
    "box7": 0,
}


# stuff
font_copyable = ['all_caps', 'bold', 'complex_script', 'cs_bold', 'cs_italic', 'double_strike', 'emboss', 'hidden', 'highlight_color', 'imprint', 'italic', 'math', 'name', 'no_proof', 'outline', 'rtl', 'shadow', 'size', 'small_caps', 'snap_to_grid', 'spec_vanish', 'strike', 'subscript', 'superscript', 'underline', 'web_hidden']


def run_or_load(filename, func, rerun=False, **kwargs):
    if os.path.exists(filename) and not rerun:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        result = func(**kwargs)
        with open(filename, 'wb') as f:
            pickle.dump(result, f)
        return result


@dataclass
class RegexEqual(str):
    string: str
    match: re.Match = None

    def __eq__(self, pattern):
        self.match = re.search(pattern, self.string)
        return self.match is not None


def split_at_positions(s, positions):
    splits = []
    start = 0
    for pos in positions:
        splits.append(s[start:pos])
        start = pos + 1
    splits.append(s[start:])
    return splits


# todo: arguably this should be done with regex or something for speed
def detect_unquoted_semicolons(s):
    positions = []
    in_single_quote = False
    in_paren = 0
    for i, char in enumerate(s):
        if char == "'" and (i == 0 or s[i-1] != '\\'):
            in_single_quote = not in_single_quote
        elif char == '(':
            in_paren += 1
        elif char == ')':
            if in_paren > 0:
                in_paren -= 1
        elif char == ';' and not in_single_quote and in_paren == 0:
            positions.append(i)
    return positions


def iter_block_with_type(parent, type_override="paragraph"):
    # https://github.com/python-openxml/python-docx/issues/40
    """
    Yield each paragraph and table child within *parent*, in document order.
    Each returned value is an instance of either Table or Paragraph. *parent*
    would most commonly be a reference to a main Document object, but
    also works for a _Cell object, which itself can contain paragraphs and tables.
    """
    if isinstance(parent, DocType):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError("something's not right")

    # print('parent_elm: '+str(type(parent_elm)))
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent), type_override
        elif isinstance(child, CT_Tbl):
            # yield Table(child, parent)  # No recursion, return tables as tables
            table = Table(child, parent)  # Use recursion to return tables as paragraphs
            for row in table.rows:
                for cell in row.cells:
                    yield from iter_block_with_type(cell, "cell")


def match_heading(item):
    # split into root and gloss
    groups = re.match(r"^(\?)? ?((?:\d\. )?\*.+)[\'‘](.+)\'(\^\^\d{0,2}\^\^)?$", superscript_paragraph(item).strip()).groups()
    if len(groups) != 4:
        breakpoint()
    questionable, root, gloss, cite_number = groups
    questionable = questionable is not None
    return questionable, root, gloss, cite_number


def match_heading_sources(item):
    # valid sources: 'EIEC', 'IEW', 'LIPP', 'LIV'
    valid_sources = ['EIEC', 'IEW', 'LIPP', 'LIV']
    # 'vgl.' is a flag that means to 'compare'. note, I do not know what it actually means
    # assert that a valid source exists
    assert any([source in item.text for source in valid_sources])
    sources = item.text.split(',')
    return [source.strip() for source in sources]


def superscript_markings(run):
    if run.font.superscript:
        return f"^^{run.text}^^"
    return run.text


def superscript_paragraph(item):
    return "".join([superscript_markings(run) for run in item.runs])


def markup_paragraph(item):
    # super_scripted_text = "".join([f"{'^^' if run.font.superscript else ''}{run.text}{'^^' if run.font.superscript else ''}" for run in item.runs])
    modified_run_texts = []

    if "\t" in item.text:
        tab_found = False
    else:
        tab_found = True

    for run in item.runs:
        # if there was a tab, but we haven't found it yet: then just apply the superscript markings
        if "\t" not in run.text and not tab_found:
            modified_run_texts.append(superscript_markings(run))
        elif "\t" in run.text:
            tab_found = True
            modified_run_texts.append(run.text)
        elif run.italic or run.font.name == "Greek":
            modified_run_texts.append(f"//{superscript_markings(run)}//")
        else:
            modified_run_texts.append(superscript_markings(run))

    # print("".join(modified_run_texts))
    def fix_multi_italic(match_obj):
        return "//" + match_obj.group(0).replace("//", "") + "//"
    fixed_text = re.sub("\/\/\S+\/\/", fix_multi_italic, "".join(modified_run_texts))
    # print(fixed_text)
    return fixed_text


def stem_exceptions(item, super_scripted_text):
    if "? *(dʰ)ĝʰ-(m̥)m-e/on-14" in item.text:
        return super_scripted_text[:32], super_scripted_text[32:]
    if "*(dʰ)ĝʰ-m-(i)i̯o/ah2- aksl." in item.text:
        return super_scripted_text[:26], super_scripted_text[26:]
    return None, None


def match_stem(item):
    # try replacing 4 spaces with a tab.
    super_scripted_text = markup_paragraph(item).rstrip().replace("    ", "\t", 1)
    # collapse multiple sequential tabs into a single tab.
    super_scripted_text = re.sub(r"\t+", "\t", super_scripted_text)

    if super_scripted_text.count("\t") == 1:
        stem_text, info_text = super_scripted_text.split("\t")
    elif super_scripted_text.count("\t") == 0:
        # there are some super rare exceptions that I am manually handling.
        stem_text, info_text = stem_exceptions(item, super_scripted_text)
        if stem_text is None:
            stem_text = super_scripted_text
            info_text = None
    else:
        # if there are more tabs than what makes sense then only split on the first one. downstream stuff will safely fail if that's wrong.
        stem_text, info_text = super_scripted_text.split("\t", 1)
        pass

    try:
        stem_groups = re.match(r"^(\? ?)?((?:\*|\^\^x\^\^)(?:[^\s^]|\^\^é\^\^)+)(\^\^\d{0,2}\^\^)?( \(?[mfnc]\.\)?)?( ASg\.)?(\^\^\d{0,2}\^\^)?$", stem_text.strip()).groups()
        questionable_stem, stem, stem_cite, asg, gender, gender_cite = stem_groups
        questionable_stem = questionable_stem is not None
        stem_cite = stem_cite.replace("^", "") if stem_cite is not None else None
        gender_cite = gender_cite.replace("^", "") if gender_cite is not None else None
    except AttributeError:
        # if it errors there is not much I can do, just silently move on.
        global num_bad_descendants
        num_bad_descendants += 1

        stem = None
        questionable_stem = None
        stem_cite = None
        asg = None
        gender = None
        gender_cite = None

    descendant = {
        "stem": {
            "stem": stem,
            "questionable": questionable_stem,
            "cite": stem_cite,
            "gender": gender,
            "gender_cite": gender_cite,
            "asg": asg
        },
        "reflexes": []
    }

    # Only add a new reflex if there is one
    # if info_text is not None:
    #     descendant["reflex"].append(match_reflex(info_text))
    return descendant


def match_stem_analogy(item):
    descendant = {
        "stem": {
            "stem": item.text,
            "arrow_splits": item.text.split("→"),
            "questionable": False,
            "cite": None,
            "gender": None,
            "gender_cite": None,
            "is_sub_entry": False
        },
        "reflexes": []
    }
    return descendant


def match_reflex_paragraph(item):
    info_text = markup_paragraph(item).lstrip()
    reflex = match_reflex(info_text)
    # breakpoint()
    return reflex


def match_reflex(info_text):
    # check for the presence of a `;`
    semicolon_pos = detect_unquoted_semicolons(info_text)

    reflexes = []
    for info_text_split in split_at_positions(info_text, semicolon_pos):
        (
            questionable_descendant,
            lang_abbr,
            other_abbr,
            derivative,
            derivative_gender,
            derivative_gloss,
            other_text,
            first_attested,
            cite_number
        ) = match_reflex_sub(info_text_split)

        # if at any point language abbreviation info is missing we inherit from the last seen
        if lang_abbr is None and len(reflexes):
            lang_abbr = reflexes[-1]["language"]["language_abbr"]

        reflex = {
            "language": {
                "language_abbr": lang_abbr,
            },
            "reflexes": [derivative.replace("/", "")],
            "gloss": derivative_gloss,
            "gender": derivative_gender,
            "first_attested": first_attested,
            "questionable": questionable_descendant,
            "cite": cite_number,
            "unknown_text": other_text,
            "other_abbr": other_abbr,
        }
        print(reflex)
        reflexes.append(reflex)
    if len(semicolon_pos):
        breakpoint()
    return reflexes


def match_reflex_sub(info_text):
    # this one works a bit differently in that it needs the text, since it can be called either on a part of a paragraph, or a whole paragraph.
    info_groups = re.match(r"^(\? ?)?(\S+\.)?( \S+\.)? (-?\/\/[^']+?\/\/ )([mfnc]\. )?('.+' ?)?(.+?)(\(.+\))?(\^\^\d+\^\^)?$", info_text).groups()
    questionable_descendant, lang_abbr, other_abbreviations, derivative, derivative_gender, derivative_gloss, other_text, first_attested, cite_number = info_groups

    # process these a bit more
    questionable_descendant = questionable_descendant is not None
    derivative = derivative.strip()
    derivative_gloss = derivative_gloss.strip("'") if derivative_gloss is not None else derivative_gloss
    # todo: match lang abbreviation to a language
    # todo: derivative gloss may actually fail if there are multiple and may need to be subdivided
    cite_number = cite_number.replace("^", "") if cite_number is not None else None

    return questionable_descendant, lang_abbr, other_abbreviations, derivative, derivative_gender, derivative_gloss, other_text, first_attested, cite_number


def match_reflex_sub_alt(info_text):
    breakpoint()
    return


def match_nil_parts1(document):
    # try to just iterate through all the stuff
    cell_count = 0
    other_state = False
    dict_entries = []
    current_entry = None

    tab_header_count = 0
    reflex_and_other_count = 0
    for item, item_type in iter_block_with_type(document):
        print(f"---{item_type}---")
        print(item.text)
        print(item.paragraph_format.element.style)
        print(item.paragraph_format.first_line_indent)
        style = item.paragraph_format.element.style
        #
        first_line_indent = item.paragraph_format.first_line_indent is None
        # first_line_indent = item.paragraph_format.first_line_indent not in [0, -1905, -2540, -2540, -635, 449580, 358140, -447675, 3175]
        #
        # matches all the possible cases (and breakpoints when it doesn't have the correct case
        match RegexEqual(item.text), item_type, cell_count, style, other_state, first_line_indent:
            case r"^\s*$", _, _, _, _, _:
                print("skipping...")
                # empty line, anton: might mean a new entry but currently there are other ways to find that.
                other_state = False
                cell_count = 0
                continue
            # anton: ------------ THE "cell" BLOCK ------------
            case (_, "cell", 0, "Heading6", _, _) | (_, _, 0, "Heading6", _, _):
                print("Cell 1")
                # the first part of the new entry
                # create a new entry, and set it as the current one
                dict_entries.append({
                    "root": None,
                    "root_cite": None,
                    "questionable": False,
                    "gloss": None,
                    "sources": None,
                    "descendants": [],
                    # anton: below are the deprecated methods of saving this info
                    #  IMPORTANT: they should not be used later on and should eventually be deleted.
                    "cell_info": [],
                    "descendant_info": [],
                    "other": [],
                    "footnotes": [],
                })
                current_entry = dict_entries[-1]

                # if it's that weird tab separated one we are going to do something weird
                if "\t" in item.text:
                    tab_header_count += 1
                else:
                    # the actual info
                    current_entry["questionable"], current_entry["root"], current_entry["gloss"], current_entry["root_cite"] = match_heading(item)
                current_entry["cell_info"].append(item.text)
                #
                cell_count += 1
                # breakpoint()
            case _, "cell", 1, _, _, _:
                print("Cell 2")
                # the second part of the new entry, citations to other texts
                current_entry["cell_info"].append(item.text)
                current_entry["sources"] = match_heading_sources(item)
                cell_count += 1
                # breakpoint()
            case _, "cell", x, _, _, _ if x >= 2:
                print("Cell ERROR")
                # this should not happen
                breakpoint()
            # anton: ------------ THE "other" BLOCK ------------
            case "^\?? ?[\‡\*]", "paragraph", _, _, True, _:
                print("Other 1")
                # the start of a new other portion
                cell_count = 0
                current_entry["other"].append(item.text)
                reflex_and_other_count += 1
                # breakpoint()
            case ("^\t", "paragraph", _, "LIN1", True, _) \
                 | (_, "paragraph", _, ("LIN2" | "LIN3"), True, _) \
                 | (_, "paragraph", _, "LIN1", True, False):
                print("Other 2")
                # a continuation of the previous other portion
                cell_count = 0
                current_entry["other"][-1] = current_entry["other"][-1].rstrip() + "\n" + item.text.lstrip()
                reflex_and_other_count += 1
                # breakpoint()
            case "^Sonstige", "paragraph", _, _, False, _:
                print("Other 0")
                # start of an "other" section.
                other_state = True
                cell_count = 0
                # breakpoint()
            # anton: ------------ THE "descendant" BLOCK ------------
            case "^(?:\d\.)? ?\[?[\?\*x]", "paragraph", _, ("LIN1" | "LIN3" | "BodyText"), False, _:
                print("Descendant 1")
                if "→" in item.text:
                    breakpoint()
                if not first_line_indent:
                    breakpoint()
                    continue
                # the start of a new descendants portion (the ? and the * may not be exhaustive, but I will find a better way to do it later)
                cell_count = 0
                descendant = match_stem(item)
                current_entry["descendants"].append(descendant)
                current_entry["descendant_info"].append(item.text)
                reflex_and_other_count += 1
                # breakpoint()
            case ("^\t", "paragraph", _, ("LIN1" | "LIN3" | "BodyText"), False, _) \
                 | (_, "paragraph", _, "LIN2", False, _) \
                 | (_, "paragraph", _, ("LIN1" | "LIN3" | "BodyText"), False, False):
                print("Descendant 2")
                # a continuation of the previous descendants portion
                # match_reflex_paragraph(item)
                cell_count = 0
                current_entry["descendant_info"][-1] = current_entry["descendant_info"][-1].rstrip() + "\n" + item.text.lstrip()
                reflex_and_other_count += 1
                # breakpoint()
            # anton: ------------ THE "footnotes" BLOCK ------------
            case _, "paragraph", _, ("Literatur2" | "Literatur 2"), _, _:
                print("Footnotes 1")
                cell_count = 0
                # the footnotes section means that the other block is over
                other_state = False
                current_entry["footnotes"].append(item.text)
                # breakpoint()
            # anton: ------------ THE something went wrong BLOCK ------------
            case _, "paragraph", _, _, _, _:
                # if it's a paragraph but hasn't been caught by any other block
                cell_count = 0
                breakpoint()
            case _:
                # if it hasn't been caught by any other block
                breakpoint()


def make_new_entry():
    return {
        "root": None,
        "root_cite": None,
        "questionable": False,
        "gloss": None,
        "sources": None,
        "descendants": [],
        # anton: below are the deprecated methods of saving this info
        #  IMPORTANT: they should not be used later on and should eventually be deleted.
        "cell_info": [],
        "descendant_info": [],
        "other": [],
        "footnotes": [],
    }


def process_cell(item, cell_state, current_entry, all_entries):
    global num_parts
    # make new entry if we see a new cell
    if cell_state == 1:
        num_parts["box1"] += 1
        new_entry = make_new_entry()

        # match the heading as best we can
        new_entry["questionable"], new_entry["root"], new_entry["gloss"], new_entry["root_cite"] = match_heading(item)

        # a safety precaution to make sure we are not missing anything
        new_entry["cell_info"].append(item.text)

        all_entries.append(new_entry)
        current_entry = all_entries[-1]
        pass
    if cell_state == 2:
        num_parts["box2"] += 1
        current_entry["cell_info"].append(item.text)
        current_entry["sources"] = match_heading_sources(item)
        pass
    return current_entry


def amend_runs(item, delimiter, collapse_sequential=True):
    runs_list = [[]]
    # this just sorta staggers them so from [0,1,2,3] we get [(1,0), (2,1), (3,2), (None,3)]
    for next_run, run in zip(item.runs[1:] + [None], item.runs):
        if delimiter in run.text:
            # if we see a delimiter in the next one, and the current one is just a delimiter, then we skip (this condition does not run)
            if collapse_sequential and next_run is not None and run.text.strip() == delimiter.strip() and delimiter in next_run.text:
                pass
            else:
                runs_list.append([])
        runs_list[-1].append(run)
    return runs_list


def copy_item(item, new_runs=()):
    item_copy = copy.deepcopy(item)
    # reset the text to remove the runs
    item_copy.text = ""
    if len(new_runs) == 0:
        new_runs = item.runs
    for run in new_runs:
        # make the new run
        item_copy.add_run(run.text, run.style)
        # copy over all the copyable attributes
        for font_prop in font_copyable:
            setattr(item_copy.runs[-1].font, font_prop, getattr(run.font, font_prop))
    # process it
    return item_copy


def process_weird_heading(item, current_entry, all_entries):
    # if this contains a tab, it's probably a new entry but was weirdly placed into a heading6 instead of a table
    new_entry = None
    try:
        if "\t" in item.text:
            # collapse multiple subsequent tabs into a single, then split both the text and the runs.
            cells = re.sub(r"\t+", "\t", item.text).split("\t")
            # we only handle cases where there are exactly 2 parts to this, error on anything else
            assert len(cells) == 2
            split_runs = amend_runs(item, "\t")

            # for each cell we run it through the standard process
            new_entry = None
            for i, (cell, runs) in enumerate(zip(cells, split_runs)):
                # make the proper copies of each sub item (there is no easy way to do this)
                item_copy = copy_item(item, runs)
                new_entry = process_cell(item_copy, i+1, new_entry, all_entries)
            # return it nothing errored, otherwise it jump to the except clause
            return new_entry
    except Exception as err:
        print(err)
        if new_entry == all_entries[-1]:
            all_entries.pop()
    return current_entry


def process_descendant_initial(item, first_indent, current_entry):
    global num_parts
    num_parts["box3"] += 1
    num_parts["box4"] += 1
    # anton: why is there sometimes an arrow?
    if "→" in item.text:
        descendant = match_stem_analogy(item)
        current_entry["descendants"].append(descendant)
        current_entry["descendant_info"].append(item.text)
        return
    # anton: why is there sometimes some amount of indentation?
    if first_indent is not None:
        # print(first_indent, item.text)
        # breakpoint()
        pass
    descendant = match_stem(item)
    current_entry["descendants"].append(descendant)
    current_entry["descendant_info"].append(markup_paragraph(item))
    current_entry["descendants"][-1]["stem"]["is_sub_entry"] = item.paragraph_format.element.style == "LIN3"


def process_descendant_continuation(item, first_indent, current_entry):
    global num_parts
    num_parts["box4"] += 1
    # todo: currently descendant continuation info needs something more complex that a state machine (AKA regex will not work)
    #  This could possibly be built with some combo of regex and other logic, but currently is too difficult a task to attempt.
    # match_reflex_paragraph(item)

    # anton: note the indent info, currently we don't use it to mean anything but it might.
    indent_info = f"[indented by {first_indent}]" if first_indent is not None else ""
    current_entry["descendant_info"][-1] = current_entry["descendant_info"][-1].rstrip() + "\n\t" + indent_info + item.text.lstrip()
    pass


def process_other_initial(item, first_indent, current_entry):
    global num_parts
    num_parts["box5"] += 1
    num_parts["box6"] += 1
    indent_info = f"[indented by {first_indent}]" if first_indent is not None else ""
    current_entry["other"].append(indent_info + item.text)
    pass


def process_other_continuation(item, first_indent, current_entry):
    global num_parts
    num_parts["box6"] += 1
    indent_info = f"[indented by {first_indent}]" if first_indent is not None else ""
    current_entry["other"][-1] = current_entry["other"][-1].rstrip() + "\n" + indent_info + item.text.lstrip()
    pass


def process_footnotes(item, current_entry):
    global num_parts
    num_parts["box7"] += 1
    current_entry["footnotes"].append(markup_paragraph(item))
    pass


def match_nil_parts(document):
    # for tracking statistics about the doc
    global num_parts

    # tracking states
    cell_state = 0
    other_state = False

    # tracking current entry
    current_entry = None

    # normal styles found in most descendant and other listings
    normal_styles = ["LIN1", "LIN3", "BodyText"]

    # storing the results
    all_entries = []

    possible_styles = set()

    for item, item_type in tqdm(iter_block_with_type(document), ncols=150, total=len(list(iter_block_with_type(document)))):
        possible_styles.add(item.paragraph_format.element.style)
        # first skip anything that is an empty line. Empty lines also signify the end of an "other" items block
        if item.text.strip() == "":
            other_state = False
            continue

        # debug printout
        # print(f"---{item_type}---")
        # print(f"text= {item.text}")
        # print(f"style= {item.paragraph_format.element.style}")
        # print(f"indent= {item.paragraph_format.first_line_indent}")

        # various properties of the text
        regex_item = RegexEqual(item.text)
        style = item.paragraph_format.element.style
        first_indent = item.paragraph_format.first_line_indent

        # look for new root entries. These are either in a table (hence "cell") or are empty of any entries and thus not in tables
        # anton: why are some not in tables? What does this mean?
        if item_type == "cell":
            # cells come in two parts (and never more than 2). We keep track of these
            cell_state += 1
            current_entry = process_cell(item, cell_state, current_entry, all_entries)
            continue
        elif style == "Heading6":
            # breakpoint()
            global num_weird_headings
            current_entry = process_weird_heading(item, current_entry, all_entries)
            num_weird_headings += 1
            continue
        else:
            cell_state = 0

        # There is a weird block of items classified as "other" items.
        # We have to mark where they start and end because they start with a signifier and end in an empty line, but are otherwise the same as descendants.
        if regex_item == "^Sonstige":
            other_state = True
            continue

        # format for descendant blocks:
        #   paragraph type, not in other_state, and some additional conditions
        if not other_state and item_type == "paragraph" and style != "Literatur2":
            # initial blocks also match some regex and are normally styled
            # regex: at the beginning of the line ("^")
            #   may have a number (e.g. "1."),
            #   may have a space,
            #   may have a [,
            #   may have a ? (which may be followed by another space sometimes),
            #   always has a * or an x, which is always followed by something else
            if style in normal_styles and regex_item == "^\?? ?(?:\d\.)? ?\??\[? ?[\*x-].+":
                process_descendant_initial(item, first_indent, current_entry)
                continue

            # continuation blocks are also normally styled, and either start with a tab or have some first line indent
            # anton: The first line indent may actually mean something, I don't know.
            # regex: begins the line ("^") with a tab ("\t").
            elif style in normal_styles and (regex_item == "^\t" or first_indent is not None):
                process_descendant_continuation(item, first_indent, current_entry)
                continue

            # other continuation blocks just have a style of LIN2, which seems relatively consistent.
            # anton: this again might mean something, but currently I do not know what it means.
            #  These also sometimes have first_indent, which may mean something
            elif style == "LIN2":
                process_descendant_continuation(item, first_indent, current_entry)
                continue

            # at least one of the conditions above should have tripped, so this breakpoint should never be reached.
            # breakpoint()

        # format for the "other" block.
        # it always starts with a line on its own with "Sonstige" (this is handled above), and being a paragraph
        if other_state and item_type == "paragraph" and style != "Literatur2":
            # initial blocks start the line with: optional ?, optional space, then either a double dagger (‡) or a star (*)
            if regex_item == "^\?? ?[\‡\*]":
                process_other_initial(item, first_indent, current_entry)
                continue
            # continuation blocks start either with a tab (with LIN1 style), or are in specific styles (LIN2/LIN3), or contain an indent
            # anton: The indent might mean something, idk
            elif (regex_item == "^\t" and style == "LIN1") or (style in ["LIN2", "LIN3"]) or (style == "LIN1" and first_indent is not None):
                process_other_continuation(item, first_indent, current_entry)
                continue

        # format for footnotes.
        # anton: I do not currently try to do anything with these but eventually should figure out what number they start with (something that is persistent
        #  across several lines) and associate then with that number.
        if style == "Literatur2":
            process_footnotes(item, current_entry)
            continue
        breakpoint()
        pass
    return all_entries


def extract_number_and_text(s):
    match = re.match(r'^(\d+)\s*(.*)', s)
    if match:
        number = match.group(1)
        text = match.group(2)
        return number, text
    else:
        return None, s


def categorize_footnotes(entry):
    # for each footnote
    numbered_footnotes = defaultdict(list)
    latest_number = None
    for footnote in entry["footnotes"]:
        number, text = extract_number_and_text(footnote)
        if number in numbered_footnotes:
            # this should never happen
            breakpoint()
        # if the footnote starts with a number it makes a new set of footnotes
        if number is not None:
            numbered_footnotes[number].append(text)
            latest_number = number
        # otherwise it adds to the last set of footnotes
        elif latest_number is not None:
            numbered_footnotes[latest_number].append(text)
        # if this happens before a new set of footnotes is added, then something went really wrong
        else:
            breakpoint()
    entry["numbered_footnotes"] = dict(numbered_footnotes)
    # in the very last footnote (if it exists) look for a bunch of tabs. whatever is after that is the abbreviation of the author(s) that wrote that entry

    entry["footnote_attribution"] = None
    if len(numbered_footnotes) > 0:
        last_footnote = numbered_footnotes[list(numbered_footnotes)[-1]][-1]
        matches = re.findall(r"\t*(\(\S+\))$", last_footnote)
        if len(matches) > 0:
            entry["footnote_attribution"] = matches[-1].strip()

    return entry


def reflex_prompt(stem, info):
    prompt = "\n".join([
        (
            "I need you to help digitize entries in a German Proto-Indo European etymological dictionary. "
            "These are structured generally with a stem (which itself can be questionable, have a gender, etc.) followed by a tab with the reflex info. "
            "Sometimes the stem and following tab is missing, I will try to give it to you separately but it may also be missing (labeled as None) so wherever it is present you should use that information. "
            "After the tab (if the stem exists in the text) the reflex info consists of some amount of lines that generally start with a tab, then the German abbreviation for the language the reflex is in, the reflex itself (which is generally but not always surrounded with // to indicate italics), whether it is questionable (it will be marked as questionable by a leading '?'), its gender abbreviation if applicable, the gloss in quotes, and where it was first attested (generally in parentheses). "
            "Occasionally there is a number surrounded by '^^' (which used to indicate that it is superscript text) that refers to the footnotes, which needs to be extracted. "
            "Sometimes there are multiple reflexes for a single language often separated by a semicolon, in these cases create a new reflex entry for each, duplicating the relevant info for each entry. "
            "Sometimes there are multiple glosses for a single reflex, generally quoted and separated by a semicolon too, keep these together as a single string."
        ),
        "",
        "I need you to extract what the reflex is, and for each reflex note its:",
        "- language abbreviation",
        "- the reflex",
        "- questionable (true or false)",
        "- gloss",
        "- gender (if it exists, from m., f., n., c.)",
        "- first_attested (if they exist)",
        "- attested_info",
        "- footnote_num",
        "- other_abbr if there are other abbreviations unrecognized, or '' if there aren't",
        "- unknown_text, if you see some text that is otherwise impossible to classify",
        "",
        "I also want you to find the following information for the stem:",
        "- stem",
        "- stem_questionable (true or false)",
        "- stem_footnote_num (if they exist)",
        "- stem_gender",
        "",
        f"Here is the text for the stem '{stem}':" if stem is not None else "I could not find the stem myself in the text, regardless here is the text:",
        "```",
        f"{info}",
        "```",
        "",
        "```json",
        "{",
        "  'stem': string,",
        "  'stem_questionable': bool,",
        "  'stem_footnote_num': string|null,",
        "  'stem_gender': string|null,",
        "  'reflexes': [",
        "    {",
        "      'language_abbreviation': string,",
        "      'reflex': string,",
        "      'questionable': bool,",
        "      'gloss': string,",
        "      'gender': string|null,",
        "      'first_attested': string|null,",
        "      'attested_info': string|null,",
        "      'footnote_num': string|null,",
        "      'other_abbr': string|null,",
        "      'unknown_text': string|null",
        "    }",
        "  ]",
        "}",
        "```",
    ])
    expected_schema = {
        'stem': str,
        'stem_questionable': bool,
        'stem_footnote_num': (str, type(None)),
        'stem_gender': (str, type(None)),
        'reflexes': [
            {
                'language_abbreviation': str,
                'reflex': str,
                'questionable': bool,
                'gloss': (str, type(None)),
                'gender': (str, type(None)),
                'first_attested': (str, type(None)),
                'attested_info': (str, type(None)),
                'footnote_num': (str, type(None)),
                'other_abbr': (str, type(None)),
                'unknown_text': (str, type(None))
            }
        ]
    }

    return prompt, expected_schema


def gpt_entries(all_entries):
    global num_gpt_skipped
    from pyperclip import copy, paste
    total_entries = len(all_entries)
    for entry_count, entry in enumerate(all_entries):
        total_descendants = len(entry["descendant_info"])
        assert len(entry["descendant_info"]) == len(entry["descendants"])
        for descendant_count, (descendants, info) in enumerate(zip(entry["descendants"], entry["descendant_info"])):
            if descendants["stem"]["stem"] is None:
                num_gpt_skipped += 1
                # continue

            prompt, expected_schema = reflex_prompt(descendants["stem"]["stem"], info)

            messages, response = gpt_functions.query_gpt(
                [prompt],
                model="gpt-4o-mini",
                json_mode=True,
                note=f"{descendant_count+1}/{total_descendants} -> {entry_count+1}/{total_entries}",
                expected_schema=expected_schema
            )

            json_text = gpt_functions.extract_code_block(response['content'])
            reflexes = json.loads(json_text)

            # missing, extra = gpt_functions.check_schema(expected_schema, reflexes)
            # if not (len(missing) == 0 and len(extra) == 0):
            #     digest, prompt_string = gpt_functions.get_proper_digest("gpt-4o-mini", None, [prompt])
            #     breakpoint()
            #     gpt_functions.delete_response_digest(digest)
            #     # exit()

            # sanity check to make sure that the stems match
            if not reflexes["stem"] == descendants["stem"]["stem"]:
                # breakpoint()
                pass

            descendants["reflexes"] = reflexes["reflexes"]
            descendants["gpt_stem"] = {k: v for k, v in reflexes.items() if "stem" in k}
    gpt_functions.print_cost()
    pass


def csv_nil():
    with open("nil_gpt_attempt2.pkl", 'rb') as f:
        all_entries = pickle.load(f)
    # unroll everything into a flat list of entries (with no sub lists)
    csv_entries = []
    for entry in all_entries:
        descendants = entry["descendants"]
        for descendant in descendants:
            stem = descendant["stem"]
            gpt_stem = descendant["gpt_stem"]
            reflexes = descendant["reflexes"]
            for reflex in reflexes:
                csv_entry = {
                    # root info
                    "root": ("?" if entry["questionable"] and entry["questionable"] is not None else "") + entry["root"],
                    "gloss": entry["gloss"],
                    # stem info
                    "stem": ("?" if (stem["questionable"] or False) else "") + (stem["stem"] or "Not Found"),
                    "stem_gender": stem["gender"],
                    "stem_cite": stem["cite"],
                    "stem_gender_cite": stem["gender_cite"],
                    "gpt_stem": ("?" if (gpt_stem["stem_questionable"] or False) else "") + (gpt_stem["stem"] or "Not Found"),
                    "gpt_stem_gender": gpt_stem["stem_gender"],
                    # reflex info
                    "reflex": reflex["reflex"],
                    "reflex_gloss": reflex["gloss"],
                    "reflex_gender": reflex["gender"] or "",
                    "reflex_lang": reflex['language_abbreviation'] or "",
                    "first_attested": reflex["first_attested"] or "",
                    "other_attested_info": reflex["attested_info"] or "",
                    "footnote_num": reflex["footnote_num"] or "",
                    "other_abbreviations": reflex["other_abbr"],
                    "other_unknown_text": reflex["unknown_text"],
                    "footnote": "\n".join(entry["numbered_footnotes"].get(reflex["footnote_num"], "")),
                }
                csv_entries.append(csv_entry)
    df = pd.DataFrame(csv_entries)
    df.to_csv("data_nil/gpt_corrections/NIL - GPT Organized.csv")
    # breakpoint()


def main():
    document = Document('data_nil/NIL (edited).docx')

    # all_entries = run_or_load("temp.pkl", match_nil_parts, document=document)
    all_entries = run_or_load("temp.pkl", match_nil_parts, rerun=True, document=document)
    # all_entries = match_nil_parts(document)
    for entry in all_entries:
        categorize_footnotes(entry)
    # breakpoint()
    # STATS for those interested:
    # 4 special cases for headers
    # 5935 different reflexes (number of lines in either the box 4 or 6 in the chart).
    gpt_entries(all_entries)
    with open("nil_gpt_attempt2.pkl", 'wb') as f:
        pickle.dump(all_entries, f)
    # breakpoint()
    pass


if __name__ == '__main__':
    main()
    csv_nil()
    pass
