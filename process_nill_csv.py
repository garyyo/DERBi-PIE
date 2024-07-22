from dataclasses import dataclass

from docx import Document
from docx.document import Document as DocType
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph
import regex as re


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
    sources = item.text.split(',')
    return sources


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
        stem_groups = re.match(r"^(\? ?)?(\*(?:[^\s^]|\^\^é\^\^)+)(\^\^\d{0,2}\^\^)?( \(?[mfnc]\.\)?)?(\^\^\d{0,2}\^\^)?$", stem_text.strip()).groups()
        questionable_stem, stem, stem_cite, gender, gender_cite = stem_groups
        questionable_stem = questionable_stem is not None
        stem_cite = stem_cite.replace("^", "") if stem_cite is not None else None
        gender_cite = gender_cite.replace("^", "") if gender_cite is not None else None
    except AttributeError:
        # if it errors there is not much I can do, just silently move on.
        stem = None
        questionable_stem = None
        stem_cite = None
        gender = None
        gender_cite = None

    descendant = {
        "stem": {
            "stem": stem,
            "questionable": questionable_stem,
            "cite": stem_cite,
            "gender": gender,
            "gender_cite": gender_cite
        },
        "reflexes": []
    }

    # Only add a new reflex if there is one
    # if info_text is not None:
    #     descendant["reflex"].append(match_reflex(info_text))
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
    # make new entry if we see a new cell
    if cell_state == 1:
        all_entries.append(make_new_entry())
        current_entry = all_entries[-1]

        # match the heading as best we can
        current_entry["questionable"], current_entry["root"], current_entry["gloss"], current_entry["root_cite"] = match_heading(item)

        # a safety precaution to make sure we are not missing anything
        current_entry["cell_info"].append(item.text)
        pass
    if cell_state == 2:
        current_entry["cell_info"].append(item.text)
        current_entry["sources"] = match_heading_sources(item)
        pass
    return current_entry


def process_descendant_initial(item, first_indent, current_entry):
    # anton: why is there sometimes an arrow?
    if "→" in item.text:
        # breakpoint()
        return
    # anton: why is there sometimes some amount of indentation?
    if first_indent is not None:
        # breakpoint()
        pass
    descendant = match_stem(item)
    current_entry["descendants"].append(descendant)
    current_entry["descendant_info"].append(item.text)


def process_descendant_continuation(item, first_indent, current_entry):
    # todo: currently descendant continuation info needs something more complex that a state machine (AKA regex will not work)
    #  This could possibly be built with some combo of regex and other logic, but currently is too difficult a task to attempt.
    # match_reflex_paragraph(item)

    # anton: note the indent info, currently we don't use it to mean anything but it might.
    indent_info = f"[indented by {first_indent}]" if first_indent is not None else ""
    current_entry["descendant_info"][-1] = current_entry["descendant_info"][-1].rstrip() + "\n" + indent_info + item.text.lstrip()
    pass


def process_other_initial(item, first_indent, current_entry):
    indent_info = f"[indented by {first_indent}]" if first_indent is not None else ""
    current_entry["other"].append(indent_info + item.text)
    pass


def process_other_continuation(item, first_indent, current_entry):
    indent_info = f"[indented by {first_indent}]" if first_indent is not None else ""
    current_entry["other"][-1] = current_entry["other"][-1].rstrip() + "\n" + indent_info + item.text.lstrip()
    pass


def match_nil_parts(document):
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

    # collecting stats
    num_weird_headings = 0

    for item, item_type in iter_block_with_type(document):
        possible_styles.add(item.paragraph_format.element.style)
        # first skip anything that is an empty line. Empty lines also signify the end of an "other" items block
        if item.text.strip() == "":
            other_state = False
            continue

        # debug printout
        print(f"---{item_type}---")
        print(f"text= {item.text}")
        print(f"style= {item.paragraph_format.element.style}")
        print(f"indent= {item.paragraph_format.first_line_indent}")

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
            current_entry["footnotes"].append(item.text)
            continue
        breakpoint()
        pass
    breakpoint()


def main():
    document = Document('data_nil/NIL (edited).docx')

    match_nil_parts(document)
    breakpoint()
    # STATS for those interested:
    # 4 special cases for headers
    # 5935 different reflexes (number of lines in either the box 4 or 6 in the chart).
    pass


if __name__ == '__main__':
    main()
    pass
