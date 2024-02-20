import json
import re
from collections import defaultdict

import pandas as pd
import tqdm

csv_kwargs = {"index": False}


def parse_string(input_string):
    # breakdown by parts
    broken_string = input_string

    # find language
    matches = re.findall(r"([^:]+):\s?", broken_string)
    language = matches[0] if len(matches) else ""

    if language == "":
        return [None]

    # remove language from the string
    broken_string = broken_string[len(language)+1:].strip()

    if broken_string == "":
        return [None]

    # find comma boundaries
    matches = re.finditer(r",(?![^\(]*\))", broken_string)
    if matches is None:
        matches = []

    # for each found entry
    sequence_start = 0
    entries = []
    for match_obj in matches:
        found_entry = broken_string[sequence_start:match_obj.span()[0]].strip()
        sequence_start = match_obj.span()[1]
        entries.append(found_entry)
    found_entry = broken_string[sequence_start:].strip()
    entries.append(found_entry)

    parsed_entries = []
    # process the entries into ones that can be returned (language, word, pronunciation, meaning)
    for entry in entries:
        matches = re.findall(r"([^\(]+)(?:\(.+\))?", entry)
        if len(matches) == 0:
            continue
        word = matches[0].strip()
        parsed_entries.append({
            "language": language,
            "word": word
        })
    return parsed_entries if len(parsed_entries) != 0 else [None]


def old_parse_string(input_string):
    matches = re.findall(r"([^:]+):\s?([^\s]*)\s*(?:\((.+), [\"“]([^\"]+)[\"”]\))?", input_string)

    return dict(zip(["language", "word", "pronunciation", "meaning"], matches[0])) if len(matches) else None


# takes in an entry and returns a list of edges: (language, word) -> (descended language, descended word)
# sometimes the edges returned will include languages that have not yet been processed
#   e.g. a PIE root may return edges from a proto-hellenic word to an ancient greek one
def process_descendants(entry):
    # the "descendants" list needs to be processed linearly, paying attention to depth.
    #   depth 1: some sort of text relating to a form of this root?
    #   depth 2: Language name: descended form?
    #       the languages here seem to be mostly proto-x languages, tends to include links but some are for pages that have not been created.
    #   depth 3: Language name: word (pronunciation?, "meaning, sometimes comma separated")
    #   depth n: likely continues
    # Ideally we would only go from current language, but not all languages have pages, some are combined like this

    # gather the edges
    current_node = (entry["lang"], entry["word"])
    seen_nodes = [current_node] + [None for i in range(20)]
    edges = []
    # edges = [(("start_node", "start_node"), current_node)]

    if "descendants" not in entry.keys():
        return []

    for line in entry["descendants"]:
        current_depth = line["depth"] - 1
        # todo: handle links to other pages of forms of the same root.
        if current_depth == 0:
            continue

        # TODO: I currently trim to only the first word found. THIS NEEDS TO BE RECTIFIED ASAP
        node_data = parse_string(line["text"])[0]
        # node_data = old_parse_string(line["text"])

        child_node = None
        if node_data is not None:
            node_data["language"] = node_data["language"].strip("→⇒ \t\n")
            child_node = (node_data["language"], node_data["word"])

        # todo: sometimes there just is not a word in the intermediary, for now I will ignore those.
        if node_data is not None and node_data["word"].strip() == "":
            continue

        # sometimes it says "[script needed]" or something, which I have no idea how to deal with, so we will ignore it
        bad_words = [
            "[Term?]",
            "[script needed]",
            "Hieroglyphs needed",
            "[Book",
            "[Manichaean needed]"
        ]
        if any(word in line["text"] for word in bad_words):
            continue

        # if "stupeō" in line["text"]:
        #     breakpoint()

        # investigate_words = ["["]
        # if any(word in line["text"] for word in investigate_words):
        #     print(f"https://en.wiktionary.org/wiki/{entry['original_title']}")
        #     print(line["text"])
        #     breakpoint()
        #     continue

        seen_nodes[current_depth] = child_node

        for node in seen_nodes[:current_depth]:
            # if a node is ever none that means this part is flawed somehow and needs to be manually looked at.
            if node is None or child_node is None:
                continue
            k1, k2 = node
            if k1.strip() == "" or k2.strip() == "":
                breakpoint()
            if node is None or child_node is None:
                breakpoint()
            edges.append((node, child_node))

    # an explanation of the above for loops:
    #   we need to link from PIE to child terms, but we also need to link form intermediary children to final children
    #   to do this I keep track of which ones have been seen on previous levels, and only add nodes up to the seen level.
    #   This SHOULD mean that anything on previous levels will be overwritten when the time comes, but it might be written as None,
    #   which is a mistake in logic or in processing the data, or even a mistake in the data itself.
    #   For now, we just ignore those cases.
    return edges


def main():
    # load the preprocessed data from kaikki.org
    edges = set()
    # approx number of lines in the "all-words" version so I can get better time estimates
    num_lines = 9333951
    # with open("data_wiktionary/kaikki.org-dictionary-ProtoIndoEuropean.json", encoding="utf-8") as fp:
    with open("data_wiktionary/kaikki.org-dictionary-all-words.json", encoding="utf-8") as fp:
        for line in tqdm.tqdm(fp, total=num_lines):
            data = json.loads(line)
            # wiki_data[data["word"]] = data
            for edge in process_descendants(data):
                edges.add(edge)

    # edges = [edge for entry in wiki_data.values() for edge in process_descendants(entry)]
    nodes = {node for edge in edges for node in edge}

    print(f"num nodes: {len(nodes)}")

    # write to csv for visualization
    df = pd.DataFrame([{"source": source, "target": target} for (source, target) in edges])
    df.to_csv("data_wiktionary/graph.csv", **csv_kwargs)
    # breakpoint()
    pass


if __name__ == '__main__':
    main()
    pass
