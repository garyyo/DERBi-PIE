import json
import re
from collections import defaultdict

import pandas as pd


csv_kwargs = {"index": False}


def parse_string(input_string):
    if "(possibly)" in input_string:
        breakpoint()
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

        node_data = parse_string(line["text"])

        child_node = (node_data["language"], node_data["word"]) if node_data is not None else None

        # todo: sometimes there just is not a word in the intermediary, for now I will ignore those.
        if node_data is not None and node_data["word"].strip() == "":
            continue

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
    wiki_data = {}
    with open("data_wiktionary/kaikki.org-dictionary-ProtoIndoEuropean.json", encoding="utf-8") as fp:
        for line in fp:
            data = json.loads(line)
            wiki_data[data["word"]] = data

    edges = [edge for entry in wiki_data.values() for edge in process_descendants(entry)]
    nodes = {node for edge in edges for node in edge}

    # write to csv for visualization
    df = pd.DataFrame([{"source": source, "target": target} for (source, target) in edges])
    df.to_csv("data_wiktionary/graph.csv", **csv_kwargs)
    # breakpoint()
    pass


if __name__ == '__main__':
    main()
    pass
