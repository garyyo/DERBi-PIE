import json
import re
from collections import defaultdict

import pandas as pd
import tqdm

csv_kwargs = {"index": False}

# sometimes it says "[script needed]" or something, which I have no idea how to deal with, so we will ignore it
bad_words = [
    "[Term?]",
    "[script needed]",
    "Hieroglyphs needed",
    "[Book",
    "[Manichaean needed]"
]


def parse_string(input_string):
    # breakdown by parts
    broken_string = input_string

    # find language
    matches = re.findall(r"([^:]+):\s?", broken_string)
    language = matches[0] if len(matches) else ""

    if language == "":
        return [{"language": language, "word": None}]

    # remove language from the string
    broken_string = broken_string[len(language)+1:].strip()

    # remove some characters. anton: I have no clue what these mean.
    language = language.strip("→⇒ \t\n")

    if broken_string == "":
        return [{"language": language, "word": None}]

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

        # if we see any "[script needed]" or other patterns indicating that this is an incomplete entry then we ignore it and say that the word does not exist
        # anton: right now we are skipping those words, this may not be the best option
        if any(bad_word in word for bad_word in bad_words):
            # word = None
            continue

        parsed_entries.append({
            "language": language,
            "word": word
        })
    return parsed_entries if len(parsed_entries) != 0 else [{"language": language, "word": None}]


def old_parse_string(input_string):
    matches = re.findall(r"([^:]+):\s?([^\s]*)\s*(?:\((.+), [\"“]([^\"]+)[\"”]\))?", input_string)

    return dict(zip(["language", "word", "pronunciation", "meaning"], matches[0])) if len(matches) else None


def tree_descendants(entry):
    # reorganize the descendants data to be a nested object
    root = {
        "language": entry["lang"],
        "word": entry["word"],
        "descended": []
    }
    node_stack = [[root]]
    for line_num, line in enumerate(entry["descendants"]):
        depth = line["depth"] if line["depth"] > 0 else 1
        parsed_nodes = parse_string(line["text"])

        # if there are no nodes here
        if len(parsed_nodes) == 0:
            breakpoint()

        for node in parsed_nodes:
            if node is None:
                continue
            node["descended"] = []

        # if we went up the tree
        if depth < len(node_stack):
            # trim the stack to the proper depth
            node_stack = node_stack[:depth]

        current_nodes = node_stack[-1]

        # add nodes
        for node in current_nodes:
            node["descended"] += parsed_nodes

        node_stack.append(parsed_nodes)

        # todo: if we went sideways on the tree?
    return root


def process_descendants_new(entry):
    if "descendants" not in entry.keys():
        return []

    # reorganize the entire entry["descendants"] stuff into a tree, grab the root node of that tree
    root_node = tree_descendants(entry)
    trim_none_word_tree(root_node)

    # recursively
    edges = set()
    add_edges_recursive(root_node, edges)
    # add_edges_recursive(root_node, edges, [root_node])
    # breakpoint()
    return edges


def add_edges_recursive(node, edges, previous_nodes=None):
    for descendant in node['descended']:
        if descendant["word"] is not None:
            edges.add(((node["language"], node["word"]), (descendant["language"], descendant["word"])))
        else:
            breakpoint()

        if previous_nodes is not None:
            # add all intermediate nodes if we want to do that
            for previous in previous_nodes:
                edges.add(((previous["language"], previous["word"]), (descendant["language"], descendant["word"])))
            add_edges_recursive(descendant, edges, previous_nodes + [node])
        else:
            # otherwise just keep going
            add_edges_recursive(descendant, edges)

    pass


def trim_none_word_tree(node):
    old_descendants = node['descended']
    new_descendants = []
    while len(old_descendants) != 0:
        descendant = old_descendants.pop(0)
        if descendant["word"] is not None:
            # good nodes get further processed
            trim_none_word_tree(descendant)
            # good nodes get added back
            new_descendants.append(descendant)
        else:
            # the children of bad nodes get added to the "to be processed" list
            old_descendants += descendant["descended"]
            # bad nodes do not get added back
            pass
    # keep the good ones
    node['descended'] = new_descendants


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
    seen_nodes = [[current_node]] + [[None] for i in range(20)]
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
        # node_data = old_parse_string(line["text"])

        child_nodes = []
        for data in node_data:
            child_node = None
            if data is not None:
                data["language"] = data["language"].strip("→⇒ \t\n")
                child_node = (data["language"], data["word"])

            # todo: sometimes there just is not a word in the intermediary, for now I will ignore those.
            if data is not None and data["word"] is not None and data["word"].strip() == "":
                continue

            if any(word in line["text"] for word in bad_words):
                continue

            child_nodes.append(child_node)

        # if "stupeō" in line["text"]:
        #     breakpoint()

        # investigate_words = ["["]
        # if any(word in line["text"] for word in investigate_words):
        #     print(f"https://en.wiktionary.org/wiki/{entry['original_title']}")
        #     print(line["text"])
        #     breakpoint()
        #     continue

        seen_nodes[current_depth] = child_nodes

        # todo: this is a flawed implementation, redo this so that its just recursive
        for child_node in child_nodes:
            for nodes in seen_nodes[:current_depth]:

                for node in nodes:
                    # if a node is ever none that means this part is flawed somehow and needs to be manually looked at.
                    if node is None or child_node is None:
                        continue
                    # k1, k2 = node
                    # if k1 is None or k1.strip() == "" or k2 is None or k2.strip() == "":
                    #     breakpoint()
                    # if node is None or child_node is None:
                    #     breakpoint()
                    edges.append((node, child_node))

    # an explanation of the above for loops:
    #   we need to link from PIE to child terms, but we also need to link form intermediary children to final children
    #   to do this I keep track of which ones have been seen on previous levels, and only add nodes up to the seen level.
    #   This SHOULD mean that anything on previous levels will be overwritten when the time comes, but it might be written as None,
    #   which is a mistake in logic or in processing the data, or even a mistake in the data itself.
    #   For now, we just ignore those cases.
    return edges


def gen_graph(path, num_lines=None, out_name=""):
    # load the preprocessed data from kaikki.org
    edges = set()
    count_lines = 0

    with open(path, encoding="utf-8") as fp:
        for line in tqdm.tqdm(fp, total=num_lines, ncols=150):
            count_lines += 1
            data = json.loads(line)
            # wiki_data[data["word"]] = data
            for edge in process_descendants_new(data):
                edges.add(edge)

    nodes = {node for edge in edges for node in edge}

    print(f"num lines: {count_lines}")
    print(f"num nodes: {len(nodes)}")
    print(f"num edges: {len(edges)}")

    # write to csv for visualization
    df = pd.DataFrame([{"source": source, "target": target} for (source, target) in sorted(list(edges))])
    df.to_csv(f"data_wiktionary/graph_{out_name}.csv", **csv_kwargs)
    pass


def main():
    # config = {"path": "data_wiktionary/kaikki.org-dictionary-ProtoIndoEuropean.json", "num_lines": 1518, "out_name": "PIE"}
    config = {"path": "data_wiktionary/kaikki.org-dictionary-all-words.json", "num_lines": 9333951, "out_name": "all"}
    gen_graph(**config)
    pass


if __name__ == '__main__':
    main()
    pass
