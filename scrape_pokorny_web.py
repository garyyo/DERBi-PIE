import json
import re

import urllib.request
import urllib.parse
from collections import defaultdict

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


"""
author - anton vinogradov

It scrapes the pokorny website: https://indo-european.info/pokorny-etymological-dictionary/
"""


def skip_leading_whitespace(text, surrounding):
    preamble = ""
    lead = 0
    for letter in text:
        if letter in [" ", "\xa0"]:
            preamble += letter
            lead += 1
        else:
            break
    postamble = ""
    follow = len(text)
    for letter in reversed(text):
        if letter in [" ", "\xa0"]:
            postamble = letter + postamble
            follow -= 1
        else:
            break
    return f"{preamble}{surrounding}{text[lead:follow]}{surrounding}{postamble}"


def mark_text_effects(p):
    # Loop through each <i> tag
    italic_tags = p.find_all("i")
    for tag in italic_tags:
        # Replace <i> tags with "\\"
        tag.replace_with(skip_leading_whitespace(tag.text, r"\\"))

    # Loop through each <u> tag
    underline_tags = p.find_all("u")
    for tag in underline_tags:
        tag.replace_with(skip_leading_whitespace(tag.text, r"__"))

    # loop through each <sup> tag
    sup_tags = p.find_all("sup")
    for tag in sup_tags:
        tag.replace_with(skip_leading_whitespace(tag.text, r"^^"))

    # loop through each <sub> tag
    sub_tags = p.find_all("sub")
    for tag in sub_tags:
        tag.replace_with(skip_leading_whitespace(tag.text, r"↓↓"))
    return p


def extract_entries(root_urls):
    entries = []
    # get the web page for each root
    for root, url in tqdm(root_urls, ncols=150):
        # url = f"https://indo-european.info/pokorny-etymological-dictionary/{urllib.parse.quote(root)}.htm"
        response = requests.get(url)

        # if it does not give a 200 response code then we are either slamming them (and thus need to stop), your internet went out, or the site is down.
        if response.status_code != 200:
            print("Did not get OK response for", root, "at", url)
            continue

        # it seems that everything is stored in paragraph tags, so we will just try to get all of those
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.findAll("p")

        entry = defaultdict(list)
        entry["root"].append(root)
        last_label = None
        for p_index, p in enumerate(paragraphs):
            # if there are italic tags then we need to preserve that by replacing them with \\
            p = mark_text_effects(p)

            # if it's the first line then it's the root
            if p_index == 0:
                entry["root"].append(p.get_text())
                continue

            # the p tags seem to be in form "label" + "\xa0"*n + "value" + (optionally more value lines in rare cases) + (optionally more "\xa0"*n)
            # "\xa0" being a non-breaking space
            # if there is nothing there then it's an empty line
            # we remove all "\xa0" to the right since any trailing nbsp's are going to mess things up.
            splits = p.get_text().rstrip("\xa0").split("\xa0")

            # if there are only
            if len(splits) < 1:
                continue

            label = splits[0].strip() if len(splits) > 0 else ""
            value = "".join([split.strip() for split in splits[1:] if split.strip() != ""]).strip() if len(splits) > 1 else ""

            # if there is no label use the last one we saw
            if len(label) == 0:
                if value == "" or last_label is None:
                    continue
                label = last_label

            entry[label].append(value)
            last_label = label
        entries.append(entry)
    return entries


def get_entry_urls():
    url_base = "https://indo-european.info/pokorny-etymological-dictionary/"
    contents_url = f"{url_base}contents.htm"

    response = requests.get(contents_url)

    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.findAll("p", {"class": "Toc2"})
    links = [p.find("a") for p in paragraphs]

    root_urls = []
    for link in links:
        if "href" not in link.attrs or link["href"][0] == "#":
            continue

        text = link.get_text()
        href = link["href"]

        url = f"{url_base}{href}"
        root_urls.append((text, url))
    return root_urls


def main():
    # get the urls
    root_urls = get_entry_urls()

    # get all the entries (this function may take >10 minutes as it does web requests)
    entries = extract_entries(root_urls)

    # save it!
    with open("data_pokorny/pokorny_scraped.json", "w", encoding="utf-8") as fp:
        json.dump(entries, fp)

    pass


def test():
    extract_entries([("test", "https://indo-european.info/pokorny-etymological-dictionary/aĝ.htm")])
    breakpoint()


if __name__ == '__main__':
    # test()
    main()
    pass
