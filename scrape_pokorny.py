import json
import re

import urllib.request
import urllib.parse
from collections import defaultdict

import requests
from bs4 import BeautifulSoup


def mark_text_effects(p):
    # Loop through each <i> tag
    italic_tags = p.find_all("i")
    for italic_tag in italic_tags:
        # Replace <i> tags with "\\"
        italic_tag.replace_with(r"\\" + italic_tag.text + r"\\")

    # Loop through each <u> tag
    underline_tags = p.find_all("u")
    for underline_tag in underline_tags:
        # Replace <i> tags with "\\"
        underline_tag.replace_with(r"__" + underline_tag.text + r"__")
    return p


def extract_entries(root_urls):
    entries = []
    # get the web page for each root
    for root, url in root_urls:
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
        for p in paragraphs:
            # if there are italic tags then we need to preserve that by replacing them with \\
            # todo: figure out if we need to do this for other formatting.
            p = mark_text_effects(p)

            # the p tags seem to be in form "label" + "\xa0"*n + "value"
            # n meaning some amount of that character (I think it's for spacing)
            # if there is nothing there then it's an empty line
            splits = p.get_text().split("\xa0")
            if len(splits) <= 1:
                continue

            label = splits[0].strip()
            value = splits[-1].strip()

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


if __name__ == '__main__':
    main()
    pass
