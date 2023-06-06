import json
import re

import urllib.request
import urllib.parse
from collections import defaultdict

import requests
from bs4 import BeautifulSoup


def extract_entries(roots):
    entries = []
    # get the web page for each root
    for root in roots:
        url = f"https://indo-european.info/pokorny-etymological-dictionary/{urllib.parse.quote(root)}.htm"
        response = requests.get(url)
        # fixme: ensure that we are getting a 200 response every time rather than assuming

        # it seems that everything is stored in paragraph tags, so we will just try to get all of those
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.findAll("p")

        entry = defaultdict(list)
        entry["root"].append(root)
        last_label = None
        for p in paragraphs:
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


def main():
    # todo: switch to using the actual page rather than relying on a downloaded copy
    # contents_url = "https://indo-european.info/pokorny-etymological-dictionary/contents.htm"
    # extract all the entries from the contents (a downloaded file)
    with open('pokorny_db/contents.html', 'r', encoding="utf-8") as fp:
        lines = fp.readlines()

    # the entries are extract using some funky regex, seems to work.
    # todo: replace this with bs4 parsing, not regex which is terrible
    roots = []
    for line in lines:
        match = re.search('(?<=a href=")(.+?)(?=\.htm)', line)
        if not match:
            continue
        roots.append(match.group(0))

    # get all the entries (this function may take >10 minutes as it does web requests)
    entries = extract_entries(roots)

    with open("data_pokorny/pokorny_scraped.json", "w", encoding="utf-8") as fp:
        json.dump(entries, fp)

    pass


def get_entry_urls():
    url_base = "https://indo-european.info/pokorny-etymological-dictionary/"
    contents_url = f"{url_base}contents.htm"

    response = requests.get(contents_url)

    soup = BeautifulSoup(response.content, "html.parser")
    contents_paragraph = soup.find('p', string='Contents')
    links = contents_paragraph.find_all_next('a')

    urls = []
    for link in links:
        text = link.get_text()
        href = link["href"]

        if href == "#":
            continue

        urls.append((text, href))
    return urls


if __name__ == '__main__':
    main2()
    pass
