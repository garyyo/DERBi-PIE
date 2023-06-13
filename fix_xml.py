import re
import pandas as pd


def remove_duplicate_attributes(xml_file):
    with open(xml_file, 'r', encoding="utf-8") as file:
        xml_data = file.read()

    # assume that each entry is on its own line
    xml_lines = xml_data.split("\n")

    # Regular expression pattern to match attribute duplicates
    pattern = r'(\w+)\s*=\s*(".*?")(?=.*\1)'

    # add each entry to a list
    entries = []
    # apply the regex to each line
    for line in xml_lines:
        attributes = re.findall('\s([A-Z]+)=\"(.*?)\"', line)
        keys = [key for key, value in attributes]
        duplicate_keys = set([key for key in keys if keys.count(key) > 1])

        # deduplicate keys
        corrected_attributes = dict(attributes)
        for dup_key in duplicate_keys:
            dup_values = [value for key, value in attributes if key == dup_key]
            # make sure the values are the same
            assert len(set(dup_values)) == 1
        # ignore the lines
        if len(corrected_attributes) > 0:
            entries.append(corrected_attributes)

    df = pd.DataFrame(entries)
    # some cols are just Y or N (or NaN), so we replace those with True and False
    binary_cols = [col for col in df.columns if len(df[col].unique()) <= 3]
    df[binary_cols] = df[binary_cols].replace("Y", True).replace("N", False)

    breakpoint()


def main():
    # Provide the path to your faulty XML file
    xml_file_path = 'pokorny_db/updatedpokorny.xml'
    remove_duplicate_attributes(xml_file_path)


if __name__ == '__main__':
    main()
    pass
