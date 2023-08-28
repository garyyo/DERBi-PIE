# DERBi PIE

This project organizes the data to go into the DERBi PIE website. There are several scripts that are used to generate the data. The `generate_*_db_data.py` series of scripts are those to run to regen the data, which is stored in `data_*/table_*.json`. The `generate_*_db_data.py` scripts are run in the following order:
1. generate_pokorny_db_data.py
2. generate_liv_db_data.py
3. generate_common_db_data.py

These should be loaded into MongoDB in the `DERBI-PIE` database as the following collection names: `common`, `pokorny`, and `liv`, along with any other dictionaries we process (we will add them to this list as we go).

All others are helpers for those scripts and have docstrings explaining some amount of purpose and usage. The generated data is added to the git repository, so it is only necessary to run the scripts if some upstream data has changed, or the processing code is updated. 

## Individual Script Caveats

### generate_pokorny_db_data.py

This one is very messy as it requires a lot of details. The comments are most helpful in understanding what is going on here. It takes quite a while to run, maybe on the order of 10 minutes.

### generate_liv_db_data.py

This one directly queries GPT though this should be moved out eventually into its own script much like `translate_meaning_gpt.py`. As such it just will not work without a GPT api key, unless the data is already cached in `gpt_caches/`. If the gpt data is present it should run in seconds, but without it expect it to take up to hours. This is part of why the GPT stuff should be moved out to its own script.

### generate_common_db_data.py

This needs to be run last because it makes changes to pokorny and liv, and generates a proper common dictionary. The actual script is fast, but writing the data back to json is slow. Should take a couple seconds at most.

### translate_meaning_gpt.py

This script tries to translate something from german to english. a better translation service should be used, this is very hacky.

### scrape_pokorny_web.py

This script scrapes a pokorny website, best not to use this as it is quite outdated. It was used to generate the pokorny data that was missing from the UTexas LRC database dump (from `data_pokorny/table_dumps/*`)

### find_missing_pokorny_gpt.py

This script uses GPT to organize the data missing from the UTexas LRC database dump. 

