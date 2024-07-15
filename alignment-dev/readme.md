### Generative AI Use

AI was used sparingly in this project to generate code snippets. Very few of these remain in the project in their original unaltered form aside from any function that plot data using mpl and seaborn which only contain minor hand edits to further customize the visualization to the author's (Anton's) liking.

Listing out all function that used any AI generated code:
- `is_valid_word()`, ChatGPT translated my text description of the regex into regex. This was largely replaced and tweaked as needed.
- `load_*()`, ChatGPT copied over my method of either saving the results of a step to file if the file does not exist, or loading it from file if it does. The original was written by me.
- translation related functions, ChatGPT suggested the use of the deep_translator library. No code was written by chatgpt.
- `break_in_two_and_try_again()`, ChatGPT was used to write the original recursive framework of this from my description, but this was largely replaced by handwritten code as to not crash
- `split_data()`, written by ChatGPT to just save time, too simple to need modification. 
- `plot_heatmaps()`, written by ChatGPT to quickly visualize the data. unchanged
- `plot_heatmaps2()`, likewise, unchanged.
- `plot_heatmaps_paper()`, used plot_heatmaps2 as a starting point, but heavily rewritten.