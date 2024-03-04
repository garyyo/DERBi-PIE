import copy
import gc
import re

import gensim
import numpy as np
import fasttext
import fasttext.util
import pandas as pd
import spacy
import errno


def find_words(words_source, lang_source, words_target, lang_target):
    # load the graph up
    graph_df = pd.read_csv("data_wiktionary/graph_all.csv")
    pattern = r"^\(['\"](.+)['\"],\s*['\"](.+)['\"]\)$"
    new_cols = ['source_lang', 'source_word', 'target_lang', 'target_word']

    # just to make the rest much faster
    lang_df = graph_df[graph_df.source.str.contains(lang_source) & graph_df.target.str.contains(lang_target)]

    # splits the [source, target] cols into [source_lang, source_word, target_lang, target_word] since the data is there but combined. This is a bit slow.
    lang_df = pd.DataFrame([dict(zip(new_cols, list(re.findall(pattern, row[0])[0]) + list(re.findall(pattern, row[1])[0]))) for i, row in lang_df.iterrows()])

    translation_df = lang_df[lang_df.source_word.isin(words_source) & lang_df.target_word.isin(words_target)]
    translation_dict = translation_df.groupby('source_word')['target_word'].apply(list).to_dict()
    reverse_translation_dict = translation_df.groupby('target_word')['source_word'].apply(list).to_dict()
    return translation_dict, reverse_translation_dict


def chain_translation(source_to_intermediary, intermediary_to_target):
    # chain them
    source_to_target = {
        source_word: list({
            target_word
            for intermediary_word in intermediary_words
            for target_word in intermediary_to_target.get(intermediary_word, [])
        })
        for source_word, intermediary_words in source_to_intermediary.items()
    }
    # remove empty entries
    source_to_target = {k: v for k, v in source_to_target.items() if len(v)}
    return source_to_target


def ft_model_to_vec(model, vec_save_path):
    # save a model's vectors to this weird file format that is compatible with ft alignment scripts
    words = model.get_words()
    with open(vec_save_path, "w", encoding="utf-8") as fp:
        fp.write(f"{len(words)} {model.get_dimension()}\n")
        for word in words:
            vector = model.get_word_vector(word)
            vector_str = " ".join([str(v) for v in vector])

            fp.write(f"{word} {vector_str}\n")
    pass


def ft_vec_to_model(model, vec_load_path):
    # todo: write all this.
    breakpoint()
    pass


def ft_save_lexicon(source_to_target, lexicon_save_path):
    # save the lexicon (seems to be in format "word_source word_target\n" * num_lines).
    # and also seems to be the equivalent words in source and target,
    # anton: but this information was gathered by analyzing the source, and thus I do not know if it is correct.
    with open(lexicon_save_path, "w", encoding="utf-8") as fp:
        for source, target in source_to_target.items():
            # always pick the first word in the target list for simplicity
            target_chosen = target[0]
            if " " in source or " " in target_chosen:
                breakpoint()
            fp.write(f"{source} {target_chosen}\n")


def main():
    # load some data to align together
    model_ft_fr = fasttext.load_model("alignment/fr_bible_model_ft.bin")
    model_ft_es = fasttext.load_model("alignment/es_bible_model_ft.bin")

    # a third model to test against (currently ALSO being used to connect translation from es to fr)
    model_ft_la = fasttext.load_model("alignment/la_bible_model_ft.bin")

    # select words between the two models that are considered roughly equivalent
    latin_to_french, french_to_latin = find_words(model_ft_la.words, "Latin", model_ft_fr.words, "French")
    latin_to_spanish, spanish_to_latin = find_words(model_ft_la.words, "Latin", model_ft_es.words, "Spanish")
    french_to_spanish = chain_translation(french_to_latin, latin_to_spanish)
    spanish_to_french = chain_translation(spanish_to_latin, latin_to_french)

    # save the vec and lexicon files to do alignment outside of this script (anton: may eventually be put back in this script)
    ft_model_to_vec(model_ft_fr, "alignment/unaligned_models/fr_bible_model.vec")
    ft_model_to_vec(model_ft_es, "alignment/unaligned_models/es_bible_model.vec")

    # the lexicon
    ft_save_lexicon(spanish_to_french, "alignment/unaligned_models/es_to_fr.lex")

    # third model to align later to test against
    ft_model_to_vec(model_ft_la, "alignment/unaligned_models/la_bible_model.vec")

    ft_save_lexicon(spanish_to_latin, "alignment/unaligned_models/es_to_la.lex")

    # align the models

    breakpoint()
    pass

    # create a new model descendant model based on the two aligned children
    # (take words that exist in test model, find their equivalent in the aligned ones, the average of the two models is the new point)
    pass

    # align the test model to the originals
    pass

    # compare the test model and the recreated model
    pass

    breakpoint()
    pass


if __name__ == '__main__':
    main()
    pass
