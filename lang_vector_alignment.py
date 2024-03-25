import copy
import gc
import re

import gensim
import gensim.models
import numpy as np
import fasttext
import fasttext.util
import pandas as pd
import spacy
import errno
import scipy.spatial.distance

import alignment_code.unsup_align
import alignment_code.unsup_multialign


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
    return vec_save_path


def ft_vec_to_model(vec_load_path):
    # we load an unaligned version of the same model, then overwrite the vectors with the aligned version
    model = gensim.models.KeyedVectors.load_word2vec_format(vec_load_path, binary=False)
    return model


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
    return lexicon_save_path


def analogize(model, w1, w2, w3, **kwargs):
    return model.most_similar(negative=[w1], positive=[w2, w3], **kwargs)


def test_french_analogies(model_fr, model_fr_aligned):
    compares = [
        ("roi", "femme", "homme"),  # King is to queen as man is to woman
        ("amour", "guerre", "paix"),  # Love is to hate as peace is to war
        ("dieu", "monde", "ciel"),  # God is to creator/Lord as heaven is to the world
        ("berger", "femme", "homme"),  # Shepherd (feminine) as shepherd is to man
        ("sagesse", "fou", "sage"),  # Wisdom is to folly as the wise man is to the fool
        ("péché", "repentance", "pécheur"),  # Sin is to salvation/redemption as sinner is to repentance
        ("prophète", "message", "dieu"),  # Prophet is to revelation as God is to message
        ("roi", "justice", "guerre")  # King is to judge/peace as war is to justice
    ]
    print("for initial/aligned respectively:")
    for word1, word2, word3 in compares:
        initial_analogy, _ = analogize(model_fr, word1, word2, word3, topn=1)[0]
        aligned_analogy, _ = analogize(model_fr_aligned, word1, word2, word3, topn=1)[0]
        correct = '✓' if initial_analogy == aligned_analogy else '✗'
        print(f"{correct} {word1:10} -> {word2:10} = {word3:10} -> {initial_analogy:>15} / {aligned_analogy:20}")


def main():
    # load two models as the source language models. these will be used to recreate the target language model
    model_fr: gensim.models.KeyedVectors = gensim.models.KeyedVectors.load("alignment/fr_bible_model.bin")
    model_es: gensim.models.KeyedVectors = gensim.models.KeyedVectors.load("alignment/es_bible_model.bin")
    # a third model to be a target. the source models will attempt to recreate this
    model_la: gensim.models.KeyedVectors = gensim.models.KeyedVectors.load("alignment/la_bible_model.bin")

    # select words between the two models that are considered roughly equivalent
    # todo: this needs to eventually be switched to find words that take a path from source 1 (french) to source 2 (spanish) instead of going through latin
    latin_to_french, french_to_latin = find_words(model_la.index_to_key, "Latin", model_fr.index_to_key, "French")
    latin_to_spanish, spanish_to_latin = find_words(model_la.index_to_key, "Latin", model_es.index_to_key, "Spanish")
    french_to_spanish = chain_translation(french_to_latin, latin_to_spanish)
    spanish_to_french = chain_translation(spanish_to_latin, latin_to_french)

    # save the vec and lexicon files to do alignment (anton: eventually remove all the file io stuff maybe)
    model_source_1_path ="alignment/unaligned_models/es_bible_model.vec"
    model_source_2_path ="alignment/unaligned_models/fr_bible_model.vec"
    model_es.save_word2vec_format(model_source_1_path)
    model_fr.save_word2vec_format(model_source_2_path)

    # the lexicon
    # lexicon_path = ft_save_lexicon(spanish_to_french, "alignment/unaligned_models/es_to_fr.lex")

    # align to that
    # alignment_code.unsup_align.main(
    #     model_source_1_path, model_source_2_path, lexicon_path,
    #     "alignment/aligned_models/es_aligned_model.vec", "alignment/aligned_models/fr_aligned_model.vec",
    #     nmax=min(len(model_es.index_to_key), len(model_fr.index_to_key))
    # )
    # aligned_file_es = "alignment/aligned_models/es_aligned_model.vec"
    # aligned_file_fr = "alignment/aligned_models/fr_aligned_model.vec"

    # multi alignment, needed to align more than one model together (which even in the base case of 2 languages is needed for comparison to the target)
    # aligned_file_es, aligned_file_fr, aligned_file_la = alignment_code.unsup_multialign.main(
    #     ["es", "fr", "la"], max_load=min(len(model_es.index_to_key), len(model_fr.index_to_key))
    # )
    aligned_file_es, aligned_file_fr, aligned_file_la = ('alignment/aligned_models/es-ma[es,fr,la].vec', 'alignment/aligned_models/fr-ma[es,fr,la].vec', 'alignment/aligned_models/la-ma[es,fr,la].vec')

    # load the aligned version now, but I can only load as a keyed vector.
    model_es_aligned = gensim.models.KeyedVectors.load_word2vec_format(aligned_file_es, binary=False)
    model_fr_aligned = gensim.models.KeyedVectors.load_word2vec_format(aligned_file_fr, binary=False)
    # model_la_aligned = gensim.models.KeyedVectors.load_word2vec_format(aligned_file_la, binary=False)

    test_french_analogies(model_fr, model_fr_aligned)

    es_to_fr_aligned_cos = []
    es_to_fr_normal_cos = []
    for es_word, fr_words in spanish_to_french.items():
        fr_word = fr_words[0]
        distance_aligned = scipy.spatial.distance.cosine(model_es_aligned[es_word], model_fr_aligned[fr_word])
        distance_normal = scipy.spatial.distance.cosine(model_es[es_word], model_fr[fr_word])
        print(f"{es_word + ' - ' + fr_word:25} : {distance_aligned:.4f} vs {distance_normal:.4f}")
        es_to_fr_aligned_cos.append(distance_aligned)
        es_to_fr_normal_cos.append(distance_normal)

    print(f"combined distance aligned: {np.sum(es_to_fr_aligned_cos):.4f} / {len(spanish_to_french)} = {np.mean(es_to_fr_aligned_cos) * 100:.4f}%")
    print(f"combined distance normal: {np.sum(es_to_fr_normal_cos):.4f} / {len(spanish_to_french)} = {np.mean(es_to_fr_normal_cos) * 100:.4f}%")
    breakpoint()

    pass

    # create a new model descendant model based on the two aligned children
    # (take words that exist in test model, find their equivalent in the aligned ones, the average of the two models is the new point)
    pass

    # align the test model to the originals
    pass

    # compare the test model and the recreated model
    pass

    # breakpoint()
    pass


if __name__ == '__main__':
    main()
    pass
