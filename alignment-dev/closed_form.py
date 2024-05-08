import json
import os
import regex as re
from collections import defaultdict, Counter

import gensim
import numpy as np
import scipy
import spacy
from deep_translator import GoogleTranslator
import tqdm


def absolute_orientation_rotation(A, B):
    # calculate sum of outer products H of A and B
    H = np.sum(np.outer(a, b.T) for a, b in zip(A, B))

    # Decompose into U, S, and V
    U, S, Vt = np.linalg.svd(H)
    # Build rotation
    R = U * Vt
    # return ˜B = B*R so each ˜b_i = b_i . R
    B_rotated = np.array([R.dot(b_i) for b_i in B])
    return B_rotated


def absolute_orientation_centered(A, B):
    # Center
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)

    A_centered = A - a_mean
    B_centered = B - b_mean

    # Rotation
    B_centered_rotated = absolute_orientation_rotation(A_centered, B_centered)

    return B_centered_rotated


def absolute_orientation(A, B):
    a_mean = A.mean(axis=0)
    B_centered_rotated = absolute_orientation_centered(A, B)

    # Translation
    B_centered_rotated_translated = B_centered_rotated + a_mean

    return B_centered_rotated_translated


def compute_scaling(A, B):
    scaling = np.sum([np.inner(a, b) for a, b in zip(A, B)]) / np.power(np.linalg.norm(B), 2)
    return scaling


def absolute_orientation_scaling(A, B):
    B_rotated = absolute_orientation_rotation(A, B)
    # compute scaling here
    scaling = compute_scaling(A, B_rotated)
    B_rotated_scaled = B * scaling
    return B_rotated_scaled


def absolute_orientation_centered_scaling(A, B):
    # same as the centered version
    B_centered_rotated = absolute_orientation_centered(A, B)

    # compute scaling here
    scaling = compute_scaling(A, B_centered_rotated)
    B_rotated_scaled = B_centered_rotated * scaling
    return B_rotated_scaled


# nlp_fr = spacy.load("fr_core_news_sm")
nlp_es = spacy.load("es_core_news_sm")


# def lemmatize_word_spacy_french(word):
#     doc = nlp_fr(word)
#     return doc[0].lemma_


def analogize(model, w1, w2, w3, **kwargs):
    return model.most_similar(negative=[w1], positive=[w2, w3], **kwargs)


def test_french_analogies(model_fr, model_fr_aligned):
    compares = [
        ("roi", "femme", "homme"),  # King is to queen as man is to woman
        ("amour", "guerre", "paix"),  # Love is to hate as peace is to war
        ("dieu", "monde", "ciel"),  # God is to creator/Lord as heaven is to the world
        ("berger", "femme", "homme"),  # Shepherd (feminine) as shepherd is to man
        # ("sagesse", "fou", "sage"),  # Wisdom is to folly as the wise man is to the fool
        # ("péché", "repentance", "pécheur"),  # Sin is to salvation/redemption as sinner is to repentance
        ("prophète", "message", "dieu"),  # Prophet is to revelation as God is to message
        ("roi", "justice", "guerre")  # King is to judge/peace as war is to justice
    ]
    print("for initial/aligned respectively:")
    for word1, word2, word3 in compares:
        initial_analogy, _ = analogize(model_fr, word1, word2, word3, topn=1)[0]
        aligned_analogy, _ = analogize(model_fr_aligned, word1, word2, word3, topn=1)[0]
        correct = '✓' if initial_analogy == aligned_analogy else '✗'
        print(f"{correct} {word1:10} -> {word2:10} = {word3:10} -> {initial_analogy:>15} / {aligned_analogy:20}")


def is_valid_word(word):
    # This regex matches strings that consist only of letter characters (including those from non-Latin alphabets)
    # and mark characters for diacritics. It excludes numbers and special characters.
    pattern = re.compile(u'^[\p{L}\p{M}]+$')

    return bool(pattern.match(word))


def match_models(model_a, language_a, model_b, language_b):
    a_words = model_a.index_to_key
    b_words = model_b.index_to_key
    if not os.path.exists(f"word_matchups/{language_a}_to_{language_b}.json"):
        a_to_b_index = []

        # b_words_lemmatized = [nlp_es(word)[0].lemma_ if is_valid_word(word) else None for word in b_words]

        # get some translations for every word, but only if they contain no weird number or symbol characters.
        translated_words = []
        with open("word_matchups/fr_translations.json", "r") as fp:
            translate_dict = json.load(fp)
            translated_words = list(translate_dict.keys())

        for i, word in tqdm.tqdm(enumerate(a_words), ncols=160, total=len(a_words)):
            # redo translation for words that are after the 50% mark, are valid, (redo clause), are the same when translated, and are not capitalized (thus likely proper nouns)
            if ((i/len(a_words)) > 0.5) and is_valid_word(word) and word in translate_dict and word == translate_dict[word] and word.lower() == word:
                word = try_translate(a_words, language_a, language_b, translate_dict, translated_words, word)
                translated_words.append(word)
            elif word in translate_dict:
                translated_words.append(translate_dict[word])
            elif not is_valid_word(word):
                translated_words.append(None)
            else:
                word = try_translate(a_words, language_a, language_b, translate_dict, translated_words, word)
                translated_words.append(word)
            pass

        translate_dict = dict(zip(a_words, translated_words))
        with open('word_matchups/fr_translations.json', "w") as fp:
            json.dump(translate_dict, fp)

        with open('word_matchups/fr_translations.json', "r") as fp:
            translate_dict = json.load(fp)

        # todo: lemmatize only when needed,
        for word, translated_word in translate_dict.items():
            # if the word or the translated word are either not valid words (like if they are symbols or numbers or something), just skip, don't even try.
            if not is_valid_word(word):
                a_to_b_index.append(None)
                continue
            if not is_valid_word(translated_word):
                a_to_b_index.append(None)
                continue
            # translated_lemmatized_word = nlp_es(translated_word)[0].lemma_
            # if translated_lemmatized_word in b_words_lemmatized:
            #     new_index = b_words_lemmatized.index(translated_lemmatized_word)
            # anton: original way of doing it,
            if translated_word.lower() in b_words:
                new_index = b_words.index(translated_word.lower())
            else:
                new_index = None
            a_to_b_index.append(new_index)

        a_to_b = {a_words[i]: (b_words[j] if j is not None else None) for i, j in enumerate(a_to_b_index)}
        with open(f"word_matchups/{language_a}_to_{language_b}.json", "w") as fp:
            json.dump(a_to_b, fp)
    else:
        with open(f"word_matchups/{language_a}_to_{language_b}.json", "r") as fp:
            a_to_b = json.load(fp)

    # {word: count for word, count in Counter(a_to_b.values()).items() if count != 1}
    b_to_a = defaultdict(list)
    for a, b in a_to_b.items():
        b_to_a[b].append(a)
    missing_a = sorted(set(b_to_a[None]))
    missing_b = sorted(set(b_words) - set(b_to_a.keys()))

    keys_a = [word for word, translated_word in a_to_b.items() if translated_word is not None]
    vectors_a = np.array([model_a.get_vector(word) for word in keys_a])
    new_model_a = gensim.models.KeyedVectors(vectors_a.shape[1])
    new_model_a.add_vectors(keys_a, vectors_a)

    keys_b = [translated_word for word, translated_word in a_to_b.items() if translated_word is not None]
    vectors_b = np.array([model_b.get_vector(word) for word in keys_b])
    new_model_b = gensim.models.KeyedVectors(vectors_b.shape[1])
    new_model_b.add_vectors(keys_b, vectors_b)

    return new_model_a, new_model_b, a_to_b, b_to_a


def try_translate(a_words, language_a, language_b, translate_dict, translated_words, word):
    try:
        word = GoogleTranslator(source=language_a, target=language_b).translate(word)
        return word
    except Exception as err:
        print(err)
        translated_words.append(None)

        translate_dict = dict(zip(a_words, translated_words))
        with open('word_matchups/fr_translations_err.json', "w") as fp:
            json.dump(translate_dict, fp)
        return None


def main():
    # load datasets
    # todo: these datasets are invalid for what we are trying to do since words between them are not equivalent
    #  aka a_i ~= b_i DOES NOT hold true.
    #  we would need to either match the equivalent words, or restrict to the subset of words that we know to be equivalent
    model_fr: gensim.models.KeyedVectors = gensim.models.KeyedVectors.load("../alignment/fr_bible_model.bin")
    model_es: gensim.models.KeyedVectors = gensim.models.KeyedVectors.load("../alignment/es_bible_model.bin")

    new_model_fr, new_model_es, french_to_spanish, spanish_to_french = match_models(model_fr, "fr", model_es, "es")

    # memory management attempt
    del model_fr
    del model_es

    # apply the alignment to B
    model_fr_ao_vectors = absolute_orientation_centered_scaling(new_model_es.vectors.copy(), new_model_fr.vectors.copy())

    model_fr_ao: gensim.models.KeyedVectors = gensim.models.KeyedVectors(model_fr_ao_vectors.shape[1])
    model_fr_ao.add_vectors(new_model_fr.index_to_key, model_fr_ao_vectors)

    test_french_analogies(new_model_fr, model_fr_ao)

    es_to_fr_aligned_cos = []
    es_to_fr_normal_cos = []
    es_aligned_cos = []
    for fr_word, es_word in french_to_spanish.items():
        if fr_word is None or es_word is None:
            continue
        distance_aligned = scipy.spatial.distance.cosine(new_model_es[es_word], model_fr_ao[fr_word])
        distance_normal = scipy.spatial.distance.cosine(new_model_es[es_word], new_model_fr[fr_word])
        baseline = scipy.spatial.distance.cosine(new_model_fr[fr_word], model_fr_ao[fr_word])
        print(f"{es_word + ' - ' + fr_word:25} : {distance_aligned:.4f} vs {distance_normal:.4f} vs {baseline:.4f}")
        es_to_fr_aligned_cos.append(distance_aligned)
        es_to_fr_normal_cos.append(distance_normal)
        es_aligned_cos.append(baseline)

    print(f"combined distance aligned: {np.sum(es_to_fr_aligned_cos):.4f} / {len(es_to_fr_aligned_cos)} = {np.mean(es_to_fr_aligned_cos) * 100:.4f}%")
    print(f"combined distance normal: {np.sum(es_to_fr_normal_cos):.4f} / {len(es_to_fr_normal_cos)} = {np.mean(es_to_fr_normal_cos) * 100:.4f}%")
    print(f"unaligned v. aligned: {np.sum(es_aligned_cos):.4f} / {len(es_aligned_cos)} = {np.mean(es_aligned_cos) * 100:.4f}%")
    breakpoint()
    pass


if __name__ == '__main__':
    main()
    pass
