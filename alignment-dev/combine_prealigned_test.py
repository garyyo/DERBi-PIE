import csv
import glob
import json
import os.path
import random
import shutil
import time
from collections import defaultdict

import cltk
import numpy as np
import pandas as pd
import spacy
from PyMultiDictionary import MultiDictionary
import gensim_evaluations.wikiqueries
import gensim_evaluations.methods
from deep_translator import GoogleTranslator
from gensim.models import KeyedVectors, Word2Vec, FastText
import regex as re
from nltk import sent_tokenize, RegexpTokenizer
from scipy.spatial.distance import cosine
from tqdm import tqdm


# region filter words
def is_valid_word(word):
    # This regex matches strings that consist only of letter characters (including those from non-Latin alphabets)
    # and mark characters for diacritics. It excludes numbers and special characters.
    pattern = re.compile(u'^[\p{L}\p{M}]+$')

    return bool(pattern.match(word))


def load_model_words(load_model=False):
    # load es
    if os.path.exists("prealigned/cached_steps/es_words.json") and not load_model:
        with open("prealigned/cached_steps/es_words.json", "r", encoding="utf-8") as fp:
            model_es_words = json.load(fp)
        model_es = None
    else:
        model_es = KeyedVectors.load_word2vec_format("prealigned/wiki.es.align.vec", binary=False)
        model_es_words = model_es.index_to_key
        with open("prealigned/cached_steps/es_words.json", "w", encoding="utf-8") as fp:
            json.dump(model_es_words, fp)

    # load fr
    if os.path.exists("prealigned/cached_steps/fr_words.json") and not load_model:
        with open("prealigned/cached_steps/fr_words.json", "r", encoding="utf-8") as fp:
            model_fr_words = json.load(fp)
        model_fr = None
    else:
        model_fr = KeyedVectors.load_word2vec_format("prealigned/wiki.fr.align.vec", binary=False)
        model_fr_words = model_fr.index_to_key
        with open("prealigned/cached_steps/fr_words.json", "w", encoding="utf-8") as fp:
            json.dump(model_fr_words, fp)

    return model_es, model_es_words, model_fr, model_fr_words


def load_lemmatized(valid_es_words, valid_fr_words):
    # Define file paths
    es_lemmatized_filepath = "prealigned/cached_steps/es_lemmatized.json"
    fr_lemmatized_filepath = "prealigned/cached_steps/fr_lemmatized.json"

    # Load or create Spanish lemmatized words
    if os.path.exists(es_lemmatized_filepath):
        with open(es_lemmatized_filepath, "r", encoding="utf-8") as file:
            es_lemmatized = json.load(file)
    else:
        nlp_es = spacy.load("es_core_news_sm")
        es_lemmatized = [nlp_es(word)[0].lemma_ for word in tqdm(valid_es_words)]
        with open(es_lemmatized_filepath, "w", encoding="utf-8") as file:
            json.dump(es_lemmatized, file)

    # Load or create French lemmatized words
    if os.path.exists(fr_lemmatized_filepath):
        with open(fr_lemmatized_filepath, "r", encoding="utf-8") as file:
            fr_lemmatized = json.load(file)
    else:
        nlp_fr = spacy.load("fr_core_news_sm")
        fr_lemmatized = [nlp_fr(word)[0].lemma_ for word in tqdm(valid_fr_words)]
        with open(fr_lemmatized_filepath, "w", encoding="utf-8") as file:
            json.dump(fr_lemmatized, file)

    return es_lemmatized, fr_lemmatized


def prune_words(lemmatized_words, nlp, language_label):
    # Prune non-valid language words using spaCy.
    pruned = [word for word in tqdm(lemmatized_words, desc=f"Purging non-{language_label} words", ncols=160) if any(token.is_alpha for token in nlp(word))]
    return pruned


def load_or_create_pruned(fr_lemmatized, es_lemmatized):
    # Define file paths
    fr_pruned_filepath = "prealigned/cached_steps/fr_pruned.json"
    es_pruned_filepath = "prealigned/cached_steps/es_pruned.json"

    # Load or create pruned French words
    if os.path.exists(fr_pruned_filepath):
        with open(fr_pruned_filepath, "r", encoding="utf-8") as file:
            fr_pruned = json.load(file)
    else:
        nlp_fr = spacy.load('fr_core_news_sm')
        fr_pruned = prune_words(fr_lemmatized, nlp_fr, "French")
        with open(fr_pruned_filepath, "w", encoding="utf-8") as file:
            json.dump(fr_pruned, file)

    # Load or create pruned Spanish words
    if os.path.exists(es_pruned_filepath):
        with open(es_pruned_filepath, "r", encoding="utf-8") as file:
            es_pruned = json.load(file)
    else:
        nlp_es = spacy.load('es_core_news_sm')
        es_pruned = prune_words(es_lemmatized, nlp_es, "Spanish")
        with open(es_pruned_filepath, "w", encoding="utf-8") as file:
            json.dump(es_pruned, file)

    return fr_pruned, es_pruned
# endregion


# region translation
def translate_batched(lang1_words, lang1_code, lang2_code, limit_groups=None, group_sizes=200):
    # the final translation is stored in a dict of lists, each list should be of size one, but just in case store in a list anyway
    lang1_to_lang2 = defaultdict(list)

    # turn into a set to remove duplicates, with the side effect of sets don't keep ordering
    # sorted the set to remove potential accidental context from the words, and get a consistent ordering for consistent testing and debugging.
    to_translate_words = sorted(set(lang1_words))

    # we split into groups because API calls are too big of an overhead, cuts the time down by like 200x
    to_translate_groups = [to_translate_words[i:i + group_sizes] for i in range(0, len(to_translate_words), group_sizes)]
    to_translate_groups = [group for i, group in enumerate(to_translate_groups) if limit_groups is None or i in limit_groups]

    # helpful progress bar
    for words in tqdm(to_translate_groups, desc=f"translating {lang1_code}->{lang2_code}", ncols=160):
        translated_words = translation_batched(words, lang1_code, lang2_code)

        # if we run into an error we subdivide and try again repeatedly until it works >:(
        if translated_words is None:
            translated_words = break_in_two_and_try_again(words, lang1_code, lang2_code)

        # associated word with translated word (each list entry should be size 1, but to not lose data we append to list instead of potentially overwrite)
        for word, translated_word in zip(words, translated_words):
            lang1_to_lang2[word].append(translated_word)
    return lang1_to_lang2


def translation_batched(words, lang1_code, lang2_code):
    # batch them together in a way that the translator doesn't accidentally try to get context from other words
    to_translate_phrase = "\"" + "\" | \"".join(words) + "\""

    # do the translation
    translated_phrase = try_translate(to_translate_phrase, lang1_code, lang2_code)

    if translated_phrase == "" or translated_phrase is None:
        return None

    # sometimes there are weird non-standard spaces, we pretend those are standard spaces (and replace them with 1 if there are multiples in a row)
    nonstandard_spaces = r'[\s\u00A0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000\u200B]+'
    translated_phrase = re.sub(nonstandard_spaces, ' ', translated_phrase)

    # un-batch them to get the actual meanings out, split on the | first since it's probably the least likely to appear in the language and cause an issue
    translated_words = re.split(r' \| ', translated_phrase[1:-2])

    # we have to invoke regex here to handle any sort of quotes not just standard double quotes, even though we only give standard double quotes
    translated_words = [re.sub(r'[“”"„«»❝❞]', '', word) for word in translated_words]

    # if the number of words returned to me is different from the number of words given then something went wrong,
    if len(words) != len(translated_words):
        return None
    return translated_words


def try_translate(word, lang_from, lang_to):
    try:
        word = GoogleTranslator(source=lang_from, target=lang_to).translate(word)
        return word
    except Exception as err:
        print(err)
        return None


def break_in_two_and_try_again(og_words, lang1_code, lang2_code):
    list_of_words = [og_words[:len(og_words)//2], og_words[len(og_words)//2:]]
    final_translated_words = []
    for words in list_of_words:
        if len(words) == 0:
            continue
        translated_words = translation_batched(words, lang1_code, lang2_code)
        if translated_words is None:
            final_translated_words += break_in_two_and_try_again(words, lang1_code, lang2_code)
        else:
            final_translated_words += translated_words
        pass
    return final_translated_words


def load_translations(fr_words, es_words):
    # Define file paths
    fr_to_es_filepath = "prealigned/cached_steps/fr_to_es.json"
    es_to_fr_filepath = "prealigned/cached_steps/es_to_fr.json"

    # Load or create French to Spanish translations

    # if it exists, load it because this is a really long process
    if os.path.exists(fr_to_es_filepath):
        with open(fr_to_es_filepath, "r", encoding="utf-8") as file:
            fr_to_es = json.load(file)
    # otherwise do the batch translating
    else:
        fr_to_es = translate_batched(fr_words, "fr", "es")

        # Save the result to disk
        with open(fr_to_es_filepath, "w", encoding="utf-8") as file:
            json.dump(fr_to_es, file)

    # Load or create Spanish to French translations
    if os.path.exists(es_to_fr_filepath):
        with open(es_to_fr_filepath, "r", encoding="utf-8") as file:
            es_to_fr = json.load(file)
    else:
        es_to_fr = translate_batched(es_words, "es", "fr")
        # Save the result
        with open(es_to_fr_filepath, "w", encoding="utf-8") as file:
            json.dump(es_to_fr, file)

    return fr_to_es, es_to_fr


def load_translations_latin(la_words):
    la_to_fr_filepath = "prealigned/cached_steps/la_to_fr2.json"
    la_to_es_filepath = "prealigned/cached_steps/la_to_es2.json"

    # if it exists, load it because this is a really long process
    if os.path.exists(la_to_fr_filepath):
        with open(la_to_fr_filepath, "r", encoding="utf-8") as file:
            la_to_fr = json.load(file)
    # otherwise do the batch translating
    else:
        la_to_fr = translate_batched(la_words, "la", "fr")

        # Save the result to disk
        with open(la_to_fr_filepath, "w", encoding="utf-8") as file:
            json.dump(la_to_fr, file)

    if os.path.exists(la_to_es_filepath):
        with open(la_to_es_filepath, "r", encoding="utf-8") as file:
            la_to_es = json.load(file)
    # otherwise do the batch translating
    else:
        la_to_es = translate_batched(la_words, "la", "es")

        # Save the result to disk
        with open(la_to_es_filepath, "w", encoding="utf-8") as file:
            json.dump(la_to_es, file)

    return la_to_fr, la_to_es
# endregion


# region lemmatization
# (actual lemmatization needs to be done outside of this script, sorry)
def is_valid_csv(csv_file, names=None):
    try:
        # Attempt to parse the CSV
        pd.read_csv(csv_file, names=names, encoding="ansi")
    except Exception as err:
        print(err)
        return False
    return True


def latin_vocab_list():
    col_names = ["original", "normalized", "lemma", "pos", "extra_info1", "extra_info2", "extra_info3"]
    lines = []
    err_counter = 0
    for file in tqdm(glob.glob("prealigned/latin_corpus_paragraphs2/*.csv"), ncols=160, desc="combining lemlat"):
        if not is_valid_csv(file, col_names):
            print(f"Skipping invalid CSV -> {os.path.basename(file)}")
            err_counter += 1
            continue
        # the program I use to lemmatize stupidly seems to save output as ansi
        with open(file, "r", encoding="ANSI") as fp:
            lines += fp.readlines() + ["\n"]
    print(f"{err_counter=}")
    combined_csv_path = "prealigned/lemmatized_bak2.csv"
    with open(combined_csv_path, "w", encoding="utf-8") as fp:
        fp.writelines(lines)
    latin_words_df = pd.read_csv(combined_csv_path, names=col_names)

    latin_words_df["original"] = latin_words_df["original"].str.lower()

    # LEMLAT replaces V with U, but Google Translate doesn't seem to like that
    # but it also doesn't quite like replacing ALL U's with V's
    # latin_words_df[["normalized", "lemma"]] = latin_words_df[["normalized", "lemma"]].apply(lambda col: col.str.replace("u", "v"))

    normalized_to_lemmas = latin_words_df.groupby('normalized')['lemma'].agg(lambda x: list(set(x))).to_dict()
    lemma_to_normalized = latin_words_df.groupby('lemma')['normalized'].agg(lambda x: list(set(x))).to_dict()

    original_to_lemmas = latin_words_df.groupby('original')['lemma'].agg(lambda x: list(set(x))).to_dict()
    lemma_to_original = latin_words_df.groupby('lemma')['original'].agg(lambda x: list(set(x))).to_dict()

    return normalized_to_lemmas, lemma_to_normalized, original_to_lemmas, lemma_to_original

# endregion


# region relating words
def relate_words(valid_fr_words, fr_lemmatized, fr_pruned, fr_to_es, valid_es_words, es_lemmatized, es_pruned, es_to_fr, la_to_es, la_to_fr):
    """
    This function will take a latin translation into french, then find if that translation exists in our model
        If it exists (or a lemmatized version exists that can be related to the model)
            it then tries to find the corresponding spanish words in the spanish model
            and relates them to the latin word
        It also adds the model version of the french word and relates it to the latin, if it exists.
    It also does the reverse, for spanish -> french
    Then it saves the french and spanish relations to latin, because this process is a little bit slow.
    """
    latin_relations_filepath = "prealigned/cached_steps/latin_relations2.json"

    # if it exists, load it because this is a really long process
    if os.path.exists(latin_relations_filepath):
        with open(latin_relations_filepath, "r", encoding="utf-8") as file:
            latin_to_model_words = json.load(file)
    # otherwise relate them
    else:
        latin_to_model_words = {}
        # fixme: when relating words like this, we should probably actually consider the a group of latin words as the same word.
        #  Then that group gets related to a group of es words, and a group of fr words.
        #  Then later on that entire group gets the same vector,
        for la_word in tqdm(la_to_fr, ncols=160, desc="relating words"):
            # load the french and spanish
            fr_word = la_to_fr[la_word][0]
            es_word = la_to_es[la_word][0]

            linked_es_words, model_fr_word = relate_word(fr_word, valid_fr_words, valid_es_words, fr_lemmatized, es_lemmatized, fr_pruned, es_pruned, fr_to_es)
            linked_fr_words, model_es_word = relate_word(es_word, valid_es_words, valid_fr_words, es_lemmatized, fr_lemmatized, es_pruned, fr_pruned, es_to_fr)

            # we also add the model word if it is found
            if model_es_word is not None:
                linked_es_words.add(model_es_word)
                if model_es_word not in valid_es_words:
                    breakpoint()
            if model_fr_word is not None:
                linked_fr_words.add(model_fr_word)
                if model_fr_word not in valid_fr_words:
                    breakpoint()

            latin_to_model_words[la_word] = {
                "fr": sorted(linked_fr_words),
                "es": sorted(linked_es_words),
            }
        with open(latin_relations_filepath, "w", encoding="utf-8") as file:
            json.dump(latin_to_model_words, file)
    return latin_to_model_words


def relate_word(lang1_word, valid_lang1_words, valid_lang2_words, lang1_lemmatized, lang2_lemmatized, lang1_pruned, lang2_pruned, lang1_to_lang2):
    lang2_words = set()

    # find where the word is in lang1
    model_lang1_word, lemmatized_lang1_word = find_model_lemma(lang1_word, valid_lang1_words, lang1_lemmatized, lang1_pruned)

    # and if that fails we give up on this word
    if model_lang1_word is None or lemmatized_lang1_word is None:
        return lang2_words, model_lang1_word

    # now get the corresponding lang2 word(s)
    translated_lang2_words = lang1_to_lang2[lemmatized_lang1_word]
    for translated_lang2_word in translated_lang2_words:
        # and do much the same process of seeing if they exist in the spanish model
        translated_model_es_word, translated_lemmatized_es_word = find_model_lemma(translated_lang2_word, valid_lang2_words, lang2_lemmatized, lang2_pruned)

        # and if we fail to find it, we just do not add it
        if translated_model_es_word is None or translated_lemmatized_es_word is None:
            continue

        lang2_words.add(translated_lang2_word)
    return lang2_words, model_lang1_word


def find_model_lemma(word, valid_words, lemmatized, pruned):
    model_word = None
    lemmatized_word = None
    # first check if it is in the model words itself
    # note that valid_words is a subset of model_words, so it's better to check the valid ones
    if word in valid_words:
        model_word = word
        lemmatized_word = lemmatized[valid_words.index(word)]
        # but it cant be pruned, otherwise we cant relate it to the other language
        lemmatized_word = lemmatized_word if lemmatized_word in pruned else None

    # if that fails try the lemmatized version (which can be related back to a model word)
    # note that pruned is a subset of lemmatized, so it's better to check the pruned ones
    elif word in pruned:
        model_word = valid_words[lemmatized.index(word)]
        lemmatized_word = word

    return model_word, lemmatized_word
# endregion


# region train/test
def load_latin_corpus():
    corpus_processed_path = "prealigned/cached_steps/corpus_processed.json"
    if not os.path.exists(corpus_processed_path):
        # build corpus as list of lists of words (list of sentences where each sentence is a list of words)
        latin_corpus = "prealigned/latin_corpus2.txt"
        with open(latin_corpus, "r", encoding="utf-8") as fp:
            paragraphs = " ".join(fp.readlines()).split("\n \n")

        # split paragraphs into sentences, split sentences into words
        tokenized_paragraphs = [tokenize_corpus(paragraph) for paragraph in paragraphs]
        all_sentences = paragraphs_to_sentences(tokenized_paragraphs)
        all_words = [word for sentence in all_sentences for word in sentence]
        with open(corpus_processed_path, "w", encoding="utf-8") as fp:
            json.dump({"tokenized_paragraphs": tokenized_paragraphs, "all_sentences": all_sentences, "all_words": all_words}, fp)
        print("words tokenized")
    else:
        with open(corpus_processed_path, "r", encoding="utf-8") as fp:
            obj = json.load(fp)
        tokenized_paragraphs, all_sentences, all_words = obj["tokenized_paragraphs"], obj["all_sentences"], obj["all_words"]

    return tokenized_paragraphs, all_sentences, all_words


def tokenize_corpus(text):
    # split into sentences
    # the text often has newlines to keep a consistent line length, and not to just denote paragraphs, so we replace newline with spaces
    # and remove any errant carriage returns.
    sentences = sent_tokenize(text.replace("\n", " ").replace("\r", "").lower())
    # since this is not a NLTK-able language, we use a simple regexp tokenizer
    tokenized_sentences = [
        # filter out things that are clearly not words
        [word for word in RegexpTokenizer(r"\w+").tokenize(sentence) if is_valid_word(word)]
        for sentence in sentences
    ]
    # remove empty sequences
    tokenized_sentences = [sentence for sentence in tokenized_sentences if len(sentence) != 0]
    return tokenized_sentences


def split_data(data, train_ratio=0.8, test_ratio=0.2, val_ratio=0):
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio != 1.0:
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
        print(f"Warning: Ratios did not sum to 1. Normalized to {train_ratio=}, {test_ratio=}, {val_ratio=}")

    n = len(data)

    # Compute split indices
    train_end = int(train_ratio * n)
    val_end = int(val_ratio * n) + train_end

    # Split data
    train_set = data[:train_end]
    val_set = data[train_end:val_end]
    test_set = data[val_end:]

    return train_set, val_set, test_set


def paragraphs_to_sentences(paragraphs):
    return [
        sentence
        for paragraph in paragraphs
        for sentence in paragraph
        if len(sentence)
    ]


def init_tests():
    categories = [
        'Q9415',  # emotion
        'Q60539481',  # negative emotion
        'Q4271324',  # mythical character
        'Q6256',  # country
        'Q515',  # city
        # these don't do anything actually
        # 'Q397',  # Latin
        # 'Q7198',  # Ovid
        # 'Q1747689'  # Ancient Rome
    ]
    langs = ['la']
    test_filename = "test_ooo_topk"
    gensim_evaluations.wikiqueries.generate_test_set(items=categories, languages=langs, filename=test_filename)
    return f'{test_filename}_{langs[0]}.txt'


def set_new_vectors(model, latin_keyed_vectors, lock_f_val, silent=False):
    # set the vectors that we do have
    new_keys = set(latin_keyed_vectors.index_to_key).intersection(set(model.wv.index_to_key))
    new_keys = [key for key in latin_keyed_vectors.index_to_key if key in new_keys]
    new_vectors = [latin_keyed_vectors.vectors[latin_keyed_vectors.key_to_index[key]] for key in new_keys]
    skipped_keys = list(set(latin_keyed_vectors.index_to_key) - set(model.wv.index_to_key))

    # skipped_vectors = [latin_keyed_vectors.vectors[latin_keyed_vectors.key_to_index[key]] for key in skipped_keys]
    missing_keys = list(set(model.wv.index_to_key) - set(latin_keyed_vectors.index_to_key))
    num_used = len(new_keys)
    num_total = len(latin_keyed_vectors.index_to_key)
    num_vocab = len(model.wv.index_to_key)
    num_skipped = len(skipped_keys)
    num_missing = len(missing_keys)

    if not silent:
        print(f"Reusing {num_used} of {num_total} latin generated vectors ({num_used / num_total * 100:.2f}%), skipped {num_skipped} vectors")
        print(f"Vectors Cover {num_used} of {num_vocab} vocab ({num_used / num_vocab * 100:.2f}%), {num_missing} vectors are set to random")
        print(f"Skipped key: {skipped_keys[:20]}{'...' if len(skipped_keys)> 20 else ''}, # = {len(skipped_keys)}")

    model.wv[new_keys] = new_vectors

    # lock the vectors that we set because we don't want them to move around.
    lock_f = np.ones([model.wv.vectors.shape[0]])
    lock_f[[model.wv.key_to_index[key] for key in new_keys]] = lock_f_val
    if model.__class__.__name__ == "FastText":
        model.wv.vectors_vocab_lockf = lock_f
        model.wv.vectors_ngrams_lockf = lock_f
    else:
        model.wv.vectors_lockf = lock_f

    if not silent:
        print("vectors set")
    pass


def test_models(model_wv, model_blind_wv, ft=True):
    # using OddOneOut and Topk from gensim_evaluations
    cat_file = 'test_ooo_topk_la.txt'
    scores: dict[tuple] = {
        'ooo_descendant': run_test(gensim_evaluations.methods.OddOneOut, cat_file, model_wv, k_in=3, allow_oov=True, sample_size=1000, ft=ft, silent=True),
        'ooo_blind': run_test(gensim_evaluations.methods.OddOneOut, cat_file, model_blind_wv, k_in=3, allow_oov=True, sample_size=1000, ft=ft, silent=True),
        # 'topk_descendant': run_test(gensim_evaluations.methods.Topk, cat_file, model_wv, k=3, allow_oov=True, ft=ft),
        # 'topk_blind': run_test(gensim_evaluations.methods.Topk, cat_file, model_blind_wv, k=3, allow_oov=True, ft=ft),
    }

    print_scores(scores)


def run_test(method, cat_file, model_wv, **method_kwargs):
    result = method(cat_file=cat_file, model=model_wv, **method_kwargs)
    # cuts out some annoying text from labels, and puts the "overall" label in the same dict for easy parsing
    formatted_result = {**{instance_of[13:-5]: score for instance_of, score in result[1].items()}, **{"overall": result[0]}}
    return formatted_result


def print_scores(scores):
    score_types = [" "] + list(list(scores.values())[0].keys())
    max_col_width = max(len(label) for label in score_types)
    max_col_width = max_col_width + max_col_width % 2
    print(*[f"{label:^{max_col_width}}" for label in score_types], "", sep=" | ")
    for test, test_scores in scores.items():
        print(f"{test:>{max_col_width}} |", end="")
        for category, score in test_scores.items():
            print(f"{score:^{max_col_width + 3}.4f}", end="")
        print("   ")


def analogy_accuracy(wv, analogy_tests, top_n=5):
    correct_top1 = 0
    correct_topn = 0
    distances = []
    closest = []

    for A, B, C, expected in analogy_tests:
        # Compute vector D such that D = B - A + C
        vector_D = wv[B] - wv[A] + wv[C]

        # Get the most similar words to vector D
        most_similar = wv.similar_by_vector(vector_D, topn=top_n)
        most_similar_words = [word for word, _ in most_similar]

        # Check if the expected word is in the top 1 and top n results
        if expected in most_similar_words[:1]:
            correct_top1 += 1
        if expected in most_similar_words[:top_n]:
            correct_topn += 1

        # Compute the distance between vector D and the true value of the vector for the expected word
        distance = cosine(vector_D, wv[expected])
        distances.append(distance)

        closest.append(most_similar_words[:1])

    # Calculate accuracy
    top1_accuracy = correct_top1 / len(analogy_tests)
    topn_accuracy = correct_topn / len(analogy_tests)

    return {
        "top1_accuracy": top1_accuracy,
        "topn_accuracy": topn_accuracy,
        "distances": distances,
        "closest": closest,
    }

# endregion


# essentially, main part 1
def generate_latin_daughter_vectors():
    # load models (or just the words since we are not making use of the models yet). model words are specifically words as they exist in the model
    model_es, model_es_words, model_fr, model_fr_words = load_model_words(load_model=False)
    print("loaded words")

    # 1. filter out model words that contain non-word characters
    valid_es_words = [word for word in tqdm(model_es_words, desc="filtering spanish", ncols=160) if is_valid_word(word)]
    valid_fr_words = [word for word in tqdm(model_fr_words, desc="filtering french", ncols=160) if is_valid_word(word)]
    print("filtered out non-words")

    # (alt) turn each model word into some base lemmatized form, and (todo) try to filter out proper nouns.
    #    then do the above with the lemmatized versions and just consider all lemmatized versions of that word
    es_lemmatized, fr_lemmatized = load_lemmatized(valid_es_words, valid_fr_words)
    print("lemmatized words")
    print(f"Spanish Reduction: {100 * (1 - len(set(es_lemmatized)) / len(valid_es_words)):.2f}%")
    print(f"French Reduction: {100 * (1 - len(set(fr_lemmatized)) / len(valid_fr_words)):.2f}%")

    # (another) remove words that are not actually french and spanish? This doesn't work as well as I want it to
    fr_pruned, es_pruned = load_or_create_pruned(fr_lemmatized, es_lemmatized)
    print("pruned out-of-language words")
    print(f"Spanish Reduction: {100 * (1 - len(set(es_pruned)) / len(set(es_lemmatized))):.2f}%")
    print(f"French Reduction: {100 * (1 - len(set(fr_pruned)) / len(set(fr_lemmatized))):.2f}%")

    # 2. for each word left, translate it into the other language to associate them together
    # find translation of (french -> spanish) and (spanish -> french)
    fr_to_es, es_to_fr = load_translations(fr_pruned, es_pruned)

    # 3. for each word in latin corpus, lemmatize it, translate to French and Spanish, and relate it to the existing french and spanish model words.
    # load the lemmatized latin (some of the parts are processed outside of this script)
    latin_normalized_to_lemmas, latin_lemma_to_normalized, latin_original_to_lemmas, latin_lemma_to_original = latin_vocab_list()
    # lemma is not enough, we probably should translate the originals too since the LEMLAT does weird things
    latin_translate_to_original = defaultdict(set)
    for lemma, originals in latin_lemma_to_original.items():
        for original in originals:
            latin_translate_to_original[original].add(original)
        latin_translate_to_original[lemma] = latin_translate_to_original[lemma].union(set(originals))

    # isolate the latin lemmas for translation
    # fixme: remove lemmas with symbols in them since we can't handle them right now, but eventually it would be better to keep those
    latin_lemma_words = [lemma for lemma in latin_lemma_to_normalized.keys() if not re.search(r'["/\\\'()\[\]{}\-+]', lemma)]
    # fixme: the same for original words with symbols. They might be actual words but I have no way of handling them right now.
    latin_translate_words = latin_lemma_words + [original for original in latin_original_to_lemmas.keys() if not re.search(r'["/\\\'()\[\]{}\-+]', original)]
    # remove duplicates, sort for consistent ordering to help debugging and to turn back into a list
    latin_translate_words = sorted(set(latin_translate_words))

    # translate from latin to french and spanish, this will help associate with each other (latin <--> french <--> spanish)
    la_to_fr, la_to_es = load_translations_latin(latin_translate_words)
    # relate the latin words to an existing spanish or french word
    latin_to_model_words = relate_words(valid_fr_words, fr_lemmatized, fr_pruned, fr_to_es,
                                        valid_es_words, es_lemmatized, es_pruned, es_to_fr,
                                        la_to_es, la_to_fr)
    # filter out latin words that don't have at least 1 corresponding spanish or french model word
    latin_to_model_words = {
        latin: model_words
        for latin, model_words in latin_to_model_words.items()
        if len(model_words["fr"]) > 0 and len(model_words["es"]) > 0
    }
    print(
        f"{len(latin_to_model_words)} of {len(la_to_fr)} latin words remaining from translation (which expands # of words). "
        f"Lost {(1 - len(latin_to_model_words) / len(la_to_fr)) * 100:.2f}%"
    )
    # 4. gather the lemmatized French and Spanish word vectors, average the groups together (optionally check if they are in a cluster of some sort),
    #    and use that as the starting value for the vector for Latin (perhaps change the method of training later to not allow for these values to change?)
    # ok now we need the actual models
    print("Loading the actual models (this may take a couple minutes)", end="")
    start_time = time.time()
    model_es, model_es_words, model_fr, model_fr_words = load_model_words(load_model=True)
    end_time = time.time()
    print(f". Yeah so it turns out it took {(end_time - start_time)/60:.02f} minutes")

    latin_vectors = {}
    for i, (latin, model_words) in enumerate(tqdm(latin_to_model_words.items(), ncols=160, desc="assigning vectors")):
        # some words are missing from the model but are in the
        found_fr_words = model_words["fr"]
        missing_model_words_fr = [word for word in found_fr_words if word not in model_fr.index_to_key]
        if len(missing_model_words_fr) > 0:
            print(f"Words are missing in the french model: {missing_model_words_fr} (despite allegedly coming from said model)")
        if len(missing_model_words_fr) == len(found_fr_words):
            print("All words missing from model, skipping entry")
            continue

        found_es_words = model_words["es"]
        missing_model_words_es = [word for word in found_es_words if word not in model_es.index_to_key]
        if len(missing_model_words_es) > 0:
            print(f"Words are missing in the spanish model: {missing_model_words_es} (despite allegedly coming from said model)")
        # weirdly enough, sometimes spanish words are in the model word list, but not in the model vector list.
        if len(missing_model_words_es) == len(found_es_words):
            print("All words missing from model, skipping entry")
            continue

        # calculate a centroid of the vectors given and then throw out any that are too far away from that center.
        # todo: use some method to figure out which are truly outliers, currently just using the closest to the center of the vectors.
        # find the valid words
        valid_fr = [model_fr.get_vector(word) for word in found_fr_words if word in model_fr.index_to_key]
        valid_es = [model_es.get_vector(word) for word in found_es_words if word in model_es.index_to_key]
        # find the center of all words
        centroid = np.mean([np.mean(valid_fr, axis=0), np.mean(valid_es, axis=0)], axis=0)
        # for each language find the distance of the vectors that center
        scored_fr = [cosine(vec, centroid) for vec in valid_fr]
        scored_es = [cosine(vec, centroid) for vec in valid_es]
        # only use the closest vector to the center of the valid points
        la_vector = np.mean([valid_fr[np.argmin(scored_fr)], valid_es[np.argmin(scored_es)]], axis=0)

        # the old way, does not filter for potentially outlier vectors
        # # get the center of french words
        # fr_vector = np.mean(valid_fr, axis=0)
        # # get the center of spanish words
        # es_vector = np.mean(valid_es, axis=0)
        # # get the center between the two
        # la_vector = np.mean([fr_vector, es_vector], axis=0)

        # turn the latin lemmas back into in-vocab latin words
        # todo: latin_translate_to_original technically probably has original words that have more than one translate.
        #  Thus we should be building out those groups first, then taking the average vector.
        #  Ideally these vectors would be similar to each other so doing that would be ok.
        for latin_original_word in latin_translate_to_original[latin]:
            latin_vectors[latin_original_word] = la_vector
        pass
    latin_keyed_vectors = KeyedVectors(vector_size=len(next(iter(latin_vectors.values()))))
    latin_keyed_vectors.add_vectors(list(latin_vectors.keys()), list(latin_vectors.values()))

    print(f"{len(latin_keyed_vectors.index_to_key)} of {len(latin_original_to_lemmas)} latin words have vectors. {(len(latin_keyed_vectors.index_to_key)/len(latin_original_to_lemmas.keys()))*100:.2f}% of words have a vector.")

    return latin_keyed_vectors


def main(force_regenerate=False):
    """
    The full process:
    1-4: generate latin descendant vectors
    5.a: initialize descendant model
    5.b: load descendant vectors into the model
    5.c: train the model
    5.d: test the model
    6:   the same as 5 but with a blind model (init, train, test)
    """
    # 1-4: create the latin daughter vectors
    latin_daughter_vector_filepath = "prealigned/latin_daughter_vectors3.vec"
    if os.path.exists(latin_daughter_vector_filepath) and not force_regenerate:
        latin_keyed_vectors = KeyedVectors.load(latin_daughter_vector_filepath)
    else:
        latin_keyed_vectors = generate_latin_daughter_vectors()
        latin_keyed_vectors.save(latin_daughter_vector_filepath)

    # 5. train a latin model on the corpus using these vectors as initial values.
    # initialize model
    # model_type = Word2Vec
    model_type = FastText
    w2v_params = {
        "vector_size": latin_keyed_vectors.vector_size,
        # "min_count": 1,
        # "window": 5,
        # "epochs": 100,
        "seed": 42,
        # default = 0.025
        # "alpha": 0.045,
        # default = 0.0001
        "min_alpha": 0.0001,
        "workers": 8,
        # below are the optimized values
        # "alpha": 0.05, "epochs": 60, "min_count": 1.0, "negative": 15.0, "window": 3.0,
        # these are the values used by facebook in their original training
        # "alpha": 0.05, "epochs": 60, "min_count": 1.0, "negative": 10, "window": 5, "min_n": 5, "max_n": 5
        # todo: remove this
        "alpha": 0.05, "epochs": 20, "min_count": 1.0, "negative": 10, "window": 5
    }
    model = model_type(**w2v_params)

    all_paragraphs, all_sentences, all_words = load_latin_corpus()

    # build the vocab
    model.build_vocab(all_sentences)
    model.init_weights()
    print("vocab built")

    # set vectors (and lock them maybe)
    set_new_vectors(model, latin_keyed_vectors, 1)

    # randomly shuffle (deterministically)
    random.seed(42)
    random.shuffle(all_paragraphs)

    # separate into test-train-validate sets on paragraph (test/validation sets are being ignored for now since I can't use them)
    train_paragraphs, _, _ = split_data(all_paragraphs, 1, 19, 0)
    train_sentences = paragraphs_to_sentences(train_paragraphs)

    model.train(train_sentences, total_examples=model.corpus_count, epochs=model.epochs)
    print("model trained")

    model.save("latin_models/model_descendant.bin")

    # 6. train another latin model on the corpus without these vectors.
    # init model. Making sure to keep the parameters the same between the two
    model_blind = model_type(**w2v_params)

    model_blind.build_vocab(all_sentences)
    print("vocab built - blind model")

    model_blind.train(train_sentences, total_examples=model.corpus_count, epochs=model.epochs)
    print("model trained - blind model")

    model_blind.save("latin_models/model_blind.bin")

    # 7. (maybe?) train another latin model on a much larger corpus.
    # todo

    # run tests
    # 1. Run latin based analogy tests
    # todo

    test_models(model.wv, model_blind.wv, ft=model_type.__name__ == "FastText")

    # breakpoint()
    pass


if __name__ == '__main__':
    main()
    # test_models(Word2Vec.load("latin_models/model_descendant.bin").wv, Word2Vec.load("latin_models/model_blind.bin").wv)
    pass
