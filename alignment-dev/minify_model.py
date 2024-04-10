import pandas as pd
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from tqdm import tqdm

for lang in tqdm(["es", "fr", "it"]):
    # Load the Word2Vec model
    model = Word2Vec.load(f"wiki/{lang}wiki")

    words = model.wv.key_to_index
    counts = [model.wv.vocab[word].count for word in words]

    # models are big, this might help. calling the gc explicitly also might help but for now just this.
    del model

    # Load the vocabulary from the CSV file
    vocab_df = pd.DataFrame({
        'word': words,
        'count': counts
    })
    vocab_df.to_csv(f"wiki/{lang}_wiki_vocab.tsv", encoding="utf-8", sep="\t", index=False)

    # Filter out words that are not used enough, at empty string (which is used a lot for some reason?)
    filtered_vocab_bool = (vocab_df['count'] > 1) & (vocab_df["word"] != "")
    filtered_vocab_df = vocab_df[filtered_vocab_bool]
    removed_words_df = vocab_df[~filtered_vocab_bool]

    # Save the filtered vocabulary and the removed entries to files
    filtered_vocab_df.to_csv(f"wiki/{lang}_wiki_vocab_pared.txt", index=False)

    # Load the associated vectors
    vectors = np.load(f"wiki/{lang}wiki.wv.vectors_vocab.npy")

    # Remove the rows from vectors that correspond to removed words
    # This assumes that the rows in the vectors file are in the same order as the vocabulary
    words_to_remove_indices = removed_words_df.index.tolist()
    pared_vectors = np.delete(vectors, words_to_remove_indices, axis=0)

    # Create a KeyedVectors instance with the pared-down list of vocab and vectors
    kv_model = KeyedVectors(pared_vectors.shape[1])
    pared_keys = filtered_vocab_df['word'].tolist()
    kv_model.add_vectors(pared_keys, pared_vectors)

    # Save the new KeyedVectors model
    kv_model.save(f"wiki/{lang}_pared_keyed_vectors.kv")