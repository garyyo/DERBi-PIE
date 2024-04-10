import os
from itertools import product
import gensim
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from alignment_code.unsup_multialign import multi_align_from_keyed_vectors
from closed_form import match_models


def calculate_translation_metrics(model1, model2, translation_dict):
    correct_translations = 0
    total_translations = 0

    all_y_true = []
    all_y_pred = []

    for word in model1.index_to_key:
        if word not in translation_dict or not any(translation in model2.key_to_index for translation in translation_dict[word]):
            continue

        test_vector = model1[word]
        true_translations = [word for word in translation_dict[word] if word in model2.index_to_key]
        most_similar_word = model2.most_similar(positive=[test_vector, ])[0][0]

        if most_similar_word in true_translations:
            correct_translations += 1

        total_translations += 1

        # For precision, recall, F1: considering each possible translation as a separate instance
        for possible_translation in true_translations:
            all_y_true.append(1)  # True translation
            all_y_pred.append(1 if possible_translation == most_similar_word else 0)

    accuracy = correct_translations / total_translations if total_translations > 0 else 0
    precision = precision_score(all_y_true, all_y_pred, zero_division=0)
    recall = recall_score(all_y_true, all_y_pred, zero_division=0)
    f1 = f1_score(all_y_true, all_y_pred, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def grid_search(unaligned_model1, unaligned_model2, translation_dict, align_func, config_dict, eval_func):
    best_score = -1
    best_params = {}
    all_scores = []

    # Generate all combinations of parameter values
    param_names = config_dict.keys()
    for values in product(*config_dict.values()):
        params = dict(zip(param_names, values))
        print(params)

        # Align the models using the current parameter combination
        aligned_model1, aligned_model2 = align_func(unaligned_model1, unaligned_model2, **params)

        # Evaluate the alignment
        metrics = eval_func(aligned_model1, aligned_model2, translation_dict)
        score = metrics['accuracy']  # You can choose to optimize on accuracy or any other metric

        # Update the best parameters if the current score is better
        if score > best_score:
            best_score = score
            best_params = params

        all_scores.append((params, metrics))
        print(f"Params: {params}, Metrics: {metrics}")  # Optional: Print current params and scores for monitoring

    return best_params, best_score, all_scores


def plot_heatmaps(all_scores, folder_path):
    assert all(len(params.keys()) == 2 for params, _ in all_scores), "Only two parameters should be changed at a time."

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Extract parameter names
    param_names = list(next(iter(all_scores))[0].keys())

    # Initialize a DataFrame for each metric
    metrics_dfs = {}

    for params, metrics in all_scores:
        for metric_name, metric_value in metrics.items():
            if metric_name not in metrics_dfs:
                metrics_dfs[metric_name] = pd.DataFrame(columns=param_names)
            # Add metric value to the DataFrame
            metrics_dfs[metric_name].loc[params[param_names[0]], params[param_names[1]]] = metric_value

    for metric_name, df in metrics_dfs.items():
        # Reset index for better heatmap plotting
        df.reset_index(inplace=True)
        df = df.pivot(index=param_names[0], columns=param_names[1], values=metric_name)

        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, fmt=".2f", cmap="rocket", cbar_kws={'label': metric_name}, vmin=0, vmax=1)
        plt.title(f'Heatmap of {metric_name} for Parameter Combinations')
        plt.ylabel(param_names[0])
        plt.xlabel(param_names[1])

        # Save the figure
        plt.savefig(os.path.join(folder_path, f'{metric_name}_heatmap.png'))
        plt.close()


def unsupervised_multi_alignment(unaligned_model1, unaligned_model2, **params):
    aligned_models = multi_align_from_keyed_vectors(
        [unaligned_model1, unaligned_model2],
        max_load=min(len(unaligned_model1.index_to_key), len(unaligned_model2.index_to_key)),
        **params
    )
    return aligned_models[0], aligned_models[1]


unsup_multialign_config = {
    "lr": [0.001, 0.01],
    "batch_size": [100, 200, 300, 500]
}


def main():
    # range of things to test

    # load models
    # model_fr: gensim.models.KeyedVectors = gensim.models.KeyedVectors.load("alignment/fr_bible_model.bin")
    # model_es: gensim.models.KeyedVectors = gensim.models.KeyedVectors.load("alignment/es_bible_model.bin")
    # newer models
    model_fr: gensim.models.KeyedVectors = gensim.models.KeyedVectors.load("wiki/fr_pared_keyed_vectors.kv")
    model_es: gensim.models.KeyedVectors = gensim.models.KeyedVectors.load("wiki/es_pared_keyed_vectors.kv")

    # use only a subset of the models that match words
    new_model_fr, new_model_es, french_to_spanish, spanish_to_french = match_models(model_fr, "fr", model_es, "es")

    # breakpoint()

    # a test call to the metric function
    # metric = calculate_translation_metrics(new_model_es, new_model_fr, spanish_to_french)

    # the eval function
    # anton: there are a number of possible metric
    #  1. word translation accuracy
    #   take vector of word from model1,
    #   find the word corresponding to the closest point in model2 to that vector,
    #   if that matches the known translation, +1 points, otherwise +0.
    #  2. Cross lingual word similarity
    #   This seems more difficult, I will try the other one first.
    eval_func = calculate_translation_metrics

    # load method
    method_func = unsupervised_multi_alignment

    config_dict = unsup_multialign_config
    best_params, best_score, all_scores = grid_search(new_model_es, new_model_fr, spanish_to_french, method_func, config_dict, eval_func)

    # graph metric
    plot_heatmaps(all_scores, "graphs/")

    pass


if __name__ == '__main__':
    main()
    pass
