import functools
import itertools
import multiprocessing
from matplotlib.ticker import FuncFormatter

from matplotlib import pyplot as plt
import seaborn as sns

from combine_prealigned_test import *


def train_and_test_desc(all_sentences, all_paragraphs, ratio, latin_keyed_vectors, param_found, model_type=Word2Vec, lock_f_val=0, trial=0, **params):
    if np.any(((np.array([0.7000000000000001, 0.9]) + .00001) > lock_f_val) * (lock_f_val > (np.array([0.7000000000000001, 0.9]) - .00001))):
        print("shhh~~")
        lock_f_val += 0.0001
    # print(f"Running: {printable_params}")

    # make model from scratch
    model = model_type(**params, **param_found)

    # build the vocab
    model.build_vocab(all_sentences)
    model.init_weights()
    # print("inti")

    # set the vectors that we do have
    set_new_vectors(model, latin_keyed_vectors, lock_f_val, silent=True)
    # print("vectord")

    # split up the corpus
    train_paragraphs, _, _ = split_data(random.sample(all_paragraphs, len(all_paragraphs)), ratio, 1-ratio, 0)
    train_sentences = paragraphs_to_sentences(train_paragraphs)
    # print("corpd")

    # train
    model.train(
        train_sentences,
        total_examples=model.corpus_count,
        epochs=model.epochs
    )
    # print("trained")

    # test
    result: tuple = gensim_evaluations.methods.OddOneOut(
        cat_file='test_ooo_topk_la.txt',
        model=model.wv,
        k_in=3,
        allow_oov=True,
        sample_size=1000,
        silent=True,
        ft=model_type.__name__ == "FastText"
    )

    # we only want the overall score, not the individuals
    # print(f"Score desc {result[0]:.05f}")
    return result[0]


def train_and_test_norm(all_sentences, all_paragraphs, ratio, latin_keyed_vectors, param_found, model_type=Word2Vec, lock_f_val=0, trial=0, **params):
    # make model from scratch
    model = model_type(**params, **param_found)

    # build the vocab
    model.build_vocab(all_sentences)
    model.init_weights()
    # print("inttts -regulr")


    # split up the corpus
    train_paragraphs, _, _ = split_data(random.sample(all_paragraphs, len(all_paragraphs)), ratio, 1-ratio, 0)
    train_sentences = paragraphs_to_sentences(train_paragraphs)
    # print("copr -regulr")

    # train
    model.train(
        train_sentences,
        total_examples=model.corpus_count,
        epochs=model.epochs
    )
    # print("trained -regulr")

    # test
    result: tuple = gensim_evaluations.methods.OddOneOut(
        cat_file='test_ooo_topk_la.txt',
        model=model.wv,
        k_in=3,
        allow_oov=True,
        sample_size=1000,
        silent=True,
        ft=model_type.__name__ == "FastText"
    )

    # we only want the overall score, not the individuals
    # print(f"Score norm {result[0]:.05f}")
    return result[0]


def worker(run_func, args, kwargs, result_queue):
    try:
        result = run_func(*args, **kwargs)
        result_queue.put(result)
    except Exception as e:
        print(e)
        result_queue.put(-1)


def intermediary(run_func, skip_mp=False, *args, **kwargs):
    # using mp is for a shoddy thread crash protection, but it has a significant overhead
    if skip_mp:
        return run_func(*args, **kwargs)

    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=worker, args=(run_func, args, kwargs, result_queue))

    process.start()
    process.join()  # Wait for the process to finish

    if process.exitcode != 0:
        # Process crashed
        print(f"Crashed in {kwargs}, moving on")
        return -1
    else:
        # Process finished normally
        result = result_queue.get()
        return result


def plot_heatmaps(df):
    df = df.round({'ratio': 2, 'lock_f_val': 2})

    # Group by 'ratio' and 'lock_f_val' and calculate mean and standard deviation
    grouped = df.groupby(['ratio', 'lock_f_val']).agg(
        descendant_mean=('descendant', 'mean'),
        descendant_std=('descendant', 'std'),
        normal_mean=('normal', 'mean'),
        normal_std=('normal', 'std')
    ).reset_index()

    # Create pivot tables for heatmaps
    descendant_pivot = grouped.pivot(index="ratio", columns="lock_f_val", values="descendant_mean")
    normal_pivot = grouped.pivot(index="ratio", columns="lock_f_val", values="normal_mean")

    # Create text annotations
    descendant_text = grouped.pivot(index="ratio", columns="lock_f_val", values="descendant_std")
    normal_text = grouped.pivot(index="ratio", columns="lock_f_val", values="normal_std")

    plt.figure(figsize=(14, 6))

    # Descendant heatmap
    if "descendant" in df.columns:
        plt.subplot(1, 2, 1)
        ax1 = sns.heatmap(descendant_pivot, annot=False, cmap="rocket", cbar=True)
        for i in range(descendant_pivot.shape[0]):
            for j in range(descendant_pivot.shape[1]):
                mean_val = descendant_pivot.iloc[i, j]
                std_val = descendant_text.iloc[i, j]
                text = f'{mean_val:.2f}\n±{std_val:.2f}'
                ax1.text(j + 0.5, i + 0.5, text, ha='center', va='center', color='black')
        plt.title("Descendant Heatmap")
        plt.xlabel("lock_f_val")
        plt.ylabel("ratio")

    # Normal heatmap
    if "normal" in df.columns:
        plt.subplot(1, 2, 2)
        ax2 = sns.heatmap(normal_pivot, annot=False, cmap="rocket", cbar=True)
        for i in range(normal_pivot.shape[0]):
            for j in range(normal_pivot.shape[1]):
                mean_val = normal_pivot.iloc[i, j]
                std_val = normal_text.iloc[i, j]
                text = f'{mean_val:.2f}\n±{std_val:.2f}'
                ax2.text(j + 0.5, i + 0.5, text, ha='center', va='center', color='black')
        plt.title("Normal Heatmap")
        plt.xlabel("lock_f_val")
        plt.ylabel("ratio")

    plt.tight_layout()
    plt.show()


def interpolate(df, target, value_col='lock_f_val'):
    result = {}
    for col in df.columns:
        if col != value_col:
            val1 = df[df[value_col] == df[value_col].min()][col].values[0]
            val2 = df[df[value_col] == df[value_col].max()][col].values[0]
            x1 = df[value_col].min()
            x2 = df[value_col].max()
            interpolated_value = val1 + (val2 - val1) * (target - x1) / (x2 - x1)
            result[col] = interpolated_value
    return result


def grid_tests():
    # params that don't need optimization
    # todo: maybe use the gensim defaults instead of setting them
    param_found = {
        'vector_size': 300, 'workers': 8, 'seed': 42, "min_alpha": 0.0001, "alpha": 0.05, "min_count": 1.0, "negative": 10, "window": 5,
        # we might play around with epochs actually...
        'epochs': 20
    }
    # these are the params we want to explore
    # param_search = {
    #     # each tuple is: (start, stop, step)
    #     # "epochs": (20, 60, 1),
    #     "ratio": (.02, .2, .02),
    #     "lock_f_val": (0.639, 0.75, .1109),
    # }
    # param_search = {
    #     # each tuple is: (start, stop, step)
    #     # the stop and the start are inclusive, we do cool math later to do this.
    #     # "epochs": (20, 60, 1),  # resolution of 41 steps
    #     "ratio": (0.1, 0.5, 0.1),  # 5 steps
    #     "lock_f_val": (0, 1, 0.5),  # 3 steps
    #     # total res ~= 15
    # }

    # get the actual values with np.arange (actually linspace since its easier to get the edges),
    # expand them out with itertools.product to get all combos
    # package them back up with good ol' dict zip
    # params_expanded = [
    #     dict(zip(param_search.keys(), params))
    #     for params in itertools.product(*[
    #         np.linspace(start, stop, int((stop-start)/step)+1) for start, stop, step in param_search.values()
    #     ])
    # ]
    param_search = {
        "ratio": [0.02, 0.04, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2],
        "lock_f_val": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "trial": [1, 2, 3, 4, 5],
        # idk why but lock_f of 0.7 doesn't seem to work, only 0.639 and 0.75 work, anything between leads to a memory access violation error
        # idk why but lock_f of 0.9 doesn't seem to work, only 0.88 and 1.0 work, anything between leads to a memory access violation error
    }
    params_expanded = [dict(zip(param_search.keys(), params)) for params in itertools.product(*list(param_search.values()))]

    # set up the tests
    model_type = Word2Vec
    # model_type = FastText

    latin_daughter_vector_filepath = "prealigned/latin_daughter_vectors3.vec"
    latin_keyed_vectors = KeyedVectors.load(latin_daughter_vector_filepath)

    all_paragraphs, all_sentences, all_words = load_latin_corpus()

    # it's easier to deal with partially binding the function here
    random.seed(42)
    search_func_desc = functools.partial(train_and_test_desc, **{
        "all_sentences": all_sentences,
        "all_paragraphs": all_paragraphs,
        "latin_keyed_vectors": latin_keyed_vectors,
        "param_found": param_found,
        "model_type": model_type
    })
    random.seed(42)
    search_func_norm = functools.partial(train_and_test_norm, **{
        "all_sentences": all_sentences,
        "all_paragraphs": all_paragraphs,
        "latin_keyed_vectors": latin_keyed_vectors,
        "param_found": param_found,
        "model_type": model_type
    })

    # shuffle the ordering of parameters too, to make the time estimation more reliable
    random.shuffle(params_expanded)

    df = pd.read_csv("paper_results/grid_search_ft_5_trials_low_res_lock_both.csv")

    results = [
        {
            # as to not redo stuff we have already done, load it from file
            # "descendant": df2[
            #         ((df2["lock_f_val"] < params["lock_f_val"]+0.001) & (df2["lock_f_val"] > params["lock_f_val"]-0.001))
            #         & ((df2["ratio"] < params["ratio"]+0.001) & (df2["ratio"] > params["ratio"]-0.001))
            #     ].iloc[0]["descendant"],
            "descendant": intermediary(search_func_desc, **params),
            **params
        }
        for params in tqdm(params_expanded, ncols=160, desc="Running tests desc")
    ]
    results_normal = [
        {
            "normal": df[
                    ((df["lock_f_val"] < params["lock_f_val"]+0.001) & (df["lock_f_val"] > params["lock_f_val"]-0.001))
                    & ((df["ratio"] < params["ratio"]+0.001) & (df["ratio"] > params["ratio"]-0.001))
                    & (df["trial"] == params["trial"])
                ].iloc[0]["normal"],
            # "normal": intermediary(search_func_norm, **params),
            **params
        }
        for params in tqdm(params_expanded, ncols=160, desc="Running tests norm")
    ]
    results_combo = [{"normal": norm["normal"], **desc} for desc, norm in zip(results, results_normal)]

    # turn those into a dataframe
    df = pd.DataFrame(results_combo)

    df.to_csv("paper_results/grid_search_ft_5_trials_low_res_lock_vocab.csv", index=False)

    plot_heatmaps(df)

    pass


if __name__ == '__main__':
    # new_df = interpolate(pd.read_csv("paper_results/grid_search_ft_missing_weird1.csv"), 0.7)
    # breakpoint()
    grid_tests()
    # plot_heatmaps(pd.read_csv("paper_results/grid_search2.csv"))
    pass
