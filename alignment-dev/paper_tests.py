import functools
import itertools
import multiprocessing
from matplotlib.ticker import FuncFormatter

from matplotlib import pyplot as plt
import seaborn as sns

from combine_prealigned_test import *


def train_and_test_desc(all_sentences, all_paragraphs, ratio, latin_keyed_vectors, param_found, model_type=Word2Vec, lock_f_val=0, **params):
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
    train_paragraphs, _, _ = split_data(all_paragraphs, ratio, 1-ratio, 0)
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


def train_and_test_norm(all_sentences, all_paragraphs, ratio, latin_keyed_vectors, param_found, model_type=Word2Vec, lock_f_val=0, **params):
    # make model from scratch
    model = model_type(**params, **param_found)

    # build the vocab
    model.build_vocab(all_sentences)
    model.init_weights()
    # print("inttts -regulr")

    # split up the corpus
    train_paragraphs, _, _ = split_data(all_paragraphs, ratio, 1-ratio, 0)
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
        result_queue.put(-1)


def intermediary(run_func, *args, **kwargs):
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


def grid_tests():
    # params that don't need optimization
    # todo: maybe use the gensim defaults instead of setting them
    param_found = {
        'vector_size': 300, 'workers': 8, 'seed': 42, "min_alpha": 0.0001, "alpha": 0.05, "min_count": 1.0, "negative": 10, "window": 5,
        # we might play around with epochs actually...
        'epochs': 20
    }
    # these are the params we want to explore
    param_search = {
        # each tuple is: (start, stop, step)
        # "epochs": (20, 60, 1),
        "ratio": (.02, .2, .02),
        "lock_f_val": (.7, .9, .2),
    }
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
    params_expanded = [
        dict(zip(param_search.keys(), params))
        for params in itertools.product(*[
            np.linspace(start, stop, int((stop-start)/step)+1) for start, stop, step in param_search.values()
        ])
    ]

    # set up the tests
    # model_type = Word2Vec
    model_type = FastText

    latin_daughter_vector_filepath = "prealigned/latin_daughter_vectors3.vec"
    latin_keyed_vectors = KeyedVectors.load(latin_daughter_vector_filepath)

    all_paragraphs, all_sentences, all_words = load_latin_corpus()

    random.seed(42)
    random.shuffle(all_paragraphs)

    # it's easier to deal with partially binding the function here
    search_func_desc = functools.partial(train_and_test_desc, **{
        "all_sentences": all_sentences,
        "all_paragraphs": all_paragraphs,
        "latin_keyed_vectors": latin_keyed_vectors,
        "param_found": param_found,
        "model_type": model_type
    })
    search_func_norm = functools.partial(train_and_test_norm, **{
        "all_sentences": all_sentences,
        "all_paragraphs": all_paragraphs,
        "latin_keyed_vectors": latin_keyed_vectors,
        "param_found": param_found,
        "model_type": model_type
    })

    results = [
        {
            "normal": intermediary(search_func_norm, **params),
            "descendant": intermediary(search_func_desc, **params),
            **params
        }
        for params in tqdm(params_expanded, ncols=160, desc="Running tests")
    ]

    # turn those into a dataframe
    df = pd.DataFrame(results)

    df.to_csv("paper_results/grid_search_ft_missing.csv", index=False)

    plot_heatmaps(df)

    pass


def plot_heatmaps(df):
    df = df.round({'ratio': 2, 'lock_f_val': 2})

    # Create pivot tables for heatmaps with keyword-only arguments to avoid warnings
    descendant_pivot = df.pivot(index="ratio", columns="lock_f_val", values="descendant")
    normal_pivot = df.pivot(index="ratio", columns="lock_f_val", values="normal")

    # Plot heatmaps
    plt.figure(figsize=(14, 6))

    # Descendant heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(descendant_pivot, annot=True, cmap="rocket", cbar=True)
    plt.title("Descendant Heatmap")
    plt.xlabel("lock_f_val")
    plt.ylabel("ratio")

    # Normal heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(normal_pivot, annot=True, cmap="rocket", cbar=True)
    plt.title("Normal Heatmap")
    plt.xlabel("lock_f_val")
    plt.ylabel("ratio")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    grid_tests()
    # plot_heatmaps(pd.read_csv("paper_results/grid_search2.csv"))
    pass
