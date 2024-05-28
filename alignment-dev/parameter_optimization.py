import functools

from combine_prealigned_test import *

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs


def plot_hyperparameter_scatter(filename):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load the JSON lines file
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # Convert data into a DataFrame
    df = pd.json_normalize(data)

    # List of parameter columns
    param_columns = [col for col in df.columns if col.startswith('params.')]

    # Create scatter plots
    plt.figure(figsize=(15, 10))
    for i, param in enumerate(param_columns):
        plt.subplot(3, 2, i + 1)
        sns.scatterplot(x=df[param], y=df['target'])
        plt.title(f'Scatter Plot of {param} vs Target')
        plt.xlabel(param)
        plt.ylabel('Target Score')

    plt.tight_layout()
    plt.show()


def optimize_descendant(all_sentences, train_sentences, int_params, latin_keyed_vectors, param_found, model_type=Word2Vec, lock_f_val=0, **w2v_params):
    print(f"Running: {lock_f_val=} | {w2v_params}")
    # some parameters need to be integers or other restrictions.
    w2v_params.update({param: int(w2v_params[param]) for param in int_params if param in w2v_params})
    print("updated param")

    # make model from scratch
    model = model_type(**w2v_params, **param_found)

    # build the vocab
    model.build_vocab(all_sentences)
    model.init_weights()

    # set the vectors that we do have
    set_new_vectors(model, latin_keyed_vectors, lock_f_val)

    # train
    model.train(
        train_sentences,
        total_examples=model.corpus_count,
        epochs=model.epochs
    )

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
    print(f"Score {result[0]:.05f}")
    return result[0]


def optimize_normal(all_sentences, train_sentences, int_params, param_found, model_type=Word2Vec, **w2v_params):
    print(f"Running: {w2v_params}")
    # some parameters need to be integers or other restrictions.
    w2v_params.update({param: int(w2v_params[param]) for param in int_params if param in w2v_params})

    # make model from scratch
    model = model_type(
        **w2v_params,
        **param_found
    )
    model.build_vocab(all_sentences)

    # train
    model.train(
        train_sentences,
        total_examples=model.corpus_count,
        epochs=model.epochs
    )

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
    print(f"Score: {result[0]:.05f}")
    return result[0]


def model_optimization():
    # params that don't need optimization
    param_found = {
        'vector_size': 300,
        'workers': 8,
        'seed': 42,
    }
    # params to optimize
    param_bounds = {
        'window': (3, 7),
        'negative': (5, 15),
        'alpha': (0.01, 0.05),
        'min_count': (1, 10),
        'epochs': (10, 200),
    }
    # int params need to be handled a bit more
    int_params = ["window", "min_count", "negative", "epochs"]

    # deal with training data
    tokenized_paragraphs, all_sentences, all_words = load_latin_corpus()

    random.seed(42)
    random.shuffle(tokenized_paragraphs)

    train_paragraphs, _, _ = split_data(tokenized_paragraphs, 1, 0, 0)
    train_sentences = paragraphs_to_sentences(train_paragraphs)
    print("corpus built")

    # test prep
    init_tests()

    # optimizer
    model_type = FastText

    # this is for the regular model
    # bound_func = optimize_normal
    # optimizer_func = functools.partial(bound_func, all_sentences, train_sentences, int_params, param_found, model_type)

    # this is only for the descendant model
    bound_func = optimize_descendant
    latin_daughter_vector_filepath = "prealigned/latin_daughter_vectors3.vec"
    latin_keyed_vectors = KeyedVectors.load(latin_daughter_vector_filepath)
    param_bounds["lock_f_val"] = (0, 1)
    optimizer_func = functools.partial(bound_func, all_sentences, train_sentences, int_params, latin_keyed_vectors, param_found, model_type)

    optimizer = BayesianOptimization(
        f=optimizer_func,
        pbounds=param_bounds,
        random_state=72,
    )

    # optimizer saving/loading
    log_attempt = 1
    model_method = "normal"
    logs_name = f"opt_logs/logs_{model_type.__name__}_{model_method}_{bound_func.__name__}_{log_attempt}"
    logs_file = f"{logs_name}.json"
    if os.path.exists(logs_file):
        print("loaded old logs")
        load_logs(optimizer, logs=[logs_file])
        pass
    logger = JSONLogger(path=logs_name)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    # run the optimizer
    print("optimizer built. Running...")
    optimizer.maximize(
        init_points=0,
        n_iter=30,
    )

    # do something with that data?
    plot_hyperparameter_scatter(logs_file)
    # ???
    pass


if __name__ == '__main__':
    model_optimization()
    # plot_hyperparameter_scatter("opt_logs/logs_FastText_normal_optimize_normal_0.json")
    pass
