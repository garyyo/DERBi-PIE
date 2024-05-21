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
        # this is only for the descendant model
        'lock_f_val': (0, 1),
    }
    # int params need to be handled a bit more
    int_params = ["window", "min_count", "negative", "epochs"]

    # deal with training data
    tokenized_paragraphs, all_sentences, all_words = load_latin_corpus()

    random.seed(42)
    random.shuffle(tokenized_paragraphs)
    train_paragraphs, test_paragraphs, validate_paragraphs = split_data(tokenized_paragraphs, 1, 0, 0)
    train_sentences = paragraphs_to_sentences(train_paragraphs)
    print("corpus built")

    # test prep
    categories = [
        'Q9415',  # emotion
        'Q60539481',  # negative emotion
        'Q4271324',  # mythical character
        'Q6256',  # country
        'Q515',  # city
    ]
    langs = ['la']
    gensim_evaluations.wikiqueries.generate_test_set(items=categories, languages=langs, filename='test_ooo_topk')
    print("tests built")

    # def optimize_train_and_test(**w2v_params):
    #     # some parameters need to be integers or other restrictions.
    #     w2v_params.update({param: int(w2v_params[param]) for param in int_params if param in w2v_params})
    #
    #     # make model from scratch
    #     model = Word2Vec(
    #         **w2v_params,
    #         **param_found
    #     )
    #     model.build_vocab(all_sentences)
    #
    #     # train
    #     model.train(
    #         train_sentences,
    #         total_examples=model.corpus_count,
    #         epochs=model.epochs
    #     )
    #
    #     # test
    #     result: tuple = gensim_evaluations.methods.OddOneOut(
    #         cat_file='test_ooo_topk_la.txt',
    #         model=model.wv,
    #         k_in=3,
    #         allow_oov=True,
    #         sample_size=1000,
    #         silent=True
    #     )
    #
    #     # we only want the overall score, not the individuals
    #     return result[0]

    def optimize_train_and_test_desc(lock_f_val=0, **w2v_params):
        # some parameters need to be integers or other restrictions.
        w2v_params.update({param: int(w2v_params[param]) for param in int_params if param in w2v_params})
        print("updated param")

        # make model from scratch
        model = Word2Vec(**w2v_params, **param_found)

        # build the vocab
        model.build_vocab(all_sentences)
        model.init_weights()
        print("vocab built")

        # set the vectors that we do have
        latin_daughter_vector_filepath = "prealigned/latin_daughter_vectors2.vec"
        latin_keyed_vectors = KeyedVectors.load(latin_daughter_vector_filepath)
        new_keys = set(latin_keyed_vectors.index_to_key).intersection(set(model.wv.index_to_key))
        new_keys = [key for key in latin_keyed_vectors.index_to_key if key in new_keys]
        new_vectors = [latin_keyed_vectors.vectors[latin_keyed_vectors.key_to_index[key]] for key in new_keys]

        skipped_keys = list(set(all_words) - set(new_keys))
        # skipped_vectors = [latin_keyed_vectors.vectors[latin_keyed_vectors.key_to_index[key]] for key in skipped_keys]

        num_used = len(new_keys)
        num_total = len(latin_keyed_vectors.index_to_key)
        num_vocab = len(set(model.wv.index_to_key))
        num_skipped = num_total - num_used

        print(f"Using {num_used} of {num_total} vectors ({num_used/num_total * 100:.2f}%), skipped {num_skipped}")
        print(f"Covers {num_used} of {num_vocab} vocab ({num_used/num_vocab * 100:.2f}%)")
        print(f"Skipped key: # = {len(skipped_keys)}")
        model.wv[new_keys] = new_vectors

        # lock the vectors that we set because we don't want them to move around.
        lock_f = np.ones([model.wv.vectors.shape[0]])
        lock_f[[model.wv.key_to_index[key] for key in new_keys]] = lock_f_val
        model.wv.vectors_lockf = lock_f
        print("vectors set")

        # randomly shuffle (deterministically)
        random.seed(42)
        random.shuffle(tokenized_paragraphs)
        # separate into test-train-validate sets on paragraph (validation set is being ignored for now)
        train_paragraphs, test_paragraphs, validate_paragraphs = split_data(tokenized_paragraphs, 1, 0, 0)

        train_sentences = paragraphs_to_sentences(train_paragraphs)
        # test_sentences = paragraphs_to_sentences(test_paragraphs)

        model.train(train_sentences, total_examples=model.corpus_count, epochs=model.epochs)
        print("model trained")
        print("trained")

        # test
        result: tuple = gensim_evaluations.methods.OddOneOut(
            cat_file='test_ooo_topk_la.txt',
            model=model.wv,
            k_in=3,
            allow_oov=True,
            sample_size=1000,
            silent=True
        )

        # we only want the overall score, not the individuals
        return result[0]

    # optimizer saving/loading
    optimizer = BayesianOptimization(
        f=optimize_train_and_test_desc,
        pbounds=param_bounds,
        random_state=12,
        verbose=2,
    )
    log_attempt = 0
    logs_path = f"opt_logs/logs_{log_attempt}_desc.log.json"
    if os.path.exists(logs_path):
        print("loaded old logs")
        load_logs(optimizer, logs=[logs_path])
        pass
    logger = JSONLogger(path=f"opt_logs/logs_{log_attempt+1}_desc.log")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    print("optimizer built. Running...")

    # run the optimizer
    optimizer.maximize(
        init_points=0,
        n_iter=16,
    )

    # do something with that data?
    plot_hyperparameter_scatter(f"{logs_path}.json")
    # ???
    pass


if __name__ == '__main__':
    model_optimization()
    pass
