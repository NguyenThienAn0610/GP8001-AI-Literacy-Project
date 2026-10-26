"""Fixed exploratory model configurations, shared by benchmark scripts."""

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from career_ml import make_pipeline, make_preprocessor


MODEL_NAMES = ("hist_gradient_boosting", "mlp_neural_network", "logistic_regression")


def make_experiment(model_name: str):
    """Return preprocessing, estimator, and a human-readable configuration."""
    if model_name == "logistic_regression":
        pipeline = make_pipeline(model_name)
        return pipeline["preprocess"], pipeline["model"], {
            "family": "Existing Logistic Regression reference",
            "profile_encoding": "Field-aware TF-IDF unigrams and bigrams",
            "dense": False,
        }
    preprocess = make_preprocessor()
    if model_name == "hist_gradient_boosting":
        preprocess.set_params(
            profile__tfidf__ngram_range=(1, 1),
            profile__tfidf__use_idf=False,
            profile__tfidf__binary=True,
            profile__tfidf__norm=None,
            sparse_threshold=0,
        )
        model = HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-6,
            random_state=42,
        )
        description = {
            "family": "Histogram Gradient Boosting",
            "profile_encoding": "Field-aware binary unigrams",
            "dense": True,
        }
    elif model_name == "mlp_neural_network":
        preprocess.set_params(profile__tfidf__ngram_range=(1, 1))
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.01,
            batch_size=128,
            learning_rate_init=0.001,
            max_iter=180,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            tol=0.0001,
            random_state=42,
        )
        description = {
            "family": "Multilayer Perceptron (64 and 32 hidden units)",
            "profile_encoding": "Field-aware TF-IDF unigrams",
            "dense": False,
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return preprocess, model, description


def describe_preprocessing(preprocess):
    """Record the exact feature encoding without serializing Python callables."""
    parameters = preprocess.get_params()
    vectorizer_fields = (
        "ngram_range", "use_idf", "smooth_idf", "binary", "norm", "sublinear_tf",
        "lowercase", "token_pattern", "min_df", "max_df", "max_features",
    )
    return {
        "profile_transformer": "career_ml.ProfileTextTransformer",
        "tokenizer": "str.split",
        "vectorizer": {
            key: parameters[f"profile__tfidf__{key}"] for key in vectorizer_fields
        },
        "sparse_threshold": parameters["sparse_threshold"],
        "numeric_imputer_strategy": parameters["numeric__imputer__strategy"],
        "numeric_scaler_with_mean": parameters["numeric__scaler__with_mean"],
        "numeric_scaler_with_std": parameters["numeric__scaler__with_std"],
    }


