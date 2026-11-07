"""The single scoring-time preprocessor, shared by training, validation and the service.

    raw record ─► FeatureBuilder ─► Winsorizer ─► ColumnTransformer
                  domain rules      learned caps   numeric:     median imputer (+ missing flags) ─► StandardScaler
                                                   categorical: most-frequent imputer ─► OneHotEncoder

Every column gets an imputer, whether or not it had missing values in the training data, so a record
missing any feature at scoring time is still handled. Output is a pandas DataFrame with stable names.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pd_model.features.transformers import ENGINEERED, FeatureBuilder, Winsorizer

MISSING_PREFIX = "missingindicator_"


def build_preprocessor(numeric: list[str], categorical: list[str], *, winsorize: dict[str, float],
                       age_placeholder: float = 99, income_cap: float = 30_000) -> Pipeline:
    raw_columns = [c for c in numeric if c not in ENGINEERED] + list(categorical)
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        # drop="first": the lowest code is the reference level, which keeps logistic regression
        # identifiable. An unseen code is encoded as all zeros, i.e. as the reference level.
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    return Pipeline([
        ("features", FeatureBuilder(raw_columns, categorical, age_placeholder, income_cap)),
        ("winsor", Winsorizer(winsorize)),
        ("columns", ColumnTransformer([("num", numeric_pipe, list(numeric)),
                                       ("cat", categorical_pipe, list(categorical))],
                                      remainder="drop", verbose_feature_names_out=False)),
    ]).set_output(transform="pandas")


def build_preprocessor_from_config(cfg) -> Pipeline:
    p = cfg.preprocessing
    return build_preprocessor(cfg.numeric, cfg.categorical, winsorize=p["winsorize"],
                              age_placeholder=p["age_placeholder"], income_cap=p["income_cap"])


def output_sources(feature_names_out, categorical: list[str]) -> dict[str, str]:
    """Map each preprocessed column to the raw/engineered feature it comes from.

    ``region_risk_1`` -> ``region_risk``; ``missingindicator_age`` -> ``age``; ``dti`` -> ``dti``."""
    sources = {}
    for name in feature_names_out:
        if name.startswith(MISSING_PREFIX):
            sources[name] = name[len(MISSING_PREFIX):]
            continue
        sources[name] = next((c for c in categorical if name.startswith(f"{c}_")), name)
    return sources


def select_outputs(feature_names_out, sources: list[str], categorical: list[str]) -> list[str]:
    """Preprocessed columns that carry the given source features (values only, no missing flags)."""
    mapping = output_sources(feature_names_out, categorical)
    return [n for n in feature_names_out if mapping[n] in sources and not n.startswith(MISSING_PREFIX)]


def learned_parameters(preprocessor: Pipeline) -> dict:
    """Human-readable view of everything the fitted preprocessor learned on the training data."""
    cols = preprocessor.named_steps["columns"]
    num, cat = cols.named_transformers_["num"], cols.named_transformers_["cat"]
    num_features = list(cols.transformers_[0][2])
    cat_features = list(cols.transformers_[1][2])
    medians = num.named_steps["impute"].statistics_
    modes = cat.named_steps["impute"].statistics_
    categories = cat.named_steps["onehot"].categories_
    scaler = num.named_steps["scale"]  # its inputs include the missing-value indicator columns
    return {
        "winsor_caps": preprocessor.named_steps["winsor"].caps_,
        "numeric_impute_median": dict(zip(num_features, map(float, medians), strict=True)),
        "categorical_impute_mode": dict(zip(cat_features, modes.tolist(), strict=True)),
        "categories": {c: list(map(str, k)) for c, k in zip(cat_features, categories, strict=True)},
        "scaler_mean": dict(zip(scaler.feature_names_in_, map(float, scaler.mean_), strict=True)),
        "scaler_scale": dict(zip(scaler.feature_names_in_, map(float, scaler.scale_), strict=True)),
        "output_features": list(preprocessor.get_feature_names_out()),
    }
