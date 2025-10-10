# Imports

import numpy as np
import pandas as pd

from typing import Union, Optional, List, Type
from pathlib import Path

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error


def load_data(filepath: Path) -> pd.DataFrame:
    """
    Load a time series dataset from a CSV file.

    Args:
        filepath (Path): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame indexed by 'Date', with stock tickers as columns.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the 'Date' column is missing.
    """
    # if not filepath.exists():
    #     raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, parse_dates=["Date"])

    if "Date" not in df.columns:
        raise ValueError("Missing required column: 'Date'")

    df.set_index("Date", inplace=True)

    return df


# def generate_lags(
#     past_lags: int,
#     future_lags: int,
# ) -> tuple[list, list]:

#     # Past lags (y - p to y - 1)
#     lags_past = [f"y-{past}" for past in range(past_lags, 0, -1)]

#     # Future lags (y + 1 to y + f)
#     lags_future = [f"y+{future}" for future in range(1, future_lags + 1)]

#     # AR lags
#     lags_ar = lags_past

#     # ARP lags
#     lags_btf = lags_past[(len(lags_past) // 2):] + lags_future

#     return lags_ar, lags_btf


def generate_lags(
    series: Union[pd.Series, pd.DataFrame],
    past_lags: int,
    future_lags: int,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Generate a lagged DataFrame from a univariate or multivariate time series.

    Parameters
    ----------
    series : pd.Series or pd.DataFrame
        Input time series (indexed by datetime).
    past_lags : int, default=1
        Number of past lags to include (y - 1, ..., y - p).
    future_lags : int, default=0
        Number of future lags to include (y + 1, ..., y + f).
    dropna : bool, default=False
        Whether to drop rows with NaNs created by shifting.

    Returns
    -------
    pd.DataFrame
        DataFrame with lagged values in the order:
        [y - p, ..., y - 1, y, y + 1, ..., y + f]
    """
    if isinstance(series, pd.Series):
        series = series.to_frame()

    # List to hold all shifted DataFrames
    dfs = []

    # Past lags (y - p to y - 1)
    for past in range(past_lags, 0, -1):
        lagged = series.shift(past)
        lagged.columns = [f"y-{past}" for _ in series.columns]
        dfs.append(lagged)

    # Current y
    current = series.copy()
    current.columns = ["y" for _ in series.columns]
    dfs.append(current)

    # Future lags (y + 1 to y + f)
    for future in range(1, future_lags + 1):
        lead = series.shift(-future)
        lead.columns = [f"y+{future}" for _ in series.columns]
        dfs.append(lead)

    # Concatenate all
    df = pd.concat(dfs, axis=1)

    return df.dropna() if dropna else df


def fit_model_from_lagged(
    df: pd.DataFrame,
    lags: List[str],
    model_class: Type[BaseEstimator],
    **model_kwargs
) -> tuple[BaseEstimator, float]:
    """
    Fit a scikit-learn-like model using lagged features to predict the 'y' column,
    and evaluate it using Mean Absolute Percentage Error (MAPE).

    Args:
        df (pd.DataFrame): DataFrame containing the 'y' column and lag/lead features.
        lags (List[str]): List of column names to be used as input features.
        model_class (Type[BaseEstimator]): A scikit-learn-compatible model class.
        **model_kwargs: Additional keyword arguments passed to the model constructor.

    Returns:
        tuple[BaseEstimator, float]: A fitted model instance and its MAPE on training data.

    Raises:
        ValueError: If 'y' column or any lag feature is missing in the DataFrame.
    """
    if 'y' not in df.columns:
        raise ValueError("The DataFrame must contain a 'y' column as target.")

    missing_lags = [lag for lag in lags if lag not in df.columns]
    if missing_lags:
        raise ValueError(f"The following lag columns are missing: {missing_lags}")

    X = df[lags].copy().values
    y = df['y'].copy().values

    model = model_class(**model_kwargs)
    model.fit(X, y)

    y_pred = model.predict(X)
    mape = mean_absolute_percentage_error(y, y_pred)

    return model, mape


def run_tsf_experiment(
    series: Union[pd.Series, pd.DataFrame],
    n_lags_past: int,
    n_lags_future: int,
    features: list,
    model_class: Type[BaseEstimator],
    train_test_ratio: float = 0.8,
    dropna: bool = True,
    **model_kwargs
) -> dict:
    """
    Run a full time series forecasting experiment using past and future lags.

    This includes:
        - Lag generation
        - Dataframe transformation
        - Train/test split
        - Model fitting
        - Prediction and MAPE evaluation

    Args:
        series (pd.Series or pd.DataFrame): Input time series.
        n_lags_past (int): Number of past lags to include.
        n_lags_future (int): Number of future lags to include.
        model_class (Type[BaseEstimator]): A scikit-learn-like model class.
        train_test_ratio (float): Ratio of train size to full dataset (default 0.8).
        dropna (bool): Whether to drop rows with NaNs in the lagged dataframe.
        **model_kwargs: Keyword arguments passed to the model constructor.

    Returns:
        dict: Dictionary with:
            - model: fitted model
            - y_pred: test predictions
            - mape: MAPE on test
            - df_train: training dataframe
            - df_test: testing dataframe
    """
    # 1. Generate lag names
    lags_past, lags_future, lags_ar, lags_arp = generate_lags(n_lags_past=2*n_lags_future,
                                                          n_lags_future=n_lags_future
                                                          )

    # 2. Create lagged dataset
    df_lagged = generate_lagged_df(
        series=y,
        n_lags_past=2*n_lags_future,
        n_lags_future=n_lags_future,
    )

    # 3. Split into train/test sets
    df_train, df_test = split_df(
        df=df_lagged,
        train_test_ratio=train_test_ratio
    )

    # 4. Fit model on training set using ARP lags (second-half past + future)
    model = fit_model(
        df=df_train,
        features=features,
        model_class=model_class,
        **model_kwargs
    )

    # 5. Predict and evaluate on training set
    y_pred_train, mape_train = predict_evaluate_model(
        model=model,
        df=df_train,
        features=features
    )

    # 5. Predict and evaluate on testing set
    y_pred_test, mape_test = predict_evaluate_model(
        model=model,
        df=df_test,
        features=features
    )

    # 6. Return results
    return {
        "model": model,
        "df_train": df_train,
        "df_test": df_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "mape_train": mape_train,
        "mape_test": mape_test,
    }



























