# Imports

import numpy as np
import pandas as pd

from typing import Union, Optional, List, Type, Tuple
from pathlib import Path

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_percentage_error

from sklearn.linear_model import LinearRegression


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


def generate_lags(
    n_lags_past: int,
    n_lags_future: int,
) -> tuple[list[str], list[str]]:
    """
    Generate lag names for AR and ARP models.

    Args:
        past_lags (int): Number of past lags (for both AR and ARP).
        future_lags (int): Number of future lags (used only in ARP).

    Returns:
        tuple[list[str], list[str]]: A tuple containing:
            - lags_ar: list of past lags for AR model (e.g., ['y-2', 'y-1'])
            - lags_arp: list of lags for ARP model using second half of past lags + future lags
                      (e.g., ['y-1', 'y+1', 'y+2'])
    """
    # Past lags (y-2, y-1, ..., y-1)
    lags_past = [f"y-{i}" for i in range(n_lags_past, 0, -1)]

    # Future lags (y+1, y+2, ..., y+f)
    lags_future = [f"y+{i}" for i in range(1, n_lags_future + 1)]

    # AR model uses all past lags
    lags_ar = lags_past

    # ARP model uses second half of past lags + all future lags
    lags_arp = lags_past[len(lags_past) // 2:] + lags_future

    return lags_past, lags_future, lags_ar, lags_arp


def generate_lagged_df(
    series: Union[pd.Series, pd.DataFrame],
    n_lags_past: int,
    n_lags_future: int,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Generate a lagged DataFrame with columns ordered:
    [y - p, ..., y - 1, y, y + 1, ..., y + f].

    Uses generate_lags() internally.
    """
    if isinstance(series, pd.Series):
        series = series.to_frame()

    # Get lag names
    lags_past, lags_future, lags_ar, lags_arp = generate_lags(n_lags_past, n_lags_future)

    # Build a single correct order (past once + y + future once)
    ordered_lags = lags_ar + ["y"] + lags_future

    dfs = []

    for lag_name in ordered_lags:
        if lag_name == "y":
            shifted = series.copy()
        elif lag_name.startswith("y-"):
            shift_val = int(lag_name.split("-")[1])
            shifted = series.shift(shift_val)
        elif lag_name.startswith("y+"):
            shift_val = int(lag_name.split("+")[1])
            shifted = series.shift(-shift_val)
        else:
            raise ValueError(f"Invalid lag name: {lag_name}")

        shifted.columns = [lag_name for _ in series.columns]
        dfs.append(shifted)

    df = pd.concat(dfs, axis=1)

    return df.dropna() if dropna else df


def split_df(
    df: pd.DataFrame,
    train_test_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into training and testing sets based on the row count.

    Args:
        df (pd.DataFrame): The DataFrame to split (e.g., lagged features).
        train_test_ratio (float): Proportion of data to use for training (e.g., 0.8).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (df_train, df_test)
    """
    if not 0 < train_test_ratio < 1:
        raise ValueError("train_test_ratio must be between 0 and 1 (exclusive).")

    split_idx = int(len(df) * train_test_ratio)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    return df_train, df_test


def fit_model(
    df: pd.DataFrame,
    features: List[str],
    model_class: Type[BaseEstimator],
    **model_kwargs
) -> BaseEstimator:
    """
    Fit a scikit-learn-like model using selected features to predict the 'y' column.

    Args:
        df (pd.DataFrame): DataFrame containing the 'y' column and feature columns.
        features (List[str]): List of column names to be used as input features.
        model_class (Type[BaseEstimator]): A scikit-learn-compatible model class.
        **model_kwargs: Additional keyword arguments passed to the model constructor.

    Returns:
        BaseEstimator: A fitted model instance.

    Raises:
        ValueError: If 'y' column or any feature column is missing in the DataFrame.
    """
    if 'y' not in df.columns:
        raise ValueError("The DataFrame must contain a 'y' column as target.")

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"The following feature columns are missing: {missing_features}")

    X = df[features].copy().values
    y = df['y'].copy().values

    model = model_class(**model_kwargs)
    model.fit(X, y)

    return model


def predict_evaluate_model(
    model: BaseEstimator,
    df: pd.DataFrame,
    features: List[str]
) -> Tuple[pd.Series, float]:
    """
    Use a fitted model to predict and evaluate MAPE on new data.

    Args:
        model (BaseEstimator): A fitted scikit-learn-like model.
        df (pd.DataFrame): DataFrame containing 'y' and feature columns.
        features (List[str]): List of column names used as model inputs.

    Returns:
        Tuple[pd.Series, float]: Tuple containing:
            - pd.Series of predictions (same index as df)
            - float: MAPE score

    Raises:
        ValueError: If required columns are missing in the DataFrame.
    """
    if 'y' not in df.columns:
        raise ValueError("The DataFrame must contain a 'y' column as target.")

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"The following feature columns are missing: {missing_features}")

    X = df[features].copy()
    y_true = df['y'].copy()

    y_pred = pd.Series(model.predict(X), index=df.index, name='y_pred')

    mape = mean_absolute_percentage_error(y_true, y_pred)

    return y_pred, mape































