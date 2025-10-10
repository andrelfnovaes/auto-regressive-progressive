# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    X = df[features].copy().values
    y_true = df['y'].copy().values

    y_pred = pd.Series(model.predict(X), index=df.index, name='y_pred')

    mape = mean_absolute_percentage_error(y_true, y_pred)

    return y_pred, mape


def evaluate_methodology(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features_train: List[str],
    features_test: List[str],
    model_class: Type[BaseEstimator],
    **model_kwargs
) -> tuple[pd.Series, float, pd.Series, float]:
    """
    Train a model on df_train and evaluate it on both train and test sets using MAPE.

    Args:
        df_train (pd.DataFrame): Training data with 'y' and feature columns.
        df_test (pd.DataFrame): Testing data with 'y' and feature columns.
        features (List[str]): List of lag feature names.
        model_class (Type[BaseEstimator]): Model class (e.g. LinearRegression).
        **model_kwargs: Optional parameters for model initialization.

    Returns:
        Tuple containing:
            - y_pred_train (pd.Series): Predictions on train set.
            - mape_train (float): MAPE on train set.
            - y_pred_test (pd.Series): Predictions on test set.
            - mape_test (float): MAPE on test set.
    """
    # Fit model on training set
    model = fit_model(
        df=df_train,
        features=features_train,
        model_class=model_class,
        **model_kwargs
    )

    # Predict + evaluate on both train and test sets
    y_pred_train, mape_train = predict_evaluate_model(model, df_train, features_train)
    y_pred_test, mape_test = predict_evaluate_model(model, df_test, features_test)

    return y_pred_train, y_pred_test, mape_train, mape_test


































def plot_forecast_vs_actual(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_pred: pd.Series,
    title: str = "Forecast vs Actual"
) -> None:
    """
    Plot training and testing actual values alongside test predictions.

    Args:
        df_train (pd.DataFrame): Training set with 'y' column.
        df_test (pd.DataFrame): Testing set with 'y' column.
        y_pred (pd.Series): Predictions on the test set (must align with df_test index).
        title (str): Plot title.
    """
    plt.figure(figsize=(12, 6))

    # Plot actual values
    plt.plot(df_train.index, df_train['y'], label='Train Actual', color='blue', linestyle='-')
    plt.plot(df_test.index, df_test['y'], label='Test Actual', color='black', linestyle='--')

    # Plot predicted values
    plt.plot(y_pred.index, y_pred, label='Test Predicted', color='orange', linestyle='--', marker='o')

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_train_test_predictions(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_pred_train: pd.Series,
    y_pred_test: pd.Series,
    title: str = "Train/Test Forecast vs Actual"
) -> None:
    """
    Plot actual vs predicted values for both train and test sets in separate subplots.

    Args:
        df_train (pd.DataFrame): Training set with 'y' column.
        df_test (pd.DataFrame): Testing set with 'y' column.
        y_pred_train (pd.Series): Predictions on the training set.
        y_pred_test (pd.Series): Predictions on the testing set.
        title (str): Main title of the plot.
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Train Plot ---
    axs[0].plot(df_train.index, df_train['y'], label='Train Actual', color='black')
    axs[0].plot(y_pred_train.index, y_pred_train, label='Train Predicted', color='orange', linestyle='--', marker='o')
    axs[0].set_title("Training Set")
    axs[0].set_ylabel("y")
    axs[0].legend()
    axs[0].grid(True)

    # --- Test Plot ---
    axs[1].plot(df_test.index, df_test['y'], label='Test Actual', color='black')
    axs[1].plot(y_pred_test.index, y_pred_test, label='Test Predicted', color='orange', linestyle='--', marker='o')
    axs[1].set_title("Testing Set")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("y")
    axs[1].legend()
    axs[1].grid(True)

    # --- Layout ---
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()












