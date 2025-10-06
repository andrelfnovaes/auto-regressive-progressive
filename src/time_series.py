
import numpy as np
import pandas as pd
from pathlib import Path

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
































