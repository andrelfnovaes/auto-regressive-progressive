
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset from CSV.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with Date as index and stocks as columns.
    """
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    return df
































