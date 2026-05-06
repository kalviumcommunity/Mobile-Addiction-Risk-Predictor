"""
Data loading module
Responsible only for loading dataset
"""

import pandas as pd


def load_data(filepath):
    """
    Load dataset from CSV file.

    Parameters:
        filepath (str): Dataset path

    Returns:
        DataFrame
    """
    try:
        df = pd.read_csv(filepath)

        if df.empty:
            raise ValueError("Dataset is empty.")

        return df

    except FileNotFoundError:
        print("❌ File not found.")