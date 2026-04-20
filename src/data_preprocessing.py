"""
Data preprocessing module
Handles loading dataset and splitting data
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    """
    Load CSV dataset.

    Parameters:
        path (str): File path

    Returns:
        DataFrame
    """
    return pd.read_csv(path)


def split_data(df, target, test_size, random_state):
    """
    Split dataset into train and test sets.

    Parameters:
        df (DataFrame): Full dataset
        target (str): Target column name
        test_size (float): Test split ratio
        random_state (int): Seed value

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )