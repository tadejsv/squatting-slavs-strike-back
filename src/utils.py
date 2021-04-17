from typing import Tuple

import pandas as pd


def train_val_test_split(
    df: pd.DataFrame, split: Tuple[float, float, float]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Produce a test, val and test split of a dataframe.

    Args:
        df: The dataframe to split
        split: A tuple of shares for (train, val, test) sets - should sum up to 1

    Returns:
        (train, val, test): The split dataframes
    """

    assert sum(split) == 1, "The sum of split should be 1"

    train = df.sample(frac=split[0], replace=False, random_state=42)
    val_test = df[~df.index.isin(train.index)]
    val = val_test.sample(
        frac=split[1] / (split[1] + split[2]), replace=False, random_state=42
    )
    test = val_test[~val_test.index.isin(val.index)]

    return train, val, test


CALL_COST = 400 / 0.1


def calculate_profit(df: pd.DataFrame) -> pd.Series:
    """Calculate profit """

    profit = df["sale_flg"] * df["sale_amount"].fillna(0) - df["contacts"] * CALL_COST

    return profit