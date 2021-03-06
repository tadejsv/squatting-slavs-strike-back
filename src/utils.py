from typing import Tuple

import numpy as np
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


def signed_log10_1p(x):
    return np.sign(x) * np.log10(np.abs(x) + 1)

class ProfitMetric(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight=None):
        profit_sum = 0
        count = 0 

        for i in range(len(target)):
            count += 1

            count_item = approxes[i] > 0
            if count_item:
                profit_sum += target[i]

        return profit_sum / count