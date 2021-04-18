import gc

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd


months_to_quarters = {
    "2018-09-30": "Q1",
    "2018-10-31": "Q1",
    "2018-11-30": "Q1",
    "2018-12-31": "Q2",
    "2019-01-31": "Q2",
    "2019-02-28": "Q2",
    "2019-03-31": "Q3",
    "2019-04-30": "Q3",
    "2019-05-31": "Q3",
    "2019-06-30": "Q4",
    "2019-07-31": "Q4",
    "2019-08-31": "Q4",
}


def make_features():

    #######################
    # Make balance features

    balances = pd.read_csv("data/balance.csv")
    balances["quarter"] = balances["month_end_dt"].replace(months_to_quarters)

    # Sum up across all accounts by month
    account_sums = balances.groupby(["client_id", "month_end_dt"])[
        ["avg_bal_sum_rur", "max_bal_sum_rur", "min_bal_sum_rur"]
    ].sum()

    # Get average range (max - min) for all clients
    account_sums["range"] = (
        account_sums["max_bal_sum_rur"] - account_sums["min_bal_sum_rur"]
    )
    avg_range = account_sums["range"].mean(level="client_id", skipna=True)

    # Get the average amount in the last month
    balance_last = account_sums.iloc[
        account_sums.index.get_level_values("month_end_dt") == "2019-08-31"
    ]
    balance_last = balance_last["avg_bal_sum_rur"]
    balance_last.index = balance_last.index.droplevel("month_end_dt")
    all_avg = account_sums["avg_bal_sum_rur"].mean(level=0)

    # Get mean avg by quarters
    bal_mean_q = balances.groupby(["client_id", "quarter"])["avg_bal_sum_rur"].mean()

    m3 = bal_mean_q[bal_mean_q.index.get_level_values("quarter") == "Q4"].droplevel(1)
    m6 = bal_mean_q[
        bal_mean_q.index.get_level_values("quarter").isin(["Q4", "Q3"])
    ].mean(level=0)
    m12 = bal_mean_q.mean(level=0)

    balance_ft = pd.DataFrame(
        {
            "balance_range": avg_range,
            "balance_last_avg": balance_last,
            "balance_m3": m3,
            "balance_all_avg": all_avg,
            "balance_rel_range": avg_range / all_avg,
            "balance_diff_m3": balance_last - m3,
            "balance_diff_m6": balance_last - m6,
            "balance_diff_m3_m6": m3 - m6,
            "balance_diff_m3_m12": m3 - m12,
            "balance_diff_m3_rel": (balance_last - m3) / balance_last,
            "balance_diff_m6_rel": (balance_last - m6) / balance_last,
            "balance_diff_m3_m6_rel": (m3 - m6) / balance_last,
            "balance_diff_m3_m12_rel": (m3 - m12) / balance_last,
        }
    )
    del (
        all_avg,
        balances,
        account_sums,
        avg_range,
        balance_last,
        bal_mean_q,
        m3,
        m6,
        m12,
    )

    #######################
    # Make aum features

    aum = pd.read_csv("data/aum.csv")
    aum["quarter"] = aum["month_end_dt"].replace(months_to_quarters)

    # Sum up across all accounts for each month
    aum_sums = aum.groupby(["client_id", "month_end_dt", "quarter"])[
        "balance_rur_amt"
    ].sum()

    # Get mean and STD for last few months for client
    aum_std = aum_sums.std(level="client_id", skipna=True)
    aum_mean = aum_sums.mean(level="client_id", skipna=True)

    # Get the average amount in the last month
    aum_last = aum_sums.iloc[
        aum_sums.index.get_level_values("month_end_dt") == "2019-08-31"
    ]
    aum_last.index = aum_last.index.droplevel(["month_end_dt", "quarter"])

    # Get quarter averages
    aum_sums_q = aum_sums.mean(level=["client_id", "quarter"])

    m3 = aum_sums_q[aum_sums_q.index.get_level_values("quarter") == "Q4"].droplevel(1)
    m6 = aum_sums_q[
        aum_sums_q.index.get_level_values("quarter").isin(["Q4", "Q3"])
    ].mean(level=0)
    m12 = aum_sums_q.mean(level=0)

    # Create features
    aum_ft = pd.DataFrame(
        {
            "aum_std": aum_std,
            "aum_last": aum_last,
            "aum_m3": m3,
            "aum_all_avg": aum_mean,
            "aum_volatility": aum_std / aum_mean,
            "aum_diff_m3": aum_last - m3,
            "aum_diff_m6": aum_last - m6,
            "aum_diff_m3_m6": m3 - m6,
            "aum_diff_m3_m12": m3 - m12,
            "aum_diff_m3_rel": (aum_last - m3) / aum_last,
            "aum_diff_m6_rel": (aum_last - m6) / aum_last,
            "aum_diff_m3_m6_rel": (m3 - m6) / aum_last,
            "aum_diff_m3_m12_rel": (m3 - m12) / aum_last,
        }
    )
    del aum, aum_sums, aum_std, aum_mean, aum_last, aum_sums_q, m3, m6, m12

    #######################
    # Make client features

    client = pd.read_csv("data/client.csv")

    # Take out citizenship and job_type, they are useless
    client_ft = client.set_index("client_id")[
        ["gender", "age", "region", "city", "education"]
    ]
    del client

    # Filter out cities and region to only those above 200, make them categorical (not numerical)
    region_counts = client_ft["region"].value_counts(dropna=False)
    top_regions = region_counts[region_counts > 200].index

    city_counts = client_ft["city"].value_counts(dropna=False)
    top_cities = city_counts[city_counts > 200].index

    client_ft.loc[~client_ft["city"].isin(top_cities), "city"] = None
    client_ft.loc[~client_ft["region"].isin(top_regions), "region"] = None

    client_ft["city"] = client_ft["city"].astype("str")
    client_ft["region"] = client_ft["region"].astype("str")

    client_ft[["gender", "education"]] = client_ft[["gender", "education"]].fillna(
        "nan"
    )

    #######################
    # Make transaction features
    column_names = ["client_id", "tran_amt_rur", "mcc_cd"]
    transaction = pd.read_csv("data/trxn.csv", usecols=column_names)
    transaction_ft = transaction[column_names]
    del transaction
    gc.collect()
    transaction_ft["mcc_cd"] = transaction_ft["mcc_cd"].astype("str")

    temp_trs = transaction_ft.groupby(["client_id", "mcc_cd"]).sum().reset_index()
    transaction_ft = temp_trs.loc[temp_trs.groupby("client_id").tran_amt_rur.idxmax()]
    transaction_ft = transaction_ft.set_index("client_id")
    transaction_ft["tran_amt_rur"] = transaction_ft["tran_amt_rur"].fillna("nan")
    del temp_trs
    gc.collect()

    #######################
    # Make payment features
    payments = pd.read_csv("data/payments.csv")

    payments["month"] = pd.to_datetime(payments["day_dt"]).apply(
        lambda x: x + MonthEnd(1)
    )
    payments["month"] = payments["month"].apply(lambda x: x.strftime("%Y-%m-%d"))
    payments["quarter"] = payments["month"].replace(months_to_quarters)

    # Get pensioneers
    pensioneers = payments.query('pmnts_name == "Pension receipts"').client_id.unique()
    pensioneers = pd.Series(1.0, index=pensioneers)

    payments_sums = payments.groupby(["client_id", "month", "quarter"])["sum_rur"].sum()

    # Get mean and STD for last few months for client
    payments_std = payments_sums.std(level="client_id", skipna=True)
    payments_mean = payments_sums.mean(level="client_id", skipna=True)

    # Get payments last month
    payments_last = payments_sums.iloc[
        payments_sums.index.get_level_values("month") == "2019-08-31"
    ]
    payments_last.index = payments_last.index.droplevel(["month", "quarter"])

    # Get quarter averages
    payments_sums_q = payments_sums.mean(level=["client_id", "quarter"])

    m3 = payments_sums_q[
        payments_sums_q.index.get_level_values("quarter") == "Q4"
    ].droplevel(1)
    m6 = payments_sums_q[
        payments_sums_q.index.get_level_values("quarter").isin(["Q4", "Q3"])
    ].mean(level=0)
    m12 = payments_sums_q.mean(level=0)

    # Create features
    payments_ft = pd.DataFrame(
        {
            "is_pensioneer": pensioneers,
            "payments_std": payments_std,
            "payments_last": payments_last,
            "payments_m3": m3,
            "payments_m6": m6,
            "payments_all_avg": payments_mean,
            "payments_volatility": payments_std / payments_mean,
            "payments_diff_m3": payments_last - m3,
            "payments_diff_m6": payments_last - m6,
            "payments_diff_m3_m6": m3 - m6,
            "payments_diff_m3_m12": m3 - m12,
            "payments_diff_m3_rel": (payments_last - m3) / payments_last,
            "payments_diff_m6_rel": (payments_last - m6) / payments_last,
            "payments_diff_m3_m6_rel": (m3 - m6) / payments_last,
            "payments_diff_m3_m12_rel": (m3 - m12) / payments_last,
        }
    ).fillna({"is_pensioneer": 0})
    del (
        payments,
        payments_sums,
        payments_std,
        payments_mean,
        payments_last,
        payments_sums_q,
        m3,
        m6,
        m12,
        pensioneers,
    )

    ############################
    # Caclculate mystery features
    mystery_feats = pd.read_csv(
        "data/funnel.csv",
        usecols=["client_id"] + [f"feature_{i}" for i in range(1, 11)],
    )
    mystery_feats = mystery_feats.set_index("client_id")
    mystery_feats["feature_1"] = mystery_feats["feature_1"].astype(str)

    #############################
    # Merge all features and save

    full_data = pd.concat(
        [balance_ft, aum_ft, client_ft, transaction_ft, payments_ft, mystery_feats],
        axis=1,
    )
    full_data = full_data.fillna({"mcc_cd": "nan"})
    full_data.to_pickle("final_version.pickle")


if __name__ == "__main__":
    make_features()