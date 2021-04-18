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

    balances = pd.read_csv("train_data/balance.csv")
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
    last_month_avg = account_sums.iloc[
        account_sums.index.get_level_values("month_end_dt") == "2019-08-31"
    ]
    last_month_avg = last_month_avg["avg_bal_sum_rur"]
    last_month_avg.index = last_month_avg.index.droplevel("month_end_dt")
    all_avg = account_sums["avg_bal_sum_rur"].mean(level=0)

    # Get mean avg by quarters
    bal_mean_q = balances.groupby(["client_id", "quarter"])["avg_bal_sum_rur"].mean()

    # Get last month avg - 3 months avg
    m3 = bal_mean_q[bal_mean_q.index.get_level_values("quarter") == "Q4"].droplevel(1)
    m3_diff = last_month_avg - m3

    # Get last month avg - 6 months avg
    m6_diff = bal_mean_q[
        bal_mean_q.index.get_level_values("quarter").isin(["Q4", "Q3"])
    ].mean(level=0)
    m6_diff = last_month_avg - m6_diff

    # Get last 3 months - last 6 months
    m3_m6diff = m6_diff - m3_diff

    # Get last 3 months - last 12 months
    m3_m12diff = bal_mean_q.mean(level=0)
    m3_m12diff = m3 - m3_m12diff

    balance_ft = pd.DataFrame(
        {
            "balance_range": avg_range,
            "balance_last_avg": last_month_avg,
            "balance_m3": m3,
            "balance_all_avg": all_avg,
            "balance_rel_range": avg_range / all_avg,
            "balance_diff_m3": m3_diff,
            "balance_diff_m6": m6_diff,
            "balance_diff_m3_m6": m3_m6diff,
            "balance_diff_m3_m12": m3_m12diff,
            "balance_diff_m3_rel": m3_diff / last_month_avg,
            "balance_diff_m6_rel": m6_diff / last_month_avg,
            "balance_diff_m3_m6_rel": m3_m6diff / last_month_avg,
            "balance_diff_m3_m12_rel": m3_m12diff / last_month_avg,
        }
    )
    del (
        all_avg,
        balances,
        account_sums,
        avg_range,
        last_month_avg,
        bal_mean_q,
        m3,
        m3_diff,
        m6_diff,
        m3_m6diff,
        m3_m12diff,
    )

    #######################
    # Make aum features

    aum = pd.read_csv("train_data/aum.csv")
    aum["quarter"] = aum["month_end_dt"].replace(months_to_quarters)

    # Sum up across all accounts for each month
    aum_sums = aum.groupby(["client_id", "month_end_dt", "quarter"])[
        "balance_rur_amt"
    ].sum()

    # Get mean and STD for last few months for client
    aum_std = aum_sums.std(level="client_id", skipna=True)
    aum_mean = aum_sums.mean(level="client_id", skipna=True)
    aum_volatility = aum_std / aum_mean

    # Get the average amount in the last month
    last_month_aum = aum_sums.iloc[
        aum_sums.index.get_level_values("month_end_dt") == "2019-08-31"
    ]
    last_month_aum.index = last_month_aum.index.droplevel(["month_end_dt", "quarter"])

    # Get quarter averages
    aum_sums_q = aum_sums.mean(level=["client_id", "quarter"])

    # Get last month avg - 3 months avg
    m3 = aum_sums_q[aum_sums_q.index.get_level_values("quarter") == "Q4"].droplevel(1)
    m3_diff = last_month_aum - m3

    # Get last month avg - 6 months avg
    m6_diff = aum_sums_q[
        aum_sums_q.index.get_level_values("quarter").isin(["Q4", "Q3"])
    ].mean(level=0)
    m6_diff = last_month_aum - m6_diff

    # Get last 3 months - last 6 months
    m3_m6diff = m6_diff - m3_diff

    # Get last 3 months - last 12 months
    m3_m12diff = aum_sums_q.mean(level=0)
    m3_m12diff = m3 - m3_m12diff

    # Create features
    aum_ft = pd.DataFrame(
        {
            "aum_std": aum_std,
            "aum_last": last_month_aum,
            "aum_m3": m3,
            "aum_all_avg": aum_mean,
            "aum_volatility": aum_volatility,
            "aum_diff_m3": m3_diff,
            "aum_diff_m6": m6_diff,
            "aum_diff_m3_m6": m3_m6diff,
            "aum_diff_m3_m12": m3_m12diff,
            "aum_diff_m3_rel": m3_diff / last_month_aum,
            "aum_diff_m6_rel": m6_diff / last_month_aum,
            "aum_diff_m3_m6_rel": m3_m6diff / last_month_aum,
            "aum_diff_m3_m12_rel": m3_m12diff / last_month_aum,
        }
    )
    del (
        aum,
        aum_sums,
        aum_std,
        aum_mean,
        last_month_aum,
        aum_sums_q,
        m3,
        m3_diff,
        m6_diff,
        m3_m6diff,
        m3_m12diff,
    )

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

    # Get pensioneers
    pensioneers = payments.query('pmnts_name == "Pension receipts"').client_id.unique()
    pensioneers = pd.DataFrame({"is_pensioneer": 1}, index=pensioneers)

    # Get date to first of month for grouping
    payments["month"] = pd.to_datetime(payments["day_dt"]).apply(
        lambda x: x + MonthEnd(1)
    )
    payments_months = (
        payments.groupby(["client_id", "month"]).sum("sum_rur").reset_index()
    )
    del payments

    payments_months = payments_months.pivot(index="client_id", columns="month")
    payments_months.columns = [
        f'payments_{c[1].strftime("%Y_%m_%d")}' for c in payments_months.columns
    ]
    payments_months["payments_mean"] = payments_months.mean(axis=1)
    payments_months["payments_std"] = payments_months.std(axis=1)

    payments_ft = pd.concat([payments_months, pensioneers], axis=1).fillna(
        {"is_pensioneer": 0}
    )
    del payments_months, pensioneers

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