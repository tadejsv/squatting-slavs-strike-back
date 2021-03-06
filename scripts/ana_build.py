import numpy as np
import pandas as pd
import gc


def make_features():
    
    #######################
    # Make transaction features
    column_names = ['client_id', 'tran_amt_rur', 'mcc_cd']
    transaction = pd.read_csv('data/trxn.csv', usecols=column_names)
    transaction_ft = transaction[column_names]
    del transaction
    gc.collect()
    transaction_ft['mcc_cd'] = transaction_ft['mcc_cd'].astype('str')

    temp_trs = transaction_ft.groupby(['client_id', 'mcc_cd']).sum().reset_index()
    transaction_ft = temp_trs.loc[temp_trs.groupby('client_id').tran_amt_rur.idxmax()]
    transaction_ft = transaction_ft.set_index('client_id')
    transaction_ft['tran_amt_rur'] = transaction_ft['tran_amt_rur'].fillna('nan')
    del temp_trs
    gc.collect()
    
    #######################
    # Make balance features

    balances = pd.read_csv("data/balance.csv")

    # Sum up across all accounts
    account_sums = balances.groupby(["client_id", "month_end_dt"])[
        ["avg_bal_sum_rur", "max_bal_sum_rur", "min_bal_sum_rur"]
    ].sum()
    del balances
    gc.collect()

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

    # Get differences from the average in the last month
    account_sums["diff_last_month"] = account_sums["avg_bal_sum_rur"] - last_month_avg
    diffs = (
        account_sums["diff_last_month"]
        .reset_index()
        .query('month_end_dt != "2019-08-31"')
    )
    diffs = diffs.pivot(index="client_id", columns="month_end_dt")

    # Put together all balance features
    diffs.columns = [f"balance_diff_{c[1]}" for c in diffs.columns]
    balance_ft = diffs
    balance_ft["avg_range"] = avg_range
    balance_ft["last_month_avg"] = last_month_avg
    del diffs, account_sums, last_month_avg, avg_range
    gc.collect()

    #######################
    # Make aum features

    aum = pd.read_csv('data/aum.csv')

    # Sum up across all accounts
    aum_sums = aum.groupby(['client_id', 'month_end_dt'])[['balance_rur_amt']].sum()
    del aum
    gc.collect()

    # Get STD for last few months for client
    aum_std = aum_sums.std(level='client_id', skipna=True)

    # Get the average amount in the last month
    last_month_aum = aum_sums.iloc[aum_sums.index.get_level_values('month_end_dt') == '2019-08-31']
    last_month_aum = last_month_aum['balance_rur_amt']
    last_month_aum.index = last_month_aum.index.droplevel('month_end_dt')

    # Get differences from the average in the last month
    aum_sums['diff_last_month'] = aum_sums['balance_rur_amt'] - last_month_aum
    diffs = aum_sums['diff_last_month'].reset_index().query('month_end_dt != "2019-08-31"')
    diffs = diffs.pivot(index='client_id', columns='month_end_dt')

    # Put together all balance features
    diffs.columns = [f'aum_diff_{c[1]}' for c in diffs.columns]
    aum_ft = diffs
    aum_ft['aum_std'] = aum_std
    aum_ft['aum_last_month'] = last_month_aum
    del diffs, aum_sums, aum_std, last_month_aum
    gc.collect()

    #######################
    # Make client features

    client = pd.read_csv("data/client.csv")

    # Take out citizenship and job_type, they are useless
    client_ft = client.set_index("client_id")[
        ["gender", "age", "region", "city", "education"]
    ]
    del client
    gc.collect()

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
    del region_counts, top_regions, city_counts, top_cities
    gc.collect()

    #############################
    # Merge all features and save

    full_data = pd.concat([
        balance_ft,
        client_ft,
        transaction_ft
    ], axis=1)
    full_data[['mcc_cd', 'tran_amt_rur']] = full_data[['mcc_cd', 'tran_amt_rur']].fillna('nan')
    full_data.to_pickle('final_version.pickle')


if __name__ == "__main__":
    make_features()