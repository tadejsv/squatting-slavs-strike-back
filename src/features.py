import gc
import numpy as np
import pandas as pd

from pandas.tseries.offsets import MonthEnd


def get_column_nms(cols: pd.core.indexes.multi.MultiIndex,
                   prefix='',
                   suffix=''):
    nms = ['_'.join(col) for col in cols]
    if prefix and prefix != '':
        nms = ['_'.join([prefix, nm]) for nm in nms]
    if suffix and suffix != '':
        nms = ['_'.join([nm, suffix]) for nm in nms]
    return nms


class CampaignFtExtractor:
    def __init__(self, fn='train_data/com.csv'):
        self._fn = fn

    def load_transform(self):
        campaign = pd.read_csv(self._fn)

        out = []

        for mask, subset_nm in [
            (np.full(campaign.shape[0], True), ''),
            (campaign['prod'] == 'Cash Loan', 'cash_loan')
        ]:

            ft = campaign[mask].groupby('client_id').agg({
                'agr_flg': ['sum', 'mean'],
                'dumaet':  ['sum', 'mean'],
                'otkaz':   ['sum', 'mean'],
                'count_comm': ['sum', 'mean', 'median'],
            })
            # rename columns
            ft.columns = get_column_nms(ft.columns, prefix=subset_nm)
            out.append(ft)

        del campaign
        gc.collect()
        return pd.concat(out, axis=1)


class ClientFtExtractor():
    def __init__(self, fn='train_data/client.csv'):
        self._fn = fn

    def load_transform(self):
        client = pd.read_csv(self._fn)

        # Take out citizenship and job_type, they are useless
        client_ft = client.set_index('client_id')[['gender', 'age', 'region', 'city', 'education']]

        # Filter out cities and region to only those above 200, make them categorical (not numerical)
        region_counts = client_ft['region'].value_counts(dropna=False)
        top_regions = region_counts[region_counts > 200].index

        city_counts = client_ft['city'].value_counts(dropna=False)
        top_cities = city_counts[city_counts > 200].index

        client_ft.loc[~ client_ft['city'].isin(top_cities), 'city'] = None
        client_ft.loc[~ client_ft['region'].isin(top_regions), 'region'] = None

        client_ft['city'] = client_ft['city'].astype('str')
        client_ft['region'] = client_ft['region'].astype('str')

        client_ft[['gender', 'education']] = client_ft[['gender', 'education']].fillna('nan')

        del client
        gc.collect()
        return client_ft


class BalanceFtExtractor:
    def __init__(self, fn='train_data/balance.csv'):
        self._fn = fn

    def load_transform(self):
        balances = pd.read_csv(self._fn)

        # Sum up across all accounts
        account_sums = balances.groupby(['client_id', 'month_end_dt'])[['avg_bal_sum_rur', 'max_bal_sum_rur', 'min_bal_sum_rur']].sum()

        # Get average range (max - min) for all clients
        account_sums['range'] = account_sums['max_bal_sum_rur'] - account_sums['min_bal_sum_rur']
        avg_range = account_sums['range'].mean(level='client_id', skipna=True)

        # Get the average amount in the last month
        last_month_avg = account_sums.iloc[account_sums.index.get_level_values('month_end_dt') == '2019-08-31']
        last_month_avg = last_month_avg['avg_bal_sum_rur']
        last_month_avg.index = last_month_avg.index.droplevel('month_end_dt')

        # Get differences from the average in the last month
        account_sums['diff_last_month'] = account_sums['avg_bal_sum_rur'] - last_month_avg
        diffs = account_sums['diff_last_month'].reset_index().query('month_end_dt != "2019-08-31"')
        diffs = diffs.pivot(index='client_id', columns='month_end_dt')

        # Put together all balance features
        diffs.columns = [f'balance_diff_{c[1]}' for c in diffs.columns]
        balance_ft = diffs
        balance_ft['avg_range'] = avg_range
        balance_ft['avg_last_balance'] = last_month_avg
        del balances
        gc.collect()
        return balance_ft




class AUMFtExtractor:
    def __init__(self, fn='train_data/aum.csv'):
        self._fn = fn

    def load_transform(self):
        aum = pd.read_csv(self._fn)

        # Sum up across all accounts
        aum_sums = aum.groupby(['client_id', 'month_end_dt'])[['balance_rur_amt']].sum()

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

        del aum
        gc.collect()
        return aum_ft


class TrxnFtExtractor:
    def __init__(self, fn='train_data/trxn.csv'):
        self._fn = fn

    def load_transform(self):
        column_names = ['client_id', 'tran_amt_rur', 'mcc_cd', 'tran_time']
        transaction = pd.read_csv(self._fn,
                                  usecols=column_names)

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
        return transaction_ft


class PaymentFtExtractor:
    def __init__(self, fn='train_data/payments.csv'):
        self._fn = fn

    def load_transform(self):
        payments = pd.read_csv(self._fn)

        # Get pensioneers
        pensioneers = payments.query('pmnts_name == "Pension receipts"').client_id.unique()
        pensioneers = pd.DataFrame({'is_pensioneer': 1}, index=pensioneers)

        # Get date to first of month for grouping
        payments['month'] = pd.to_datetime(payments['day_dt']).apply(lambda x: x + MonthEnd(1))
        payments_months = payments.groupby(['client_id', 'month']).sum('sum_rur').reset_index()
        del payments
        gc.collect()

        payments_months = payments_months.pivot(index='client_id', columns='month')
        payments_months.columns = [f'payments_{c[1].strftime("%Y_%m_%d")}' for c in payments_months.columns]
        payments_months['payments_mean'] = payments_months.mean(axis=1)
        payments_months['payments_std'] = payments_months.std(axis=1)

        payments_ft = pd.concat([
            payments_months,
            pensioneers
        ], axis=1).fillna({'is_pensioneer': 0})
        del payments_months, pensioneers
        gc.collect()
        return payments_ft


class MysteryFtExtractor:
    def __init__(self, fn='train_data/funnel.csv'):
        self._fn = fn

    def load_transform(self):
        mystery_feats = pd.read_csv(self._fn, usecols=['client_id'] + [f'feature_{i}' for i in range(1, 11)])
        mystery_feats = mystery_feats.set_index('client_id')
        mystery_feats['feature_1'] = mystery_feats['feature_1'].astype(str)

        return mystery_feats


# class BalanceFtExtractor:
#     def __init__(self, fn='train_data/balance.csv'):
#         self._fn = fn

#     def load_transform(self):
#         balances = pd.read_csv(self._fn)

#         del balances
#         gc.collect()
#         return balance_ft