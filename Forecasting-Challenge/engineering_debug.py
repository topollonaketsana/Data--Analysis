import pandas as pd
import numpy as np


#load the data
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

txn = pd.read_parquet('transactions_features.parquet')
fin = pd.read_parquet('financials_features.parquet')
demo = pd.read_parquet('demographics_clean.parquet')


#Transactions
txn_features = txn.groupby('UniqueID').agg({
    'TransactionAmount': [
        'sum', 'mean', 'median', 'std',
        'count', 'max', 'min'
    ],
    'StatementBalance': [
        'mean', 'median', 'std',
        'min', 'max'
    ]
})

txn_features.columns = ['_'.join(col) for col in txn_features.columns]
txn_features = txn_features.reset_index()


txn_features['amount_cv'] = (
    txn_features['TransactionAmount_std'] /
    (txn_features['TransactionAmount_mean'] + 1e-4)
)

txn_features['txn_per_balance'] = (
    txn_features['TransactionAmount_sum'] /
    (txn_features['StatementBalance_mean'] + 1)
)

txn_features['txn_intensity'] = (
    txn_features['TransactionAmount_count'] /
    (txn_features['StatementBalance_mean'] + 1)
)


# DEBIT or CREDIT BEHAVIOUR

debit_credit = pd.crosstab(
    txn['UniqueID'],
    txn['IsDebitCredit'],
    normalize='index'
).reset_index()

txn_features = txn_features.merge(debit_credit, on='UniqueID', how='left')


#

txn_features = txn_features.fillna(0)


fin_features = fin.groupby('UniqueID').agg({
    'NetInterestIncome': ['sum', 'mean', 'std'],
    'NetInterestRevenue': ['sum', 'mean'],
    'Product': 'nunique'
})

fin_features.columns = ['_'.join(col) for col in fin_features.columns]
fin_features = fin_features.reset_index()

fin_features = fin_features.fillna(fin_features.mean(numeric_only=True))



demo_features = demo[[
    'UniqueID',
    'Gender',
    'IncomeCategory',
    'AnnualGrossIncome',
    'OccupationCategory',
    'CustomerStatus'
]]

income_median = demo['AnnualGrossIncome'].median()

demo_features['AnnualGrossIncome'] = demo_features['AnnualGrossIncome'].fillna(income_median)
demo_features['Gender'] = demo_features['Gender'].fillna('Unknown')



train_df = train.merge(txn_features, on='UniqueID', how='left')
train_df = train_df.merge(fin_features, on='UniqueID', how='left')
train_df = train_df.merge(demo_features, on='UniqueID', how='left')

test_df = test.merge(txn_features, on='UniqueID', how='left')
test_df = test_df.merge(fin_features, on='UniqueID', how='left')
test_df = test_df.merge(demo_features, on='UniqueID', how='left')


# numeric columns
num_cols = train_df.select_dtypes(include=[np.number]).columns

for col in num_cols:
    if col != 'next_3m_txn_count':
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col] = test_df[col].fillna(median_val)

# categorical columns
cat_cols = train_df.select_dtypes(include='object').columns

for col in cat_cols:
    train_df[col] = train_df[col].fillna('Unknown')
    test_df[col] = test_df[col].fillna('Unknown')


# ensure no NaNs remain
assert train_df.isnull().sum().sum() == 0
assert test_df.isnull().sum().sum() == 0


train_df.to_csv('train_engineered_v2.csv', index= False)
test_df.to_csv('test_engineered_v2.csv', index= False)

print("Feature engineering complete — no missing values remaining.")