import pandas as pd
import numpy as np

from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder



train_df = pd.read_csv('train_engineered_v2.csv')
test_df = pd.read_csv('test_engineered_v2.csv')



categorical_cols = train_df.select_dtypes(include='object').columns

for col in categorical_cols:
    if col != 'UniqueID':
        le = LabelEncoder()

        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)

        combined = pd.concat([train_df[col], test_df[col]])
        le.fit(combined)

        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])



X = train_df.drop(['UniqueID', 'next_3m_txn_count'], axis=1)

# log transform 
y = np.log1p(train_df['next_3m_txn_count'])


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    train_size=0.8,
    random_state=44
)


models = {
    'RFRegressor': RandomForestRegressor(
        n_estimators=400,
        random_state=42
    ),

    'XGB': XGBRegressor(
        n_estimators=1000,
        learning_rate= 0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

best_model = None
best_score = float('inf')

for name, model in models.items():

    try:
        model.fit(X_train, y_train)
        print(f'{name} trained successfully')

        pred = model.predict(X_valid)

        pred = np.clip(pred, 0, None)

        rmse = root_mean_squared_log_error(y_valid, pred)

        print(name)
        print('RMSLE:', rmse)
        print('-' * 30)

        if rmse < best_score:
            best_score = rmse
            best_model = model

    except Exception as e:
        print(f'{name} failed: {e}')


best_model.fit(X, y)


X_test = test_df.drop(['UniqueID'], axis=1)

test_pred = best_model.predict(X_test)

# inverse log transform
test_pred = np.expm1(test_pred)

test_pred = np.clip(test_pred, 0, None)


submission = pd.read_csv('SampleSubmission.csv')

submission['next_3m_txn_count'] = test_pred

print(submission.head())

submission.to_csv('submission_v2.csv', index=False)

print('submission_v2.csv saved successfully')