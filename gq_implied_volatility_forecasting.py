
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

# Paths
train_path = 'train/ETH.csv'
test_path = 'test/ETH.csv'
submission_path = 'submission.csv'

# Load data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sub_df = pd.read_csv(submission_path)

# Feature Engineering
train_df['bid_ask_spread'] = train_df['ask_price1'] - train_df['bid_price1']
test_df['bid_ask_spread'] = test_df['ask_price1'] - test_df['bid_price1']

train_df['obi'] = (train_df['bid_volume1'] - train_df['ask_volume1']) / (train_df['bid_volume1'] + train_df['ask_volume1'])
test_df['obi'] = (test_df['bid_volume1'] - test_df['ask_volume1']) / (test_df['bid_volume1'] + test_df['ask_volume1'])

features = ['mid_price', 'bid_price1', 'bid_volume1', 'ask_price1', 'ask_volume1', 'bid_ask_spread', 'obi']
X_train = train_df[features]
y_train = train_df['label']
X_test = test_df[features]

# Model training
lgbm = LGBMRegressor(n_estimators=1000, random_state=42)
lgbm.fit(X_train, y_train)

# Predictions
preds = lgbm.predict(X_test)

# Save submission
sub_df['labels'] = preds
sub_df.to_csv(submission_path, index=False)
print("Submission saved as", submission_path)

# RMSE on train
y_train_pred = lgbm.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print(f"Train RMSE: {rmse}")
