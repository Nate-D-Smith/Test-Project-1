import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint, uniform

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBRegressor
from xgboost import plot_importance
import pickle

import yfinance as yf
from datetime import datetime
import time

df = pd.read_csv('Trendy Threads Sales/Data/trendy_threads_historical_sales.csv')
df.info()

# Add date column
df['date'] = pd.to_datetime(
    df['Month'] + ' ' + df['Year'].astype(str), 
    format='%B %Y')
df.info()

# Add WINA (franchiser) stock value
wina = yf.download('WINA', start='2012-05-01')
wina.columns = wina.columns.get_level_values(0)
wina_monthly = wina['Close'].resample('MS').first().reset_index()
wina_monthly.head(5)
wina_monthly = wina_monthly.rename(columns={'Close': 'wina_price', 'Date': 'date'})
df = df.merge(wina_monthly[['date', 'wina_price']], on='date', how='left')
df.info()

# Add unemployment data for years of operation
unemployment = pd.read_csv("Trendy Threads Sales/Data/maricopa_county_unemployment.csv", parse_dates=["observation_date"])
unemployment = unemployment.rename(columns={"observation_date":"date","AZMARI3URN":"unemployment"})
df = df.merge(unemployment[['date', 'unemployment']])
df.head()
df.info()

# Add consumer price index (CPI)
cpi = pd.read_csv('Trendy Threads Sales/Data/CPI_national.csv')
cpi['observation_date'] = pd.to_datetime(cpi['observation_date'])
cpi = cpi.rename(columns={'observation_date': 'date', 'CPIAUCSL': 'cpi_usa'})
df = df.merge(cpi[['date', 'cpi_usa']])

# Add inflation-adjusted WINA stock price
df['real_wina'] = df['wina_price'] / df['cpi_usa']

# Add rolling average sales columns
df['last_month'] = df['sales'].shift(1)
df['2mo_rolling'] = df['sales'].shift(1).rolling(window=2).mean()
df['3mo_rolling'] = df['sales'].shift(1).rolling(window=3).mean()
df['4mo_rolling'] = df['sales'].shift(1).rolling(window=4).mean()
df['5mo_rolling'] = df['sales'].shift(1).rolling(window=5).mean()

# Check best rolling average correlation to sales
df[['sales', 'last_month', '2mo_rolling', '3mo_rolling', '4mo_rolling', '5mo_rolling']].corr()

# Remove '$' and ',' from 'Sales'.
df['Sales'] = df['Sales'].str.replace(r'[$,]', '', regex=True)
df.head()

# Change 'Sales' data type to float.
df['Sales'] = df['Sales'].astype(float)
df.dtypes

# Standardize column names
df = df.rename(columns={
    'Year': 'year',
    'Month': 'month',
    'Sales': 'sales'
})
df.columns

# Add column with months as integers
df['month_int'] = df['month'].map({'January': 1, 'February': 2, 'March': 3, 'April': 4, 
                                   'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 
                                   'October':10, 'November': 11, 'December': 12})

sns.pairplot(df)
plt.show()
correlations = df.corr(numeric_only=True)
correlations

mask = df['year'] == 2019
max = df.loc[mask, 'sales'].idxmax()
df.loc[max]

### Create a model using only month and year to predict sales ###
df.columns
# Split train and test data.
y = df['sales']
X = df[['month_int', 'year']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

# Create and fit the model.
rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=3,
                           min_samples_leaf=3, min_samples_split=4)
rf.fit(X_train, y_train)

# Test the model
rf_y_pred = rf.predict(X_test)
rf_r2 = r2_score(y_test, rf_y_pred)
rf_error = mean_absolute_error(y_test, rf_y_pred)
rf_percent_error = mean_absolute_percentage_error(y_test, rf_y_pred)
rf_norm_error = rf_error / np.std(df['sales'])

# View model utility
print(f'\nR2: {rf_r2:.3f}\n'
      f'Error in dollars: ${rf_error:,.3f}\n'
      f'Error in percent: {rf_percent_error*100:.3f}%\n'
      f'Normalized Error: {rf_norm_error:.3}')

plt.figure(figsize=(20, 10))
plot_tree(
    rf.estimators_[0], 
    feature_names=X_train.columns,
    filled=True,
    rounded=True,
    fontsize=8
    )
plt.show()

### Test Perameters and features for an XGBoost model ###

# Test perameters
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'max_depth': randint(3, 8),
    'min_child_weight': randint(1, 6),
    'learning_rate': uniform(0.01, 0.2),
    'n_estimators': randint(20, 200),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4)
}

xgb_rand = RandomizedSearchCV(
    XGBRegressor(objective='reg:squarederror', random_state=42),
    param_distributions=param_dist,
    n_iter=25,
    scoring='r2',
    cv=5,
    random_state=42)

xgb_rand.fit(X_train, y_train)
print("Best parameters found:", xgb_rand.best_params_)
print("Best R² (CV):", xgb_rand.best_score_)

xgb_best = xgb_rand.best_estimator_

y_pred = xgb_best.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
norm_error = mae / np.std(df['sales'])

print(f"Test R²: {r2:.3f}")
print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"MAPE: {mape*100:.2f}%")
print(f"Normalized Error: {norm_error:.3f}")

# Test X values

feature_sets = [
    ['year', 'month_int', 'real_wina'],
    ['year', 'month_int', 'real_wina', 'last_month'],
    ['year', 'month_int', 'real_wina', '2mo_rolling'],
    ['year', 'month_int', 'real_wina', 'last_month', '2mo_rolling'],
    ['year', 'month_int', 'real_wina', 'last_month', '2mo_rolling', '4mo_rolling', '5mo_rolling']
]

param_dist = {
    'max_depth': randint(3, 8),
    'min_child_weight': randint(1, 6),
    'learning_rate': uniform(0.01, 0.2),
    'n_estimators': randint(20, 200),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4)
}

# Loop through each feature set
results = []
for features in feature_sets:
    print(f"\nTesting feature set: {features}")
    X = df[features]
    y = df['sales']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    xgb_rand = RandomizedSearchCV(
        XGBRegressor(objective='reg:squarederror', random_state=42),
        param_distributions=param_dist,
        n_iter=25,
        scoring='r2',
        cv=5,
        random_state=42
    )

    xgb_rand.fit(X_train, y_train)

    y_pred = xgb_rand.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    results.append({
        'features': features,
        'cv_r2': xgb_rand.best_score_,
        'test_r2': r2,
        'mae': mae,
        'mape': mape,
        'best_params': xgb_rand.best_params_
    })

# Show best feature set
results_df = pd.DataFrame(results)
results_df_sorted = results_df.sort_values(by='test_r2', ascending=False)
plt.figure(figsize=(8,4))
plt.barh(range(len(results_df_sorted)), results_df_sorted['test_r2'], align='center')
plt.yticks(range(len(results_df_sorted)), [', '.join(fs) for fs in results_df_sorted['features']])
plt.show()

# Create an XGBoost model using best perameters and features
df.columns
y = df['sales']
X = df[['year', 'month_int', 'real_wina', 'last_month', '2mo_rolling']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

cv_params = {'max_depth': [4,5],
             'min_child_weight': [3,5],
             'learning_rate': [.1],
             'n_estimators': [5,10,15],
             'subsample': [.7],
             'colsample_bytree': [.7]}

scoring =   {'r2': 'r2',
            'mae': 'neg_mean_absolute_error',
            'rmse': 'neg_root_mean_squared_error'}

xgb_cv = GridSearchCV(xgb, 
                      cv_params,
                      scoring=scoring,
                      cv=5,
                      refit='r2')


xgb_cv = xgb_cv.fit(X_train, y_train)

print("Best parameters:", xgb_cv.best_params_)
print("Best R² (CV):", xgb_cv.best_score_)

xgb_y_pred = xgb_cv.predict(X_test)
xgb_r2 = r2_score(y_test, xgb_y_pred)
xgb_error = mean_absolute_error(y_test, xgb_y_pred)
xgb_percent_error = mean_absolute_percentage_error(y_test, xgb_y_pred)
xgb_norm_error = xgb_error / np.std(df['sales'])

print(f'\nR2: {xgb_r2:.3f}\n'
      f'Error in dollars: ${xgb_error:,.3f}\n'
      f'Error in percent: {xgb_percent_error*100:.3f}%\n'
      f'Normalized Error: {xgb_norm_error:.3}')

best_r2_dif = xgb_cv.best_score_ - xgb_r2
if  best_r2_dif > -0.05 and best_r2_dif < 0.05:
    print('The model does not over fit the data.')
else:
    print('The model may overfit the data.')
best_r2_dif