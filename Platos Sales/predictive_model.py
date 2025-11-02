import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

import yfinance as yf
from datetime import datetime

df = pd.read_csv('trendy_threads_historical_sales.csv')
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
unemployment = pd.read_csv("maricopa_county_unemployment.csv", parse_dates=["observation_date"])
unemployment = unemployment.rename(columns={"observation_date":"date","AZMARI3URN":"unemployment"})
df = df.merge(unemployment[['date', 'unemployment']])
df.head()
df.info()

# Add consumer price index (CPI)
cpi = pd.read_csv('CPI_national.csv')
cpi['observation_date'] = pd.to_datetime(cpi['observation_date'])
cpi = cpi.rename(columns={'observation_date': 'date', 'CPIAUCSL': 'cpi_usa'})
df = df.merge(cpi[['date', 'cpi_usa']])

# Add inflation-adjusted WINA stock price
df['real_wina'] = df['wina_price'] / df['cpi_usa']
df.info()

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

### Create a model using only month and year to predict sales ###
df.columns
# Split train and test data.
y = df['sales']
X = df[['cpi_usa', 'month_int', 'year']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

# Create and fit the model.
rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=3,
                           min_samples_leaf=3, min_samples_split=4)
rf.fit(X_train, y_train)

# Test the model
rf_y_pred = rf.predict(X_test)
r2 = r2_score(y_test, rf_y_pred)
error = mean_absolute_error(y_test, rf_y_pred)
percent_error = mean_absolute_percentage_error(y_test, rf_y_pred)
rf_norm_error = error / np.std(df['sales'])

# View model utility
print(f'\nR2: {r2:.2f}\n'
      f'Error in dollars: ${error:,.2f}\n'
      f'Error in percent: {percent_error*100:.2f}%\n'
      f'Normalized Error: {rf_norm_error:.2}')

plt.figure(figsize=(20, 10))
#plot_tree(
    rf.estimators_[0], 
    feature_names=X_train.columns,
    filled=True,
    rounded=True,
    fontsize=8
)
#plt.show()
