

# This dataset from Airbnb includes various information about AirBnb listings. I want to 
# use it to help in the following hypothetical situation.

# For each city get a correlation coefficient for each graph on pairplot.
# Create new tables with p-acc, p-beds, p-rooms etc with rows named after cities.
# Find city with weekest correlation between each and price.
# Bar chart comparing cities with lowest correlation for a perameter.
# Pick city that has most.
# Company convention planner wants to decide which city to use.
# Teams of 6 staying together.

# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# import csv and create pandas dataframe
bnb = pd.read_csv('Airbnb/Airbnb_site_hotel_new.csv')

# First, I will clean the dataset.

# review dataframe
bnb.info()

# There are 23 columns with 86,186 entries.
# Columns 'price', 'reply time', 'host Certification', and 'consumer' have missing values.
# Columns 'price', 'bathrooms', 'host response rate', and 'host acceptance rate' should be 
# converted to numeric values.
# Not all column names follow the same structuring conventions

# check percentage of missing values in columns with missing values
missing_percent = (bnb.isnull().sum()/ len(bnb)*100)
print(missing_percent[missing_percent > 0])

bnb['consumer'].head(30)

# Price is an important feature and only has 6% missing values so I will drop ROWS missing 
# price values.
# Host certification and consumer are not important features so I will drop those COLUMNS.
# The meaning of the binary values in reply time is unclear so I will drop that COLUMN as well.

# drop rows with missing price values
bnb = bnb.dropna(subset=['price'])

# drop columns 'host certification', 'consumer', and 'reply time'
bnb = bnb.drop(columns=['host Certification', 'consumer', 'reply time'])
bnb.info()

# Columns 'price', 'bathrooms', 'host response rate' and 'host acceptance rate' should be 
# converted to numeric values.

# check format of columns with wrong data types
bnb[['price','bathrooms','host response rate', 'host acceptance rate']].head(10)

# It appears that 'bathrooms', 'host response rate' and 'host acceptance rate' use commas in 
# the place of decimal points.
# It wasn't clear by the head() function what was causing object Dtype for 'price'.

# find rows where price is incompatible with numeric Dtype
temp_bnb = pd.to_numeric(bnb['price'], errors='coerce') #change to numeric
non_numeric_rows = bnb[temp_bnb.isna() & bnb['price'].notna()] #find N/A values
non_numeric_rows['price'].head()

# It is now clear that 'price' also uses commas in the place of decimal points.

# replace commas with periods and convert Dtype to numeric for 'price', 'bathrooms', 
# 'host response rate', and 'host acceptance rate'
bnb[['price','bathrooms','host response rate', 'host acceptance rate']] = (
    bnb[['price','bathrooms','host response rate', 'host acceptance rate']]
    .replace(',','.',regex=True) #replace commas
    .apply(pd.to_numeric,errors='coerce')) #convert to numeric

bnb[['price','bathrooms','host response rate', 'host acceptance rate']].head(20) #confirm change
bnb[['price','bathrooms','host response rate', 'host acceptance rate']].info() #confirm float Dtype

bnb.info()

# some columns don't follow normal column naming conventions

# normalize column names
bnb.columns = bnb.columns.str.replace(' ','_')
bnb.columns = bnb.columns.str.replace('favourite','favorite') #Americanize spelling 
bnb.columns

# After cleaning the data, there are 20 columns with 80,762 entries
# and all numeric columns are int and float Dtype.

# Next, I will do Exploratory Data Analysis

bnb.select_dtypes(include='number').describe().T #numeric structure and stats
bnb.select_dtypes(exclude='number').describe().T #categorical structure and stats

# Insights:

# 'area' includes 3 areas
bnb.area.unique()
# Areas included are ['North America', 'Europe', 'Asia']

# 'city' includes 11 cities
bnb.city.unique()
bnb.city = bnb['city'].str.title() #standardize capitalization
bnb.city.unique()
# Cities included are ['Toronto' 'Newyork' 'Amsterdam' 'Berlin' 'Dublin' 'Hongkong' 'Munich', 
# 'Singapore' 'Sydney' 'Tokyo' 'Taipei']

# Relationship analysis:

# Remove outliers for visualizations
Q1 = bnb['price'].quantile(.25)
Q3 = bnb['price'].quantile(.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
bnb_inter_quant = bnb[(bnb['price'] >= lower) & (bnb['price'] <= upper)]

# Visualize price by area
plt.figure(figsize=(10,6))
sns.boxplot(data=bnb_inter_quant, x='area', y='price')
plt.title('Price Distribution by Area')
plt.show()

# Price by city
plt.figure(figsize=(14,6))
sns.boxplot(data=bnb_inter_quant, x='city', y='price')
plt.title('Price Distribution by City')
plt.xticks(rotation=45)
plt.show()

# Asia tends to be the cheapest region
# Tokyo, Taipei, and Berlin are the cheapest coountries

# Visualize relationships between variables
# bnb_pair = sns.pairplot(bnb, vars=['price', 'accommodates', 
#                                    'bathrooms', 'bedrooms', 'beds',
#                                    'host_response_rate'])
# plt.show()

# Find correlation coefficients.
corr = bnb[['price', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 
           'host_response_rate']].corr(numeric_only=True)['price'].drop('price')
print(corr)

# 'accommodates' and 'bedrooms' have the strongest correlation with price.
# 'beds' and 'bathrooms' have the weakest.
# This means we may be able to get more beds and bathrooms for a better price.

# Evaluate correlations by country.
corr = bnb[['price', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 
           'host_response_rate']].corr(numeric_only=True)['price'].drop('price')

for i in bnb.city.unique():
    corr = bnb[bnb['city'] == i ][['price', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 
           'host_response_rate']].corr(numeric_only=True)['price'].drop('price').dropna()
    print(f'{i}:\n{corr}\n')





# It could be good to use R squared to see how much the price of a room is explained by the bed count, response time etc.