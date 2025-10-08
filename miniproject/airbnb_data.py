

# This dataset from Airbnb includes various information about AirBnb listings. I want to 
# use it to help in the following hypothetical situation.
#
# I want to go on a vacation with my family in either Europe, or Asia. I want to choose a
# location for the trip based on multiple variables. My family includes me, my wife, my
# parents and my 3 siblings for a total of 6 people needing 4 beds and 3 rooms. We will be
# moving from place to place within a city every few days without prior notice so we 
# will need hosts who are likely to accept us and respond to us quickly. We are traveling 
# on a budget so the prices of the listings is imortant to us. The steps I will follow are 
# clean the dataset, do exploratory analysis, and present my findings.

# import packages
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# import csv and create pandas dataframe
bnb = pd.read_csv('Airbnb_site_hotel_new.csv')

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

# Relationship between bed accommodation number and price...

sns.scatterplot(data=bnb, x='accommodates', y='price', s=10, alpha=.8)
plt.yscale('log')
plt.show()


# Bed count and price



# Bath count and price...



# Acceptance rate and price...



# Response rate and price...



# I only want listings that meet my requirements of 





