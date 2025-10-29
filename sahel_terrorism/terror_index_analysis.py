
# Import packages
import os
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

# Load Global Terrorism Database
terror_db = pd.read_csv("globalterrorismdb_0522dist (1).xlsx")

# Evaluate
terror_db.info()
terror_db.columns.to_list()

def colinfo(column):
    print(terror_db[column].dtype)
    print(terror_db[column].head(5))
    print('\nPercent Null: ',terror_db[column].isnull().mean())
    print()

for i in terror_db:
    colinfo(i)
# %
# approxdate 93% null
# extended dont know what it is
# resolution 98%, not clear
# city .2% null
# latitude 2% 
# longitude 2%
# specificity .01%, not clear
# vicinity not clear
# location 68%, not clear
# summary 32%, not clear
# crit1 not clear
# crit2 not clear
# crit3 not clear
# doubtterr not clear
# alternative 83%, not clear
# alternative_txt ^
# multiple .01%, not clear
# attacktype2 96%
# attacktype2_txt ^
# attacktype3 99%
# attacktype3_txt ^
# targetsubtype1 5%
# targetsubtype1_txt ^
# %

data = terror_db 

