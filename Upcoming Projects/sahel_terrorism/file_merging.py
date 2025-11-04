import pandas as pd
import numpy as np

df = pd.read_excel('c:/Users/nates/OneDrive/BYUI-NateHP/DATA PROJECTS/globalterrorismdb_0522dist (1).xlsx')
df2022 = pd.read_excel('c:/Users/nates/OneDrive/BYUI-NateHP/DATA PROJECTS/globalterrorismdb_2021Jan-June_1222dist.xlsx')

df.columns
df.country_txt.head(20)

df_new = df[df['country_txt'].isin(['Burkina Faso', 'Niger', 'Mali', 'Nigeria', 'Cameroon'])]

df_new.country_txt.value_counts()
