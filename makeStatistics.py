import pandas as pd
df = pd.read_csv('f1_data.csv')

print(df['Rained'].count())