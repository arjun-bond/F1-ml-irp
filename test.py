import pandas as pd
df = pd.read_csv('f1_data.csv')


grouped = df.sort_values(by=['RoundNumber']).groupby('DriverNumber'['Position'].shift(1))
print(grouped.rolling(5).mean())