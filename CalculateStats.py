import pandas as pd
df = pd.read_csv('f1_data.csv')


df_sorted = df.sort_values(['DriverNumber', 'RoundNumber'])
df_sorted['ShiftedPosition'] = df_sorted.groupby('DriverNumber')['Position'].shift(1) #exclude current race from average

df_sorted['RecentForm'] = df_sorted.groupby('DriverNumber')['ShiftedPosition'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)

df_sorted['TrackShifted'] = df_sorted.groupby(['DriverNumber','EventName'])['Position'].shift(1) #exclude current race from average
df_sorted['TrackPerformance'] = df_sorted.groupby(['DriverNumber','EventName'])['TrackShifted'].expanding(min_periods=1).mean().reset_index(level=[0,1], drop=True)
df_sorted['TrackPerformance'] = df_sorted['TrackPerformance'].fillna(df_sorted['QualiPosition']) #Replace null values
df_sorted.to_csv('f1_data_sorted.csv', index=False)