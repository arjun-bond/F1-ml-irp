import pandas as pd
df = pd.read_csv('f1_data.csv')

#Recent Form for each driver (5 race rolling average)
df_sorted = df.sort_values(['DriverNumber', 'Year','RoundNumber'])
df_sorted['ShiftedPosition'] = df_sorted.groupby('DriverNumber')['Position'].shift(1) #exclude current race from average

df_sorted['RecentForm'] = df_sorted.groupby('DriverNumber')['ShiftedPosition'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)

# Historical track performance
df_sorted['TrackShifted'] = df_sorted.groupby(['DriverNumber','EventName'])['Position'].shift(1) #exclude current race from average
df_sorted['TrackPerformance'] = df_sorted.groupby(['DriverNumber','EventName'])['TrackShifted'].expanding(min_periods=1).mean().reset_index(level=[0,1], drop=True)
df_sorted['TrackPerformance'] = df_sorted['TrackPerformance'].fillna(df_sorted['QualiPosition']) #Replace null values

#team performance
df_sorted['CarPerformance'] = df_sorted.groupby(['TeamId','EventName'])['QualiPosition'].transform('min')

#QualiDelta
df_sorted[['Q1_y','Q2_y','Q3_y']] = df_sorted[['Q1_y','Q2_y','Q3_y']].apply(pd.to_timedelta)
df_sorted['QualiTime'] = df_sorted[['Q1_y','Q2_y','Q3_y']].min(axis=1)
df_sorted['PolePositionTime'] = df_sorted.groupby(['EventName'])['QualiTime'].transform('min')
df_sorted['QualiDelta'] = ((df_sorted['QualiTime'] - df_sorted['PolePositionTime'])/df_sorted['PolePositionTime']) * 100

df_sorted.to_csv('f1_data_sorted.csv', index=False)