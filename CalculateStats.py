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

# Drivers' championship standing going into each race (for the historical baseline)
points_map = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}
df_sorted['RacePoints'] = df_sorted['Position'].map(points_map).fillna(0)
df_sorted['CumulativePoints'] = df_sorted.groupby(['DriverNumber', 'Year'])['RacePoints'].cumsum()
df_sorted['StandingsPointsBeforeRace'] = df_sorted.groupby(['DriverNumber', 'Year'])['CumulativePoints'].shift(1).fillna(0)
df_sorted['ChampionshipRank'] = df_sorted.groupby(['Year', 'RoundNumber'])['StandingsPointsBeforeRace'].rank(ascending=False, method='first')

df_sorted.to_csv('f1_data_sorted.csv', index=False)