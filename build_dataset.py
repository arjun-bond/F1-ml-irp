import fastf1
import pandas as pd
import time

# Use cache
fastf1.Cache.enable_cache('/Users/arjunkashyap/PycharmProjects/f1-irp/cache')
all_races = []

# Fetches data from cache / internet and adds podium indicator as well as name of GP and year
for year in range(2022,2025):
    season = fastf1.get_event_schedule(year)
    for index, row in season.iterrows():
        if row['EventFormat'] == 'testing':
            continue
        event = fastf1.get_event(year, row['RoundNumber']) #get the event
        race = event.get_race()
        quali = event.get_qualifying()
        race.load()
        quali.load()

        #account for weather
        avg_air_temp = race.weather_data['AirTemp'].mean()
        avg_humidity = race.weather_data['Humidity'].mean()
        hasRained = race.weather_data['Rainfall'].any()



        # assimilate quali and race results
        quali_slim = quali.results[['DriverNumber', 'Position']].rename(columns={'Position': 'QualiPosition'})
        merged = pd.merge(race.results, quali_slim, on='DriverNumber') # make a df with quali and race pos
        merged['Podium'] = (merged['Position'] <= 3).astype(int) # Create column to indicate podium positions
        merged['RoundNumber'] = (row['RoundNumber'])
        merged['Year'] = year
        merged['EventName'] = row['EventName']
        merged['AirTemp'] = avg_air_temp
        merged['Humidity'] = avg_humidity
        merged['Rained'] = hasRained


        all_races.append(merged)

full_dataset = pd.concat(all_races, ignore_index=True)
full_dataset.to_csv('f1_data.csv', index=False)