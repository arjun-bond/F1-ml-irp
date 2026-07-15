import fastf1
import pandas as pd

# FastF1 caches data locally so you're not re-downloading every run
fastf1.Cache.enable_cache('/Users/arjunkashyap/PycharmProjects/f1-irp/cache')
all_races = []


season = fastf1.get_event_schedule(2023)
for index, row in season.iterrows():
    if row['EventFormat'] == 'testing':
        continue
    event = fastf1.get_event(2023, row['RoundNumber'])
    race = event.get_race()
    quali = event.get_qualifying()
    race.load()
    quali.load()
    quali_slim = quali.results[['DriverNumber', 'Position']].rename(columns={'Position': 'QualiPosition'})
    merged = pd.merge(race.results, quali_slim, on='DriverNumber')
    merged['Podium'] = (merged['Position'] <= 3).astype(int)
    merged['RoundNumber'] = (row['RoundNumber'])
    merged['Year'] = 2023
    merged['EventName'] = row['EventName']
    all_races.append(merged)


full_dataset = pd.concat(all_races, ignore_index=True)
full_dataset.to_csv('f1_data.csv', index=False)