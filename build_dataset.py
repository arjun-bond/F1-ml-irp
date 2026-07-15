import fastf1
import pandas as pd

# FastF1 caches data locally so you're not re-downloading every run
fastf1.Cache.enable_cache('/Users/arjunkashyap/PycharmProjects/f1-irp/cache')

season = fastf1.get_event_schedule(2023)
event = fastf1.get_event(2023, 1)
race = event.get_race()
quali = event.get_qualifying()
race.load()
quali.load()
quali_slim = quali.results[['DriverNumber', 'Position']].rename(columns={'Position': 'QualiPosition'})
merged = pd.merge(race.results, quali_slim, on='DriverNumber')

merged['Podium'] = (merged['Position'] <= 3).astype(int)

print(merged[['Position', 'Podium']])
