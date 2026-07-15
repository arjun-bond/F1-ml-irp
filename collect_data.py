import fastf1
import time
# FastF1 caches data locally so you're not re-downloading every run
fastf1.Cache.enable_cache('/Users/arjunkashyap/PycharmProjects/f1-irp/cache')



# get all data for races and qualifying in 2022-2025 seasons
for year in range(2022,2026):
    season = fastf1.get_event_schedule(year)
    print(season)
    for index, row in season.iterrows():
        if row['EventFormat'] == 'testing': # remove pre and post season testing
            continue
        event = fastf1.get_event(year, row['RoundNumber'])
        print(event.EventName)
        race = event.get_race()
        race.load()
        quali = event.get_qualifying()
        quali.load() # by loading it is cached locally preventing duplicate API reqs.
        time.sleep(0.5) # sleep to prevent rate limiting
