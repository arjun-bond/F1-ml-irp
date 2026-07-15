import fastf1
# FastF1 caches data locally so you're not re-downloading every run
fastf1.Cache.enable_cache('/Users/arjunkashyap/PycharmProjects/f1-irp/cache')



# get all data for races and qualifying in 2023 season
season = fastf1.get_event_schedule(2023)
print(season)
for index, row in season.iterrows():
    if row['EventFormat'] == 'testing':
        continue
    event = fastf1.get_event(2023, row['RoundNumber'])
    print(event.EventName)
    race = event.get_race()
    race.load()
    quali = event.get_qualifying()
    quali.load()
