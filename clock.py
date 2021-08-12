import pandas as pd
from datetime import datetime
import sys

now = datetime.now()
today = now.strftime('%d/%m/%y')
time = now.strftime('%H:%M:%S')

def clockin(today, time):
    zeros = '00:00:00'

    try:
        timeclock = pd.read_csv('timeclock.csv')
        timeclock.loc[len(timeclock.index)] = [today, time, zeros, zeros, zeros, zeros, zeros]
    except:
        timeclock = pd.DataFrame({'Date' : today,
                                  'In' : time,
                                  'Out' : zeros,
                                  'Hours' : 0,
                                  'Break In' : zeros,
                                  'Break Out' : zeros,
                                  'Break Hours' : 0},
                                 index=['Date'])
    return timeclock

def clockout(today, time):
    timeclock = pd.read_csv('timeclock.csv')
    timeclock.loc[timeclock.Date == today, 'Out'] = time
    clocked_in = datetime.strptime(timeclock.loc[timeclock.Date == today, 'In'].values[0], '%H:%M:%S')

    if datetime.strptime(timeclock.loc[timeclock.Date == today, 'Break In'].values[0], '%H:%M:%S') > datetime.strptime(timeclock.loc[timeclock.Date == today, 'Break Out'].values[0], '%H:%M:%S'):
        timeclock = breakout(today, time)

    break_hours = timeclock.loc[timeclock.Date == today, 'Break Hours']
    diff = (datetime.strptime(time, '%H:%M:%S') - clocked_in).total_seconds() / 3600 - break_hours
    timeclock.loc[timeclock.Date == today, 'Hours'] = diff
    return timeclock

def breakin(today, time):
    timeclock = pd.read_csv('timeclock.csv')
    timeclock.loc[timeclock.Date == today, 'Break In'] = time
    return timeclock

def breakout(today, time):
    timeclock = pd.read_csv('timeclock.csv')
    timeclock.loc[timeclock.Date == today, 'Break Out'] = time
    break_in = datetime.strptime(timeclock.loc[timeclock.Date == today, 'Break In'].values[0], '%H:%M:%S')
    diff = (datetime.strptime(time, '%H:%M:%S') - break_in).total_seconds() / 3600
    timeclock.loc[timeclock.Date == today, 'Break Hours'] = diff
    return timeclock


if __name__ == '__main__':
    if sys.argv[1] == 'clockin':
        clockin(today, time).to_csv('timeclock.csv', index=False)
        print(f'Clocked in at {time}.')
    elif sys.argv[1] == 'clockout':
        clockout(today, time).to_csv('timeclock.csv', index=False)
        print(f'Clocked out at {time}')
    elif sys.argv[1] == 'breakin':
        breakin(today, time).to_csv('timeclock.csv', index=False)
        print(f'Break at {time}')
    elif sys.argv[1] == 'breakout':
        breakout(today, time).to_csv('timeclock.csv', index=False)
        print(f'Back from break at {time}')