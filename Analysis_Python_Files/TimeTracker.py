import time
import pandas as pd

# used kinda like a stopwatch with split functionality
class TimeTracker:
    def __init__(self):
        self.t = {'Name':[], 'Value(s)':[], 'Delta(s)':[], 'Offset(s)':time.time() }        
        self.t['Name'].append('Initialization')
        self.t['Value(s)'].append(time.time() - self.t['Offset(s)'])
        self.t['Delta(s)'].append(0)        
    
    def clock(self, name):
        self.t['Name'].append(name)
        self.t['Value(s)'].append(time.time() - self.t['Offset(s)'])
        self.t['Delta(s)'].append(self.t['Value(s)'][-1] - self.t['Value(s)'][-2])
        
    def display(self):
        df = pd.DataFrame(self.t)
        display(df)
        