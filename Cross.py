import re
import pandas as pd
from datetime import datetime

data = pd.read_excel('../Analysis Data.xlsx', sheet_name='Cross')
data = data.loc[data.PartNumber.notnull()]
data.dropna(axis=1, how='all', inplace=True)
data.set_index('PartNumber', inplace=True)
data.loc[data.Cross == 0, 'Cross'] = data.loc[data.Cross == 0].index.values

dates = []
for col in data.columns:
    if isinstance(col, datetime):
        dates.append(col)

use_only = data[dates].T


class Cross:
    def __init__(self, cross_name):
        self.name = cross_name

        if self.name not in data.Cross.values:
            self.parts = [self.name]
        elif len(data.loc[self.name]) == 1:
            self.parts = [self.name]
        else:
            self.parts = list(data.loc[data.Cross == self.name].index)

    def get_multiplier(self):

        if re.search('/\d+$', self.name):
            splits = self.name.split('/')
            for split in splits:
                try:
                    units = int(split)
                    break
                except:
                    continue
        else:
            units = 1

        for part in self.parts:
            if re.search('/\d+$', part):
                splits = part.split('/')
                for split in splits:
                    try:
                        data.loc[part, 'multiplier'] = int(split) / units
                        break
                    except:
                        continue
            else:
                data.loc[part, 'multiplier'] = 1 / units

        return data.loc[self.parts, 'multiplier']

    def nonzero(self):
        cross = (self.get_multiplier() * use_only[self.parts]).sum(axis=1)
        return cross.iloc[cross.to_numpy().nonzero()[0].min():]
