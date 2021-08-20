import re
import pandas as pd
from datetime import datetime

data = pd.read_excel('~/OneDrive/Consumption file/Consume + Ship Data - RV edits.xlsx', sheet_name='Main')
data = data.loc[data.PartNumber.notnull()]
data.dropna(axis=1, how='all', inplace=True)
data.set_index('PartNumber', inplace=True)

dates = []
for col in data.columns:
    if isinstance(col, datetime):
        dates.append(col)

use_only = data[dates].T


class Cross:
    def __init__(self, cross_name):
        self.name = cross_name
        self.parts = list(data.loc[data.Cross == self.name].index)

    def get_multiplier(self):

        if re.search('/\d+$', self.name):
            units = int(self.name.split('/')[1])
        else:
            units = 1

        for part in self.parts:
            if re.search('/\d+$', part):
                data.loc[part, 'multiplier'] = int(part.split('/')[1]) / units
            else:
                data.loc[part, 'multiplier'] = 1 / units

        return data.loc[self.parts, 'multiplier']

    def nonzero(self):
        cross = (self.get_multiplier() * use_only[self.parts]).sum(axis=1)
        return cross.iloc[cross.to_numpy().nonzero()[0].min():]
