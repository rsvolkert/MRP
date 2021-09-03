import re
import pandas as pd
from datetime import datetime

data = pd.read_excel('../Analysis Data.xlsx', sheet_name='Main')
data = data.loc[data.PartNumber.notnull()]
data.dropna(axis=1, how='all', inplace=True)
data.set_index('PartNumber', inplace=True)

dates = []
for col in data.columns:
    if isinstance(col, datetime):
        dates.append(col)

use_only = data[dates].T
use_only['Date'] = [date.date() for date in use_only.index]
use_only.reset_index(drop=True, inplace=True)
use_only.set_index('Date', inplace=True)
use_only = use_only[use_only.columns[(use_only != 0).any()]]
part_nums = use_only.columns.to_numpy()

crosses = pd.read_excel('../Analysis Data.xlsx', sheet_name='Cross', index_col=0)

categories = pd.read_excel('../Analysis Data.xlsx', sheet_name='Categories', index_col=0)
cat_idx = [pn in use_only.columns for pn in categories.index]
cat_opts = list(categories.loc[cat_idx, 'Sales category'].unique())
if 'Disc' in cat_opts:
    cat_opts.remove('Disc')
cat_opts = ['NA' if cat is np.nan else cat for cat in cat_opts]
cat_opts.sort()

# get part numbers that are not discontinued
pn_filter = [pn not in categories.loc[categories['Sales category'] == 'Disc'].index for pn in part_nums]
part_nums = list(part_nums[pn_filter])

# generate crosses to be used
crosses = crosses.loc[part_nums]
crosses.loc[crosses.Cross == 0, 'Cross'] = crosses.loc[crosses.Cross == 0].index.values


class Cross:
    def __init__(self, cross_name):
        self.name = cross_name
        self.parts = list(crosses.loc[crosses.Cross == cross_name].index)
        self.qty = crosses.loc[self.parts, 'Qty']

    def get_multiplier(self):
        units = self.qty.loc[self.name]
        return self.qty / units

    def nonzero(self):
        cross = (self.get_multiplier() * use_only[self.parts]).sum(axis=1)
        return cross.iloc[cross.to_numpy().nonzero()[0].min():]
