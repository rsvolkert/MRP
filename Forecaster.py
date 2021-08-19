import pandas as pd
from Part import Part
from datetime import datetime
from dateutil.relativedelta import relativedelta


class Forecaster:
    def __init__(self, dat, months):
        self.dat = dat
        self.months = months
        idx = [dat.index[-1] + relativedelta(months=i+1) for i in range(self.months)]

        self.forecasts = pd.DataFrame(index=idx)
        self.predictions = pd.DataFrame()
        self.errors = pd.Series(index=dat.columns)

        self.max_mo = self.dat.index.max()

    def forecast(self):
        for part_num in self.dat.columns:
            part = Part(part_num)
            forecasts, predictions, error = part.forecast(self.months, part.nonzero())
            self.forecasts[part_num] = forecasts
            self.errors.loc[part_num] = error

            dates = [self.max_mo - relativedelta(months=i) for i in range(len(predictions))]
            if min(dates) not in self.predictions.index:
                self.predictions = self.predictions.reindex(dates)
            self.predictions.loc[dates, part_num] = predictions


if __name__ == '__main__':
    data = pd.read_excel('~/OneDrive/Consumption file/Consume + Ship Data - RV edits.xlsx', sheet_name='Main')
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

    forecaster = Forecaster(use_only, 5)
    forecaster.forecast()

    writer = pd.ExcelWriter('Analysis Data.xlsx')

    forecaster.forecasts.to_excel(writer, sheet_name='Forecasts')
    forecaster.predictions.sort_index().to_excel(writer, sheet_name='Predictions')
    forecaster.errors.to_excel(writer, sheet_name='Errors')

    writer.save()