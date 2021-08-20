import pandas as pd
from Part import Part
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
            part = Part(self.dat, part_num)
            forecasts, predictions, error = part.forecast(self.months, part.nonzero())
            self.forecasts[part_num] = forecasts
            self.errors.loc[part_num] = error / part.nonzero().mean()

            dates = [self.max_mo - relativedelta(months=i) for i in range(len(predictions))]
            if min(dates) not in self.predictions.index:
                self.predictions = self.predictions.reindex(dates)
            self.predictions.loc[dates, part_num] = predictions

    def main(self):
        self.forecast()

        forecasts = pd.read_excel('Analysis Data.xlsx', sheet_name='Forecasts', index_col=0)
        predictions = pd.read_excel('Analysis Data.xlsx', sheet_name='Predictions', index_col=0)
        errors = pd.read_excel('Analysis Data.xlsx', sheet_name='Errors', index_col=0)

        if (forecasts.index == self.forecasts.index).all():
            self.forecasts = pd.concat([self.forecasts, forecasts], axis=1)
            self.predictions = pd.concat([self.predictions, predictions], axis=1)
            self.errors = pd.concat([self.errors, errors], axis=0)

        writer = pd.ExcelWriter('Analysis Data.xlsx')

        self.forecasts.to_excel(writer, sheet_name='Forecasts')
        self.predicitons.sort_index().to_Excel(writer, sheet_name='Predictions')
        self.errors.to_excel(writer, sheet_name='Errors')

        writer.save()
