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

    def to_excel(self):
        forecasts = pd.read_excel('Analysis Data.xlsx', sheet_name='Forecasts', index_col=0)
        predictions = pd.read_excel('Analysis Data.xlsx', sheet_name='Predictions', index_col=0)
        errors = pd.read_excel('Analysis Data.xlsx', sheet_name='Errors', index_col=0)

        if (forecasts.index == self.forecasts.index).all():
            new_forecasts = self.forecasts.columns
            joint_forecasts = [pn in forecasts.columns for pn in new_forecasts]
            if list(new_forecasts[joint_forecasts]):
                forecasts.drop(new_forecasts[joint_forecasts], axis=1, inplace=True)
            self.forecasts = pd.concat([self.forecasts, forecasts], axis=1)

            new_predictions = self.predictions.columns
            joint_predictions = [pn in predictions.columns for pn in new_predictions]
            if list(new_predictions[joint_predictions]):
                predictions.drop(new_predictions[joint_predictions], axis=1, inplace=True)
            self.predictions = pd.concat([self.predictions, predictions], axis=1)

            new_errors = self.errors.index
            joint_errors = [pn in errors.index for pn in new_errors]
            if list(new_errors[joint_errors]):
                errors.drop(new_errors[joint_errors], axis=0, inplace=True)
            self.errors = pd.concat([self.errors, errors], axis=0)

        with pd.ExcelWriter('Analysis Data.xlsx', mode='a', if_sheet_exists='replace') as writer:
            self.forecasts.to_excel(writer, sheet_name='Forecasts')
            self.predicitons.sort_index().to_excel(writer, sheet_name='Predictions')
            self.errors.to_excel(writer, sheet_name='Errors')
