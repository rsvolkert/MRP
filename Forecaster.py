import sys
import time
import pandas as pd
import multiprocessing as mp
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
        self.orders = pd.DataFrame(index=self.dat.columns, columns=['p', 'd', 'q'])

        self.max_mo = self.dat.index.max()

    def forecast(self):
        total_len = len(self.dat.columns)
        processed = 0
        for part_num in self.dat.columns:
            processed += 1
            print(f'Forecasting {part_num}. Part {processed} of {total_len}.')

            start = time.time()
            part = Part(self.dat, part_num)
            forecasts, predictions, error = part.forecast(self.months, part.nonzero())
            self.forecasts[part_num] = forecasts
            self.errors.loc[part_num] = error / part.nonzero().mean()

            dates = [self.max_mo - relativedelta(months=i) for i in range(len(predictions))]
            if min(dates) not in self.predictions.index:
                self.predictions = self.predictions.reindex(dates)
            self.predictions.loc[dates, part_num] = predictions
            end = time.time()
            print(f'Forecast completed in {(end - start):.2f} seconds.')
        self.forecasts[self.forecasts < 0] = 0
        self.forecasts[self.forecasts.isnull()] = 0
        self.predictions[self.predictions < 0] = 0
        self.predictions[self.predictions.isnull()] = 0

    def to_excel(self):
        forecasts = pd.read_excel('../Analysis Data.xlsx', sheet_name='Forecasts', index_col=0).T
        predictions = pd.read_excel('../Analysis Data.xlsx', sheet_name='Predictions', index_col=0)
        errors = pd.read_excel('../Analysis Data.xlsx', sheet_name='Errors', index_col=0)

        forecasts.index = pd.to_datetime(forecasts.index).strftime('%Y-%m-%d')
        predictions.index = pd.to_datetime(predictions.index).strftime('%Y-%m-%d')

        self.forecasts.index = pd.to_datetime(self.forecasts.index).strftime('%Y-%m-%d')
        self.predictions.index = pd.to_datetime(self.predictions.index).strftime('%Y-%m-%d')

        if forecasts.index.min() == self.forecasts.index.min():
            new_forecasts = self.forecasts.columns
            joint_forecasts = [pn in forecasts.columns for pn in new_forecasts]
            if list(new_forecasts[joint_forecasts]):
                forecasts.drop(new_forecasts[joint_forecasts], axis=1, inplace=True)
            forecasts = pd.concat([self.forecasts, forecasts], axis=1)

            new_predictions = self.predictions.columns
            joint_predictions = [pn in predictions.columns for pn in new_predictions]
            if list(new_predictions[joint_predictions]):
                predictions.drop(new_predictions[joint_predictions], axis=1, inplace=True)
            predictions = pd.concat([self.predictions, predictions], axis=1)

            new_errors = self.errors.index
            joint_errors = [pn in errors.index for pn in new_errors]
            if list(new_errors[joint_errors]):
                errors.drop(new_errors[joint_errors], axis=0, inplace=True)
            errors = pd.concat([self.errors, errors], axis=0)
        else:
            forecasts = self.forecasts
            predictions = self.predictions
            errors = self.errors

        with pd.ExcelWriter('../Analysis Data.xlsx', mode='a', if_sheet_exists='replace') as writer:
            forecasts.T.to_excel(writer, sheet_name='Forecasts')
            predictions.sort_index().to_excel(writer, sheet_name='Predictions')
            errors.to_excel(writer, sheet_name='Errors')


if __name__ == '__main__':
    mp.freeze_support()

    god_response = 'y'
    while god_response == 'y':
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

        categories = pd.read_excel('../Analysis Data.xlsx', sheet_name='Categories', index_col=0)
        disc = categories.loc[categories['Sales category'] == 'Disc'].index

        pns = [pn not in disc for pn in use_only.columns]
        pns = use_only.columns[pns]
        use_only = use_only[pns]

        response = input("Do you want to run all parts? (y/n)")

        if response.lower() == 'n':
            cats = input('\nType categories separated by commas.')
            cats = cats.lower()
            cats = cats.split(',')
            cats = [cat.strip() for cat in cats]

            parts = []
            for cat in cats:
                if cat not in categories['Sales category'].str.lower().values:
                    print('There were invalid categories. Please try again.')
                    continue
                parts.extend(list(categories.loc[categories['Sales category'].str.lower() == cat].index))

            print('\nYou have selected:')
            for cat in cats:
                print(cat)
            print(f'Totalling {len(parts)} parts.')
            cat_response = input('Is this correct? (y/n)')

            while cat_response.lower() == 'n':
                cats = input('Type categories separated by commas.')
                cats = cats.lower()
                cats = cats.split(',')
                cats = [cat.strip() for cat in cats]

                parts = []
                for cat in cats:
                    if cat not in categories['Sales category'].str.lower().values:
                        print('There were invalid categories. Please try again.')
                        continue
                    parts.extend(list(categories.loc[categories['Sales category'].str.lower() == cat].index))

                print('You have selected:')
                for cat in cats:
                    print(cat)
                print(f'Totalling {len(parts)} parts.')
                cat_response = input('Is this correct? (y/n)')

            forecaster = Forecaster(use_only[parts], 6)
        else:
            forecaster = Forecaster(use_only, 6)
        start = time.time()
        forecaster.forecast()
        end = time.time()
        print(f'\nTotal elapsed time: {(end - start) / 60:.2f} minutes')
        input('Completed forecasting. Writing to Excel. Have you closed the Excel file? (press Enter to continue)')
        try:
            forecaster.to_excel()
        except:
            print('There was an error writing to excel.')
        god_response = input('\nWould you like to run more parts? (y/n)\n')

    input('Finished forecasting. Press Enter to exit.')

