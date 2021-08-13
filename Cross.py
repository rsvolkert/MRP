import re
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

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

        if re.search('/\d+', self.name):
            units = int(self.name.split('/')[1])
        else:
            units = 1

        for part in self.parts:
            if re.search('/', part):
                data.loc[part, 'multiplier'] = units / int(part.split('/')[1])
            else:
                data.loc[part, 'multiplier'] = units

        return data.loc[self.parts, 'multiplier']

    def nonzero(self):
        cross = (self.multiplier() * use_only[self.parts]).sum(axis=1)
        return cross.iloc[cross.to_numpy().nonzero[0].min():]

    def forecast(self, dat=None):
        if dat is None:
            dat = (self.get_multiplier() * use_only[self.parts]).sum(axis=1)

        def evaluate_arima(X, order):
            # split data
            train = int(len(X) * 0.66)
            train, test = X[0:train], X[train:]

            # predict
            predictions = []
            for i in range(len(test)):
                model = ARIMA(train, order=order)
                model_fit = model.fit()
                yhat = model_fit.forecast()[0]
                predictions.append(yhat)
                train = np.append(train, test[i])
            error = mean_squared_error(test, predictions)
            return error

        p_vals = range(11)
        d_vals = range(6)
        q_vals = range(6)

        best_score, best_order = float('inf'), None

        for p in p_vals:
            for d in d_vals:
                for q in q_vals:
                    order = (p, d, q)
                    try:
                        mse = evaluate_arima(dat.values, order)
                        if mse < best_score:
                            best_score, best_order = mse, order
                    except:
                        continue

        temp = dat.values
        forecasts = {}

        for i in range(5):
            model = ARIMA(temp, best_order)
            model_fit = model.fit()
            forecasts[model_fit.forecast()[0]] = model_fit.get_forecast().conf_int()[0]
            temp = np.append(temp, model_fit.forecast()[0])

        return forecasts

