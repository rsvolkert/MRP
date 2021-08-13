import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

data = pd.read_excel('~/OneDrive/Consumption file/Consume + Ship Data - RV edits.xlsx', sheet_name='Main')
data = data.loc[data.PartNumber.notnull()]
data.dropna(axis=1, how='all', inplace=True)
data.set_index('PartNumber', inplace=True)

dates = []
for col in data.columns:
    if isinstance(col, datetime):
        dates.append(col)

use_only = data[dates].T


class Part:
    def __init__(self, part_num):
        self.part = part_num
        self.data = use_only[self.part]

    def nonzero(self):
        return self.data.iloc[self.data.to_numpy().nonzero()[0].min():]

    def forecast(self, dat=None):
        if dat is None:
            dat = self.data

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

