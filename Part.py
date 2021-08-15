import sys
import warnings
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

if not sys.warnoptions:
    warnings.simplefilter('ignore')

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
            train, test = X[:train], X[train:]

            # predict
            predictions = []
            for i in range(len(test)):
                model = ARIMA(train, order=order)
                model_fit = model.fit()
                yhat = model_fit.forecast()[0]
                predictions.append(yhat)
                train = np.append(train, test[i])
            error = mean_squared_error(test, predictions)
            return error, predictions

        p_vals = range(6)
        d_vals = range(6)
        q_vals = range(6)

        best_score, best_order, best_preds = float('inf'), None, None

        for p in p_vals:
            for d in d_vals:
                for q in q_vals:
                    order = (p, d, q)
                    try:
                        mse, preds = evaluate_arima(dat.values, order)
                        if mse < best_score:
                            best_score, best_order, best_preds = mse, order, preds
                    except:
                        continue

        temp = dat.values
        forecasts = {}

        for i in range(5):
            model = ARIMA(temp, order=best_order)
            model_fit = model.fit()
            forecasts[model_fit.forecast()[0]] = model_fit.get_forecast().conf_int()[0]
            temp = np.append(temp, model_fit.forecast()[0])

        return forecasts, best_preds, best_order

    def plot(self, forecasts, preds):
        pred_idx = self.data.iloc[-len(preds):].index
        forecast_idx = pd.to_datetime([pred_idx[-1].date() + relativedelta(months=i+1) for i in range(len(forecasts))])
        idx = pred_idx.append(forecast_idx)
        all_preds = preds + [key for key in forecasts]

        return go.Scatter(x=idx, y=all_preds, name='Forecast')

