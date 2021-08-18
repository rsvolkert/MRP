import sys
import warnings
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.arima.model import ARIMA
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from evaluate_arima import evaluate_arima

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
        self.cross = data.loc[part_num, 'Cross']

    def nonzero(self):
        return self.data.iloc[self.data.to_numpy().nonzero()[0].min():]

    def forecast(self, months, dat=None):
        if dat is None:
            dat = self.data

        if len(dat) == 1:
            return [dat.iloc[0,0]] * months, [], float('inf')

        train = int(len(dat) * 0.66)
        train, test = dat.values[:train], dat.values[train:]

        score = []
        try:
            executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
            tasks = (delayed(evaluate_arima)(train, test, (p,d,q)) for p in range(6) for d in range(6) for q in range(6))
            results = executor(tasks)
            score.append(results)
        except:
            print('Error')

        scores = [list(err.keys())[0] for err in score[0]]
        best = score[0][scores.index(min(scores))]

        err = list(best.keys())[0]
        preds = list(best.values())[0][0]
        order = list(best.values())[0][1]

        temp = dat.values
        forecasts = []
        for i in range(months):
            model = ARIMA(temp, order=order, missing='drop', enforce_stationarity=False)
            model_fit = model.fit()
            yhat = model_fit.forecast()[0] if model_fit.forecast()[0] > 0 else 0

            forecasts.append(yhat)
            temp = np.append(temp, yhat)

        return forecasts, preds, err

    def plot(self, forecasts, preds):
        pred_idx = self.data.iloc[-len(preds):].index
        forecast_idx = pd.to_datetime([pred_idx[-1].date() + relativedelta(months=i+1) for i in range(len(forecasts))])
        idx = pred_idx.append(forecast_idx)
        all_preds = preds + [key for key in forecasts]

        return go.Scatter(x=idx, y=all_preds, name='Forecast')
