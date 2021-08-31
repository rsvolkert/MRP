import sys
import warnings
import numpy as np
import multiprocessing as mp
from functools import partial
from statsmodels.tsa.arima.model import ARIMA
from evaluate_arima import evaluate_arima

if not sys.warnoptions:
    warnings.simplefilter('ignore')


class Part:
    def __init__(self, dat, part_num):
        self.part = part_num
        self.data = dat[self.part]

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
            orders = partial(evaluate_arima, test=test, train=train)
            with mp.Pool() as pool:
                score = pool.map(orders, [(p,d,q) for p in range(5) for d in range(4) for q in [0,2,3,4,5]])
        except:
            print('Error')

        scores = [list(err.keys())[0] for err in score]
        best = score[scores.index(min(scores))]

        err = list(best.keys())[0]
        preds = list(best.values())[0][0]
        order = list(best.values())[0][1]

        temp = dat.values
        forecasts = []
        for i in range(months):
            try:
                model = ARIMA(temp, order=order, missing='drop', enforce_stationarity=False)
                model_fit = model.fit()
            except IndexError:
                order = (order[0], order[1]-1, order[2])
                model = ARIMA(temp, order=order, missing='drop', enforce_stationarity=False)
                model_fit = model.fit()
            yhat = model_fit.forecast()[0] if model_fit.forecast()[0] > 0 else 0

            forecasts.append(yhat)
            temp = np.append(temp, yhat)

        return forecasts, preds, err, order
