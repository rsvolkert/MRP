import sys
import warnings
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error as mae

if not sys.warnoptions:
    warnings.simplefilter('ignore')


def evaluate_arima(train, test, order):
    # predict
    predictions = []
    for i in range(len(test)):
        try:
            model = ARIMA(train, order=order, missing='drop', enforce_stationarity=False)
            model_fit = model.fit()
        except:
            return {float('inf'): float('inf')}
        yhat = model_fit.forecast()[0] if model_fit.forecast()[0] > 0 else 0

        predictions.append(yhat)
        train = np.append(train, test[i])
    err = mae(test, predictions)
    return {err: [predictions, order]}
