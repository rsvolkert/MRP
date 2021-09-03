import sys
import warnings
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor

if not sys.warnoptions:
    warnings.simplefilter('ignore')


def arima(order, train, test):
    model = ARIMA(train, order=order, missing='drop', enforce_stationarity=False)
    model_fit = model.fit(low_memory=True)
    predictions = model_fit.predict(len(train), len(train) + len(test) - 1)
    try:
        err = mae(test, predictions)
    except ValueError:
        err = float('inf')
    return {err: [order, predictions]}


def linear(degree, train, test):
    x = np.array(range(len(train))).reshape(-1, 1)
    x_pred = np.array(range(len(train), len(test) + len(train))).reshape(-1, 1)

    poly = PolynomialFeatures(degree)
    x_poly = poly.fit_transform(x)
    x_pred_poly = poly.fit_transform(x_pred)

    model = LinearRegression()
    model_fit = model.fit(x_poly, train)
    predictions = model_fit.predict(x_pred_poly)
    err = mae(test, predictions)

    return {err: [degree, predictions]}


def svc(kernel, train, test):
    x = np.array(range(len(train))).reshape(-1, 1)
    x_pred = np.array(range(len(train), len(test) + len(train))).reshape(-1, 1)

    if kernel == 'poly':
        err = {}
        degrees = range(1, 5)
        for degree in degrees:
            model = SVR(kernel=kernel, degree=degree)
            model_fit = model.fit(x, train)
            predictions = model_fit.predict(x_pred)
            err[mae(test, predictions)] = [kernel, degree, predictions]
        return {min(err.keys()): err[min(err.keys())]}
    else:
        model = SVR(kernel=kernel)
        model_fit = model.fit(x, train)
        predictions = model_fit.predict(x_pred)
        err = mae(test, predictions)
        return {err: [kernel, predictions]}


def random_forest(train, test):
    x = np.array(range(len(train))).reshape(-1, 1)
    x_pred = np.array(range(len(train), len(test) + len(train))).reshape(-1, 1)

    model = RandomForestRegressor(n_estimators=1000, criterion='mae', random_state=400)
    model_fit = model.fit(x, train)
    predictions = model_fit.predict(x_pred)
    err = mae(test, predictions)

    return {err: predictions}


def ses(train, test):
    model = SimpleExpSmoothing(train)
    model_fit = model.fit()
    predictions = model_fit.predict(len(train), len(train) + len(test) - 1)
    err = mae(test, predictions)

    return {err: predictions}


def knn(wn, train, test):
    x = np.array(range(len(train))).reshape(-1, 1)
    x_pred = np.array(range(len(train), len(test) + len(train))).reshape(-1, 1)

    weight = wn[0]
    neighbor = wn[1]

    if neighbor > len(x):
        return {float('inf'): [weight, neighbor, []]}

    model = KNeighborsRegressor(weights=weight, n_neighbors=neighbor)
    model_fit = model.fit(x, train)
    predictions = model_fit.predict(x_pred)
    err = mae(test, predictions)

    return {err: [weight, neighbor, predictions]}
