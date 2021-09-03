import sys
import warnings
import numpy as np
import pandas as pd
import multiprocessing as mp
import evaluations as evals
from functools import partial
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

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

        train = int(len(dat) * 0.75)
        train, test = dat.values[:train], dat.values[train:]

        if len(train) <= 1:
            return [dat.iloc[0]] * months, [dat.iloc[0]], np.nan

        with mp.Pool() as pool:
            orders = partial(evals.arima, train=train, test=test)
            arimas = pool.map(orders, [(p,0,q) for p in range(6) for q in range(6)])
            degrees = partial(evals.linear, train=train, test=test)
            linears = pool.map(degrees, [degree for degree in range(6)])
            kernels = partial(evals.svc, train=train, test=test)
            svcs = pool.map(kernels, ['linear', 'poly', 'rbf', 'sigmoid'])
            weights = partial(evals.knn, train=train, test=test)
            knns = pool.map(weights, [(w, n) for w in ['uniform', 'distance'] for n in range(1, 11)])

        arima_scores = [list(err.keys())[0] for err in arimas]
        linear_scores = [list(err.keys())[0] for err in linears]
        svc_scores = [list(err.keys())[0] for err in svcs]
        knn_scores = [list(err.keys())[0] for err in knns]
        rf = evals.random_forest(train, test)
        ses = evals.ses(train, test)

        map = {'arima': min(arima_scores),
               'linear': min(linear_scores),
               'svc': min(svc_scores),
               'knn': min(knn_scores),
               'random forest': list(rf.keys())[0],
               'ses': list(ses.keys())[0]}
        scores = pd.Series(map)
        best_model = scores.idxmin()

        # for sklearn models
        x = np.array(range(len(dat))).reshape(-1, 1)
        xf = np.array(range(len(dat), len(dat) + months)).reshape(-1, 1)

        if best_model == 'arima':
            idx = arima_scores.index(min(arima_scores))
            d = arimas[idx]
            err = list(d.keys())[0]
            order = list(d.values())[0][0]
            predictions = list(d.values())[0][1]

            model = ARIMA(dat.values, order=order, missing='drop', enforce_stationarity=False)
            model_fit = model.fit(low_memory=True)
            forecasts = model_fit.predict(len(dat.values), len(dat.values) + months - 1)

            return forecasts, predictions, err
        elif best_model == 'linear':
            idx = linear_scores.index(min(linear_scores))
            d = linears[idx]
            err = list(d.keys())[0]
            degree = list(d.values())[0][0]
            predictions = list(d.values())[0][1]

            poly = PolynomialFeatures(degree)
            x_poly = poly.fit_transform(x)
            xf_poly = poly.fit_transform(xf)

            model = LinearRegression()
            model_fit = model.fit(x_poly, dat.values)
            forecasts = model_fit.predict(xf_poly)

            return forecasts, predictions, err
        elif best_model == 'svc':
            idx = svc_scores.index(min(svc_scores))
            d = svcs[idx]
            err = list(d.keys())[0]
            kernel = list(d.values())[0][0]
            if kernel == 'poly':
                degree = list(d.values())[0][1]
                predictions = list(d.values())[0][2]

                model = SVR(kernel=kernel, degree=degree)
                model_fit = model.fit(x, dat.values)
                forecasts = model_fit.predict(xf)

                return forecasts, predictions, err
            else:
                predictions = list(d.values())[0][1]

                model = SVR(kernel=kernel)
                model_fit = model.fit(x, dat.values)
                forecasts = model_fit.predict(xf)

                return forecasts, predictions, err
        elif best_model == 'knn':
            idx = knn_scores.index(min(knn_scores))
            d = knns[idx]
            err = list(d.keys())[0]
            weight = list(d.values())[0][0]
            neighbors = list(d.values())[0][1]
            predictions = list(d.values())[0][2]

            model = KNeighborsRegressor(weights=weight, n_neighbors=neighbors)
            model_fit = model.fit(x, dat.values)
            forecasts = model_fit.predict(xf)

            return forecasts, predictions, err
        elif best_model == 'random forest':
            err = list(rf.keys())[0]
            predictions = list(rf.values())[0]

            model = RandomForestRegressor(n_estimators=1000, criterion='mae', random_state=400)
            model_fit = model.fit(x, dat.values)
            forecasts = model_fit.predict(xf)

            return forecasts, predictions, err
        else:
            err = list(ses.keys())[0]
            predictions = list(ses.values())[0]

            model = SimpleExpSmoothing(dat.values)
            model_fit = model.fit()
            forecasts = model_fit.predict(len(dat.values), len(dat.values) + months - 1)

            return forecasts, predictions, err
