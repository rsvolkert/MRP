import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from Cross import Cross
import webbrowser
from threading import Timer

data = pd.read_excel('Analysis Data.xlsx', sheet_name='Main')
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
part_nums = use_only.columns.to_numpy()

crosses = pd.read_excel('Analysis Data.xlsx', sheet_name='Cross', index_col=0)

categories = pd.read_excel('Analysis Data.xlsx', sheet_name='Categories', index_col=0)
cat_idx = [pn in use_only.columns for pn in categories.index]
cat_opts = list(categories.loc[cat_idx, 'Sales category'].dropna().unique())
cat_opts.remove('Disc')
cat_opts.sort()

# get part numbers that are not discontinued
pn_filter = [pn not in categories.loc[categories['Sales category'] == 'Disc'].index for pn in part_nums]
part_nums = list(part_nums[pn_filter])

# generate crosses to be used
crosses = crosses.loc[part_nums]
crosses.loc[crosses.Cross == 0, 'Cross'] = crosses.loc[crosses.Cross == 0].index.values

errors = pd.read_excel('Analysis Data.xlsx', sheet_name='Errors', index_col=0)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([

    html.H1('Rupes Forecaster', style={'text-align': 'center'}),

    html.Div([
        html.P('Please select a part.'),

        dcc.Dropdown(
            id='dropdown',
            value=crosses.Cross.unique()[0],
            clearable=False
        )
    ]),

    html.Br(),

    dcc.Checklist(
        id='options',
        options=[{'label': i, 'value': i} for i in cat_opts],
        value=[],
        labelStyle={'display': 'inline-block'},
        inputStyle={'margin-left': '20px'}
    ),

    html.Br(),

    html.Button('Reload', id='reload', n_clicks=0),

    dcc.Graph(id='cross'),
    html.Div(id='graph-container'),
    html.Div(dcc.Graph(id='empty', figure={'data': []}), style={'display': 'none'})
])


@app.callback(
    [Output('dropdown', 'options'),
     Output('dropdown', 'value')],
    Input('options', 'value')
)
def update_dropdown(options):
    if not options:
        return [{'label': i, 'value': i} for i in list(crosses.Cross.unique())], crosses.Cross.unique()[0]

    pns = categories[categories['Sales category'].isin(options)].index
    pns = pns[pns.isin(use_only.columns)]
    return_crosses = crosses.loc[pns]
    return [{'label': i, 'value': i} for i in list(return_crosses.Cross.unique())], return_crosses.Cross.unique()[0]


@app.callback(
    [Output('cross', 'figure'),
     Output('graph-container', 'children')],
    [Input('dropdown', 'value'),
     Input('reload', 'n_clicks')])
def update_graph(cross_val, n_clicks):
    data = pd.read_excel('Analysis Data.xlsx', sheet_name='Main')
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

    forecasts = pd.read_excel('Analysis Data.xlsx', sheet_name='Forecasts', index_col=0).T
    predictions = pd.read_excel('Analysis Data.xlsx', sheet_name='Predictions', index_col=0)

    # generate cross
    cross = Cross(cross_val)

    if len(cross.parts) == 1:
        part_num = cross.parts[0]

        dff = use_only[part_num].reset_index()

        x = dff.index.values.reshape(-1, 1)
        y = dff[part_num].values
        model1 = LinearRegression()
        model1.fit(x, y)
        preds1 = model1.predict(x)

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=dff.Date, y=preds1, name='Trend'))
        fig1.add_trace(go.Scatter(x=dff.Date, y=dff[part_num], name='Data'))

        if part_num in forecasts.columns:
            forecast = forecasts[part_num].dropna()
            prediction = predictions[part_num].dropna()
            forecast = prediction.append(forecast).sort_index()
            forecast = forecast.astype(int)
            fig1.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name='Forecast'))

        fig1.update_layout(title='Total Consumption')

        return fig1, []

    else:
        multiplier = cross.get_multiplier()
        parts = [pn in use_only.columns for pn in multiplier.index]
        parts = multiplier[parts].index
        crossed = multiplier.loc[parts] * use_only[parts]
        not_disc = [pn not in categories.loc[categories['Sales category'] == 'Disc'].index for pn in parts]
        crossed = crossed[crossed.columns[not_disc]]
        crossed = crossed.reset_index()

        x = crossed.index.values.reshape(-1, 1)
        y = crossed.drop('Date', axis=1).sum(axis=1).values
        model2 = LinearRegression()
        model2.fit(x, y)
        preds2 = model2.predict(x)

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=crossed.Date, y=preds2, name='Trend'))
        fig1.add_trace(go.Scatter(x=crossed.Date, y=crossed.sum(axis=1), name='Data'))

        forecasted = [pn in crossed.columns for pn in forecasts.columns]
        forecasted = forecasts[forecasts.columns[forecasted]].dropna()

        if not forecasted.empty:
            forecasted = forecasted * multiplier.loc[forecasted.columns]
            predicted = predictions[forecasted.columns] * multiplier.loc[forecasted.columns]
            forecasted = forecasted.sum(axis=1)
            predicted = predicted.sum(axis=1)

            forecast = predicted.append(forecasted).sort_index()
            forecast = forecast.astype(int)
            fig1.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name='Forecast'))

        fig1.update_layout(title='Total Consumption')

        graphs = []
        i = 0
        for part_num in cross.parts:
            dff = use_only[part_num].reset_index()

            x = dff.index.values.reshape(-1, 1)
            y = dff[part_num].values
            model1 = LinearRegression()
            model1.fit(x, y)
            preds1 = model1.predict(x)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dff.Date, y=preds1, name='Trend'))
            fig.add_trace(go.Scatter(x=dff.Date, y=dff[part_num], name='Data'))

            if part_num in forecasts.columns:
                forecast = forecasts[part_num].dropna()
                prediction = predictions[part_num].dropna()
                forecast = prediction.append(forecast).sort_index()
                forecast = forecast.astype(int)
                fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name='Forecast'))

            fig.update_layout(title=f'{part_num} Consumption')

            graphs.append(dcc.Graph(
                id=f'graph-{i}', figure=fig
            ))

        return fig1, html.Div(graphs)


def open_browser():
    webbrowser.open_new('http://localhost:8050')


if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run_server(use_reloader=False)
