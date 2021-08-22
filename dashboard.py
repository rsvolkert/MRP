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
from Forecaster import Forecaster

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
part_nums = use_only.columns

crosses = pd.read_excel('Analysis Data.xlsx', sheet_name='Cross', index_col=0)

categories = pd.read_excel('Analysis Data.xlsx', sheet_name='Categories', index_col=0)
cat_idx = [pn in use_only.columns for pn in categories.index]
cat_opts = categories.loc[cat_idx, 'Sales category'].dropna().unique()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([

    dcc.Dropdown(
        id='dropdown',
        options=[{'label': i, 'value': i} for i in part_nums],
        value=part_nums[0],
        clearable=False
    ),

    html.Br(),

    dcc.Checklist(
        id='options',
        options=[{'label': i, 'value': i} for i in cat_opts],
        value=[],
        labelStyle={'display': 'inline-block'},
        inputStyle={'margin-left': '20px'}
    ),

    html.Br(),

    html.Button('Forecast', id='btn', n_clicks=0),
    html.Div(id='container', children='Select categories and press forecast'),

    html.Br(),

    html.Button('Reload', id='reload', n_clicks=0),

    dcc.Graph(id='part_numbers'),
    dcc.Graph(id='cross')
])


@app.callback(
    [Output('part_numbers', 'figure'),
     Output('cross', 'figure')],
    [Input('dropdown', 'value'),
     Input('reload', 'n_clicks')])
def update_graph(part_num, n_clicks):
    forecasts = pd.read_excel('Analysis Data.xlsx', sheet_name='Forecasts', index_col=0)
    predictions = pd.read_excel('Analysis Data.xlsx', sheet_name='Predictions', index_col=0)

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
        forecast = forecasts[part_num]
        prediction = predictions[part_num].dropna()
        forecast = prediction.append(forecast).sort_index()
        fig1.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name='Forecast'))

    cross_name = crosses.loc[part_num, 'Cross']
    if cross_name != 0:
        cross = Cross(cross_name)
        multiplier = cross.get_multiplier()
        crossed = multiplier * use_only[cross.parts]

        y = crossed.sum(axis=1).values
        model2 = LinearRegression()
        model2.fit(x, y)
        preds2 = model2.predict(x)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=crossed.index, y=preds2, name='Trend'))
        fig2.add_trace(go.Scatter(x=crossed.index, y=crossed.sum(axis=1), name='Data'))
    else:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dff.index, y=[0]*len(dff.index)))

    fig1.update_layout(
        title='Part Usage Over Time',
        yaxis_title=part_num
    )

    fig2.update_layout(
        title='Cross Usage Over Time',
        yaxis_title=cross_name
    )

    return fig1, fig2


@app.callback(
    Output('container', 'children'),
    Input('btn', 'n_clicks'),
    State('options', 'value')
)
def forecast(n_clicks, options):
    if not n_clicks:
        return 'Press forecast when ready'

    if not options:
        return 'You must select a category'

    pns = categories[categories['Sales category'].isin(options)].index
    pns = pns[pns.isin(use_only.columns)]
    dat = use_only[pns]

    forecaster = Forecaster(dat, 5)
    forecaster.forecast()
    forecaster.to_excel()

    return 'Forecasting complete. Click reload to see changes.'


if __name__ == '__main__':
    app.run_server(debug=True)
