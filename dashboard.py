import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from sklearn.linear_model import LinearRegression
from Cross import Cross

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
part_nums = use_only.columns

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

    dcc.Graph(id='part_numbers'),
    dcc.Graph(id='cross')
])


@app.callback(
    [Output('part_numbers', 'figure'),
     Output('cross', 'figure')],
    Input('dropdown', 'value'))
def update_graph(part_num):
    dff = use_only[part_num].reset_index()

    x = dff.index.values.reshape(-1, 1)
    y = dff[part_num].values
    model1 = LinearRegression()
    model1.fit(x, y)
    preds1 = model1.predict(x)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=dff.Date, y=preds1, name='Trend'))
    fig1.add_trace(go.Scatter(x=dff.Date, y=dff[part_num], name='Data'))

    cross_name = data.loc[part_num, 'Cross']
    if cross_name != 0:
        cross = Cross(cross_name)
        multiplier = cross.get_multiplier()
        crosses = multiplier * use_only[cross.parts]

        y = crosses.sum(axis=1).values
        model2 = LinearRegression()
        model2.fit(x, y)
        preds2 = model2.predict(x)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=crosses.index, y=preds2, name='Trend'))
        fig2.add_trace(go.Scatter(x=crosses.index, y=crosses.sum(axis=1), name='Data'))
    else:
        fig2 = go.Scatter(x=dff.index, y=[0]*len(dff.index))

    return fig1, fig2


if __name__ == '__main__':
    app.run_server(debug=True)

