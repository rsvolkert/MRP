import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np

data = pd.read_excel('~/OneDrive/Consumption file/Consume + Ship Data - RV edits.xlsx', sheet_name='Main')
data = data.loc[data.PartNumber.notnull()]
data.dropna(axis=1, how='all', inplace=True)

app = dash.Dash(__name__)

app.layout = html.Div(
    children=[
        html.H1(children='Part Number Analytics'),
        html.P(children='Analyze Part Numbers'),
        dcc.Graph(
            figure={
                'data' : [

                ]
            }
        )
    ]
)
