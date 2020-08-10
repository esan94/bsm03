"""
Este script es donde se hace todo el uso de la estrategia 1.
"""
from datetime import datetime
from dateutil.relativedelta import relativedelta

from dash.dependencies import Input, Output
from sklearn.metrics.pairwise import cosine_similarity
import dash_html_components as html
import dash_core_components as dcc
import dash_table
import pandas as pd
import plotly.graph_objects as go
import numpy as np

import tools.read as t_read
import tools.preprocessing as pcg
import tools.predict as pdt
import constants as c
from app import app

# Creación del HTML para la estrategia 1.
layout_strategyone = html.Div([
    html.Div([
        html.H1(' '),
        html.Br(),
        dcc.Dropdown(
            id='cls',
            options=[
                {'label': 'Random Forest', 'value': 'rf'}],
            multi=False,
            value='rf',
            style={'width': '40%',
                   'margin':'auto',
                   'color': c.PALETTE['grey']}
        ),

        dcc.Dropdown(
            id='pred',
            options=[
                {'label': 'Strong Bull', 'value': 'strong bull'},
                {'label': 'Strong Bear', 'value': 'strong bear'},
                {'label': 'Bull', 'value': 'bull'},
                {'label': 'Bear', 'value': 'bear'},
                {'label': 'Keep', 'value': 'keep'}
            ],
            multi=False,
            value='strong bull',
            style={'width': '40%',
                   'margin':'auto',
                   'color': c.PALETTE['grey']}
        ),

        html.H1(' '),
        html.Br(),
        dcc.Dropdown(
            id='symbols',
            options=[
                {'label': ticker, 'value': ticker} for ticker in c.SYMBOLS
            ],
            multi=False,
            value='AAPL',
            style={'width': '40%',
                   'margin':'auto',
                   'color': c.PALETTE['grey']}
        ),

        dcc.RadioItems(
            id='days2pred',
            options=[
                {'label': '7d', 'value': '7'},
                {'label': '14d', 'value': '14'},
                {'label': '21d', 'value': '21'},
                {'label': '28d', 'value': '28'}
            ],
            value='7', style={
                'textAlign': 'center',
                'color': c.PALETTE['grey'],
                'margin':'auto',
                'columnCount': 4
            }
        )], style={
            'display': 'flex',
            'textAlign': 'center',
            'color': c.PALETTE['grey']}, className='row'),

    html.H1(' '),
    html.Br(),
    html.Div([
        html.Div(style={'backgroundColor': c.PALETTE['black'], 'width': '60%', 'margin': 30},
                 id='past-data-table'),
        html.Div([dcc.Graph(style={
                         'backgroundColor': c.PALETTE['black'],
                         'width': '70%', 'display': 'inline-block'}, id='past-data-graph')]),

             ], style={'backgroundColor': c.PALETTE['black'],
                       'width': '100%', 'columnCount': 2, 'display': 'inline-block'}),
    html.H1(' '),
    html.Br(),
    html.Div([html.Div(
        [html.Div(className='row', style={'backgroundColor': c.PALETTE['black'], 'margin': 30, 'width': '20%'},
             id='prediction'),
         html.H1(' '),
         html.Div(className='row', style={'backgroundColor': c.PALETTE['black'],
                             'width': '20%', 'margin': 30},
                      id='pred-cosine')]),
              html.Div(className='row', style={'backgroundColor': c.PALETTE['black'], 'margin': 30, 'width': '20%'},
                   id='best')],

                      style={'backgroundColor': c.PALETTE['black'],
                       'width': '100%', 'display': 'inline-block', 'columnCount': 2}
                       ),

    html.Br(),
    html.Div(style={
        'textAlign': 'center'
        },
        children=dcc.Link('Home', href='/', style={
            'color': c.PALETTE['red']
            })),
    html.Div(style={
        'textAlign': 'center'
        },
        children=dcc.Link('Update', href='/update', style={
            'color': c.PALETTE['red']
            })),
    html.Div(style={
        'textAlign': 'center'
        },
        children=dcc.Link('Analysis', href='/analysis', style={
            'color': c.PALETTE['red']
            }))

])

@app.callback(
    Output('best', 'children'),
    [Input('cls', 'value'),
     Input('days2pred', 'value')])
def get_best_invests(cls, days2pred):
    """
    Esta función toma el archivo de datos resumen para
    calcular cuales son los mejores índices para hacer una predicción
    basándonos en la similaridad del coseno.

    Parámetros:
    ----------
        cls str:
            Nombre del modelo ML usado (solo disponible Random Forest).
        days2pred str:
            Días sobre los que se quiere hacer la predicción.

    Return:
    -------
        dash_table.DataTable:
            DataTable con las 50 mejores compañías en las que vender o comprar
            día a día.
    """

    # Cargar las medianas de los últimos 6 meses
    df_pv = t_read.read_pv(cls, days2pred)

    # Filtrar para obtener solo los que se predijeron como fuertes
    # subidas o bajadas
    df_pv = df_pv[df_pv['pred - class'].str.startswith('strong')]

    # Cargar los últimos días de datos que tenemos para los índices
    list_df = []
    for symbol in c.SYMBOLS:
        df = t_read.read_last_data(symbol)
        list_df.append(df)
    df_day = pd.concat(list_df, ignore_index=True)
    df_day = df_day.set_index(['symbol', 'date'])[c.MODEL_COLUMNS]

    # Aplicar el escalado robusto a los datos
    X_scl = pcg.robust_scaler_model(df_day.values, days2pred)

    # Predecir con el Random Forest las clasificaciones
    pred = pdt.predict(X_scl, days2pred)

    # Aplicar un PCA para comparar con los datos de medianas
    X_pca = pcg.pca(X_scl, days2pred)
    df = pd.DataFrame({'prediction': pred}, index=df_day.index)
    for col in range(0, X_pca.shape[1]):
        df[f"P.PC {col + 1}"] = X_pca[:, col]

    df = df[df['prediction'].str.startswith('strong')]

    df = df.reset_index()

    # Calcular la similaridad del cosenos de los vectores de datos
    # de los últimos días de datos que tenemos con las medianas de
    # los últimos 6 meses
    df_pv['prediction'] = df_pv['pred - class'].str.split(' - ').str[0]
    df_tot = pd.merge(df, df_pv, on='prediction', how='inner')
    for idx in range(0, df_tot.shape[0]):
        p_pc = df_tot.loc[idx, [f"P.PC {i}" for i in range(1, 5)]].values
        pc = df_tot.loc[idx, [f"PC {i}" for i in range(1, 5)]].values
        cos_sim_rf = cosine_similarity(p_pc.reshape(1, -1), pc.reshape(1, -1))
        df_tot.loc[idx, 'cos-sim'] = cos_sim_rf[0]
    df_tot['cos-sim'] = df_tot['cos-sim'].abs()

    # Ordenar de mayor a menor y sacar las mejores 50 compañías
    df_tot = df_tot.sort_values(by=['date', 'cos-sim'], ascending=[False, False])
    df_tot = df_tot[['date', 'symbol', 'prediction', 'pred - class', 'cos-sim']]
    max_idx = df_tot.groupby(['date', 'symbol', 'prediction'])['cos-sim'].idxmax().values
    df_tot = df_tot.loc[max_idx].head(50).sort_values(by='cos-sim', ascending=False)
    return dash_table.DataTable(
                id='cosSim',
                columns=[{"name": i, "id": i} for i in df_tot.columns],
                data=df_tot.to_dict('records'),
                style_as_list_view=True,
                style_header={'backgroundColor': c.PALETTE['black'],
                              'fontWeight': 'bold'},
                style_cell={
                    'backgroundColor': c.PALETTE['black'],
                    'color': c.PALETTE['blue']
                    }
            )

@app.callback(
    Output('pred-cosine', 'children'),
    [Input('cls', 'value'),
     Input('prediction', 'children'),
     Input('days2pred', 'value')])
def cosine_sim(cls, prediction, days2pred):
    """
    Esta función calcula la similaridad del coseno
    de la compañía elegida para los días a predecir
    seleccionados mostrando finalmente una tabla con
    los resultados.

    Parámetros:
    ----------
        cls str:
            Nombre del modelo ML usado (solo disponible Random Forest).
        prediction dict:
            Diccionario con los datos de predicción de una compañía seleccionada
            previamente.
        days2pred str:
            Días sobre los que se quiere hacer la predicción.

    Return:
    -------
        dash_table.DataTable:
            DataTable con los datos de similaridad del coseno para la compañía elegida.
    """
    data_pred = prediction \
            .get('props', {}) \
            .get('data',{})[0]
    pred =  data_pred.get(f"Prediction in {days2pred} days")[0]
    actual = np.array([[data_pred.get('PC 1'), data_pred.get('PC 2'), data_pred.get('PC 3'), data_pred.get('PC 4')]])
    df_pv = t_read.read_pv(cls, days2pred)
    df_pv = df_pv[df_pv['pred - class'].str.startswith(pred)]
    cos_sim_rf = cosine_similarity(actual, df_pv.select_dtypes(float).values)

    df_pv['cos-sim-value'] = cos_sim_rf.reshape(-1, 1)
    df_pv['cos-sim-value'] = df_pv['cos-sim-value'].abs().round(3)
    df = df_pv[['pred - class', 'cos-sim-value']]
    df = df.sort_values('cos-sim-value', ascending=False)
    return dash_table.DataTable(
                id='cosSim',
                columns=[{"name": i, "id": i} for i in df.columns],
                data=df.to_dict('records'),
                style_as_list_view=True,
                style_header={'backgroundColor': c.PALETTE['black'],
                              'fontWeight': 'bold'},
                style_cell={
                    'backgroundColor': c.PALETTE['black'],
                    'color': c.PALETTE['yellow']
                    }
            )

@app.callback(
    Output('prediction', 'children'),
    [Input('symbols', 'value'),
     Input('days2pred', 'value')])
def get_pred(symbol, days2pred):
    """
    Esta función es para calcular la predicción para una compañía
    y un horizonte temporal dados-

    Parámetros:
    ----------
        symbol str:
            Nombre de la acción en bolsa.
        days2pred str:
            Días sobre los que se quiere hacer la predicción.
    Return:
    -------
        dash_table.DataTable:
            DataTable con la información de la predicción para los
            parámetros escogidos.
    """
    # Cargar la información
    dataframe = t_read.read_data_2_pred(symbol)
    date = dataframe['date']

    # Aplicar un escalado robusto antes de predecir y un PCA
    X_scl = pcg.robust_scaler_model(dataframe[c.MODEL_COLUMNS].values, days2pred)
    pred = pdt.predict(X_scl, days2pred)
    X_pca = pcg.pca(X_scl, days2pred)
    cols = [
        {"name": "Date", "id": "Date"},
        {"name": f"Prediction in {days2pred} days", "id": f"Prediction in {days2pred} days"},
        {"name": "PC 1", "id": "PC 1"},
        {"name": "PC 2", "id": "PC 2"},
        {"name": "PC 3", "id": "PC 3"},
        {"name": "PC 4", "id": "PC 4"}]
    values = [{
        "Date": date,
        f"Prediction in {days2pred} days": pred,
        'PC 1': round(X_pca[0][0], 3),
        'PC 2': round(X_pca[0][1], 3),
        'PC 3': round(X_pca[0][2], 3),
        'PC 4': round(X_pca[0][3], 3)}]
    return dash_table.DataTable(
                    id='predictData',
                    columns=cols,
                    data=values,
                    style_as_list_view=True,
                    style_header={'backgroundColor': c.PALETTE['black'],
                                  'fontWeight': 'bold'},
                    style_cell={
                        'backgroundColor': c.PALETTE['black'],
                        'color': c.PALETTE['green']
                        }
                )

@app.callback(
    Output('past-data-table', 'children'),
    [Input('cls', 'value'),
     Input('days2pred', 'value')])
def get_past_data_table(cls, days2pred):
    """
    Esta función es para ver en una tabla los datos de las medianas
    de los datos de los últimos 6 meses (acompañados por la
    función get_past_data_graph) usando los días sobre los que
    queremos predecir.

    Parámetros:
    ----------
        cls str:
            Nombre del modelo ML usado (solo disponible Random Forest).
        days2pred str:
            Días sobre los que se quiere hacer la predicción.
    Return:
    -------
        dash_table.DataTable:
            DataTable con la información de los últimos 6 meses.
    """
    df = t_read.read_pv(cls, days2pred)

    return dash_table.DataTable(
                id='pastData',
                columns=[{"name": i, "id": i} for i in df.columns],
                data=df.to_dict('records'),
                style_as_list_view=True,
                style_header={'backgroundColor': c.PALETTE['black'],
                              'fontWeight': 'bold'},
                style_cell={
                    'backgroundColor': c.PALETTE['black'],
                    'color': c.PALETTE['grey']
                    }
            )

@app.callback(
    Output('past-data-graph', 'figure'),
    [Input('cls', 'value'),
     Input('days2pred', 'value'),
     Input('pred', 'value'),
     Input('prediction', 'children')])
def get_past_data_graph(cls, days2pred, pred, prediction):
    """
    Esta función se usa para crear una gráfica dónde podamos ver
    como se describen las características principales de los datos
    de los 6 meses anteriores dado un horizonte temporal para predecir
    y una predicción elegida. Además de ver como se describe la actual
    predicción elegida.

    Parámetros:
    ----------
        cls str:
            Nombre del modelo ML usado (solo disponible Random Forest).
        days2pred str:
            Días sobre los que se quiere hacer la predicción.
        pred str:
            Predicción elegida para graficar los datos de los 6 meses
            previos.
        prediction dict:
            Diccionario con los datos de predicción de una compañía seleccionada
            previamente.

    Return:
    -------
        go.Figure:
            Plotly figure donde podremos ver en un radar plot descritas
            las componentes principales de las medianas de los datos de
            los últimos 6 meses además de la curva de la predicción actual
            asociada al índice bursátil elegido.
    """
    # Cargar la información
    data_pred = prediction.get('props', {}).get('data',{})[0]
    df = t_read.read_pv(cls, days2pred)
    df = df[df['pred - class'].str.startswith(pred)]
    df_bbdd = pd.melt(df, id_vars=['pred - class'],
                      value_vars=['PC 1', 'PC 2', 'PC 3', 'PC 4'])

    # Crear la figura para los últimos 6 meses
    fig = go.Figure()
    for idx, tag in enumerate(df['pred - class'].unique(), 1):
        df_aux = df_bbdd[df_bbdd['pred - class'] == tag]
        fig.add_trace(go.Scatterpolar(
          r=df_aux['value'].values,
          theta=df_aux['variable'].values,
          name=tag.split(' - ')[-1],
          line_color=c.PALETTE[list(c.PALETTE.keys())[idx]]
    ))

    # Añadir los datos de la predicción seleccionada
    fig.add_trace(go.Scatterpolar(
      r=[data_pred.get('PC 1'), data_pred.get('PC 2'), data_pred.get('PC 3'), data_pred.get('PC 4')],
      theta=['PC 1', 'PC 2', 'PC 3', 'PC 4'],
      name='predicted',
      line_color="white"
))

    fig.update_layout(
      polar=dict(
        bgcolor=c.PALETTE['black'],
        angularaxis = dict(showticklabels=True, color=c.PALETTE['grey']),
        radialaxis=dict(
          gridcolor = c.PALETTE['grey'],
          visible=True,
          showticklabels=False,
          showline=True
        )
      ),
      showlegend=False,
      paper_bgcolor=c.PALETTE['black'],
    )
    return fig
