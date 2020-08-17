"""
En este archivo se encuentran funciones para la lectura de los
datos descargados.
"""
from dateutil.relativedelta import relativedelta
from datetime import datetime

import pandas as pd

import constants as c
from tools.tags import pct_change

def read_data_2_pred(symbol):
    """
    Esta función sirve para leer los datos necesarios, en el
    último día de datos descargados, para predecir la
    clasificación de una compañía en bolsa.

    Parámetros:
    ----------
        symbol str:
            Nombre de la acción en bolsa.

    Return:
    -------
        pd.DataFrame:
            DataFrame con el último día de datos necesarios
            para la predicción.
    """
    # Ruta de los archivos a leer
    close_path = f"{c.PATH_SAVE_CLOSE}{symbol}.csv"
    tech_path = f"{c.PATH_SAVE_TECH}{symbol}.csv"

    # Carga de los datos
    df = pd.read_csv(close_path, names=c.COLUMNS)
    df['volume'] = df['volume'].astype(float)
    df_tech = pd.read_csv(tech_path, names=c.COLUMNS_TECH)
    dataframe = pd.merge(left=df, right=df_tech, on=['date', 'symbol'], how='inner')
    dataframe['date'] = pd.to_datetime(dataframe['date'])

    # Filtrado de los datos por el último día disponible
    dataframe = dataframe[dataframe['date'] == dataframe['date'].max()]

    return dataframe

def read_last_data(symbol):
    """
    Esta función se usa para leer los últimos día de datos.

    Parámetros:
    ----------
        symbol str:
            Nombre de la acción en bolsa.

    Return:
    -------
        pd.DataFrame:
            DataFrame con el último día de datos.
    """
    close_path = f"{c.PATH_SAVE_CLOSE}{symbol}.csv"
    tech_path = f"{c.PATH_SAVE_TECH}{symbol}.csv"
    df = pd.read_csv(close_path, names=c.COLUMNS)
    df['date'] = pd.to_datetime(df['date'])
    df_tech = pd.read_csv(tech_path, names=c.COLUMNS_TECH)
    df_tech['date'] = pd.to_datetime(df_tech['date'])
    df_aux = df[df['date'] == df['date'].max()]
    df_tech_aux = df_tech[df_tech['date'] == df_tech['date'].max()]
    df_day = pd.merge(df_aux, df_tech_aux, on=['date', 'symbol'], how='inner')

    return df_day

def read_data(symbol, year_value, is_tech=False):
    """
    Esta función sirve para leer los datos y filtrar por una fecha
    dada además de unir la información de los indicadores técnicos
    si se requiere.

    Parámetros:
    ----------
        symbol str:
            Nombre de la acción en bolsa.
        year_value str:
            Filtrado de datos por la fecha.
        is_tech bool:
            Booleano para tener en cuenta o no la información de los
            indicadores técnicos.

    Return:
    -------
        pd.DataFrame:
            DataFrame con toda la información de una empresa unida.
    """
    close_path = f"{c.PATH_SAVE_CLOSE}{symbol}.csv"
    tech_path = f"{c.PATH_SAVE_TECH}{symbol}.csv"
    df = pd.read_csv(close_path, names=c.COLUMNS)
    df['date'] = pd.to_datetime(df['date'])
    ends = year_value[-1]
    max_date = df['date'].max()
    max_date = datetime(max_date.year, max_date.month, max_date.day)
    rel_delta = relativedelta(years=int(year_value[0:-1])) if ends == "Y" \
                else relativedelta(months=int(year_value[0:-1]))
    filter = max_date - rel_delta
    df = df[df['date'] >= filter]

    # Unir la información de los indicadores tecnicos si se requiere
    if is_tech:
        df_tech = pd.read_csv(tech_path, names=c.COLUMNS_TECH)
        df_tech['date'] = pd.to_datetime(df_tech['date'])
        df = pd.merge(left=df, right=df_tech, how='inner', on=['date', 'symbol'])

    return df

def read_pv(clf, days2pred):
    """
    Esta función sirve para leer los datos usados para comparar
    los vectores de datos procesados con respecto a la mediana
    de los últimos 6 meses.

    Parámetros:
    ----------
        symbol clf:
            Modelo ML usado (Random Forest es el único disponible).
        days2pred str:
            Días sobre los que se quiere hacer la predicción.

    Return:
    -------
        pd.DataFrame:
            DataFrame con los datos de los últimos 6 meses filtrado
            para el modelo y los días pedidos.
    """
    path_pv = f"{c.PATH_PAST_VAL}2020_1.csv"
    df = pd.read_csv(path_pv, names=c.COLS_PAST_DATA)

    # Filtrado de los datos
    df = df[df['tag_y'].str.startswith(f"{days2pred} - {clf}")]
    df['pred - class'] = df['tag_y'].str.split(' - ', 2).str[-1]

    df = df.drop('tag_y', axis=1)
    return df[['pred - class', 'PC 1', 'PC 2', 'PC 3', 'PC 4']].round(3)
