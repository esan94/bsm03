"""
En este archivo se encuentras las funciones necesarias para
clasificar los registros de cierre en bolsa.
"""
from dateutil.relativedelta import relativedelta
from datetime import datetime

import pandas as pd
import numpy as np
import numba as nb


def sign(value):
    """
    Función para calcular si un valor dado es positivo
    o negativo. 1 implica un valor positivo y -1 indica
    un valor negativo.

    Parámetros:
    ----------
        value float:
            Valor sobre el que se quiere calcular si es
            positivo o negativo.

    Return:
    ------
        int:
            Entero que indica si el valor es positivo o
            negativo (1 -> +; -1 -> -).
    """
    return 1 if value >= 0 else -1

def pct_change(df, periods):
    """
    Función para calcular el porcentaje de cambio de un
    día con respecto a un periodo dado, sobre la columna
    del dataframe de precio de cierre.

    Parámetros:
    ----------
        df pd.DataFrame:
            DataFrame sobre el que calcular el porcentaje
            de cambio.
        periods int:
            Número de días con respecto al cual se quiere
            calcular el porcentaje de cambio.

    Return:
    ------
        pd.DataFrame:
            DataFrame con la columna de porcentaje de
            cambio calculada.
    """
    df = df[['symbol', 'date', 'close']]
    df = df.sort_values(by=['date'], ascending=False)
    df[f'pct_change_{periods}'] = - df['close'].pct_change(periods)
    return df.dropna()

@nb.jit
def stats(arr_val, sign_function, rel_delta):
    """
    Esta función calcula los percentiles 10, 25, 60, 75
    y 90 de las distribuciones de cuatro meses atrás para
    una fecha data. Además calcula el IQR y los límites
    para considerar un valor como outlier.

    Parámetros:
    ----------
        arr_val np.array:
            Array con los valores de cierre de un
            ticket dado.
        sign_function function:
            Función para clasificar un registro como que
            aumenta el valor o que lo pierde.
        rel_delta relativedelta:
            Meses relativos sobre los que calcular
            las distribuciones.

    Return:
    ------
        np.array:
            Valores concatenados de ticket, fechas y
            estadísticos sobre los datos de cierre.
    """

    # Vectorización de la funcióm sign_function
    vect_sign = np.vectorize(sign_function)
    result = np.zeros((arr_val.shape[0], 7))
    row = 0
    arr_sd = arr_val[:, 0:2]
    for date in arr_val[:, 1]:
        # Filtrado de los ultimos 4 meses a fecha dada
        datetime_ = datetime(date.year, date.month, date.day)
        arr_aux = arr_val[(arr_val[:, 1] >= datetime_-rel_delta) \
                    & (arr_val[:, 1] <= datetime_)]
        arr_aux = np.concatenate((arr_aux, vect_sign(arr_aux[:,3]).reshape(-1, 1)),
                                 axis=1)

        arr_pos = arr_aux[arr_aux[:, 4] == 1][:, 3]
        arr_neg = arr_aux[arr_aux[:, 4] == -1][:, 3]
        if arr_aux.shape[0] <= 78:
            break

        # Inicialización de los estadísticos y cálculo
        # de estos
        stats_pos = np.zeros(8)
        stats_neg = np.zeros(8)
        for idx, percentil in enumerate([10, 25, 60, 75, 90]):
            stats_pos[idx] += 0 if len(arr_pos) == 0 \
                              else np.percentile(arr_pos, percentil)
            stats_neg[idx] += 0 if len(arr_neg) == 0 \
                              else np.percentile(arr_neg, percentil)

        stats_pos[5] += stats_pos[3] - stats_pos[1]
        stats_pos[6] += stats_pos[1] - stats_pos[5] * 1.5
        stats_pos[7] += stats_pos[3] + stats_pos[5] * 1.5
        stats_neg[5] += stats_neg[3] - stats_neg[1]
        stats_neg[6] += stats_neg[1] - stats_neg[5] * 1.5
        stats_neg[7] += stats_neg[3] + stats_neg[5] * 1.5

        # Rellena el array con los valores finales
        result[row, 0] = arr_aux[0, 3]
        result[row, 1] = arr_aux[0, 4]

        if result[row, 1] == 1:
            result[row, 2] = stats_pos[0]
            result[row, 3] = stats_pos[2]
            result[row, 4] = stats_pos[4]
            result[row, 5] = stats_pos[6]
            result[row, 6] = stats_pos[7]
        else:
            result[row, 2] = stats_neg[0]
            result[row, 3] = stats_neg[2]
            result[row, 4] = stats_neg[4]
            result[row, 5] = stats_neg[6]
            result[row, 6] = stats_neg[7]

        row += 1
    return np.concatenate((arr_sd, result), axis=1)

def tagging(df, periods):
    """
    Función para clasificar los registros en subida/bajada
    fuerte (strong bull/bear), en subidas o bajadas (bull,
    bear), se mantiene (keep) y outliers positivos y nega-
    tivos (outlier (bull, bear)).

    Parámetros:
    ----------
        df pd.DataFrame:
            DataFrame sobre el que realizar la clasifica-
            ción.
        periods int:
            Número de días con respecto al cual se quiere
            realizar la clasificación.

    Return:
    ------
        pd.DataFrame:
            Dataframe con los registros clasificados.
    """

    # Clasificación de los registros negativos. En este caso
    # "outlier bear" será todo aquello que sea menor que el iqr_min
    # (que es p25 - IQR * 1.5), "strong bear" será todo
    # aquello que se enuentre entre iqr_min y el percentil 60,
    # "bear" es todo aquello que se encuentra entre el per-
    # centil 60 y el 90 y "keep" es el percentil 10 más cerca-
    # no a cero.
    df.loc[(df['sign'] == -1) & (df['iqr_min'] >= df[f'pct_change_{periods}']),
           f'tag_{periods}'] = 'outlier bear'
    df.loc[(df['sign'] == -1) & (df['iqr_min'] < df[f'pct_change_{periods}']),
           f'tag_{periods}'] = 'strong bear'
    df.loc[(df['sign'] == -1) & (df['p60'] <= df[f'pct_change_{periods}']),
           f'tag_{periods}'] = 'bear'
    df.loc[(df['sign'] == -1) & (df['p90'] < df[f'pct_change_{periods}']),
           f'tag_{periods}'] = 'keep'

    # Clasificación de los registros positivos. En este caso
    # "outlier bull" será todo aquello que sea mayor que el iqr_max
    # (que es p75 + IQR * 1.5), "strong bull" será todo
    # aquello que se enuentre entre iqr_max y el percentil 60,
    # "bull" es todo aquello que se encuentra entre el per-
    # centil 60 y el 10 y "keep" es el percentil 10 más cerca-
    # no a cero.
    df.loc[(df['sign'] == 1) & (df['iqr_max'] <= df[f'pct_change_{periods}']),
           f'tag_{periods}'] = 'outlier bull'
    df.loc[(df['sign'] == 1) & (df['iqr_max'] > df[f'pct_change_{periods}']),
           f'tag_{periods}'] = 'strong bull'
    df.loc[(df['sign'] == 1) & (df['p60'] >= df[f'pct_change_{periods}']),
           f'tag_{periods}'] = 'bull'
    df.loc[(df['sign'] == 1) & (df['p10'] > df[f'pct_change_{periods}']),
           f'tag_{periods}'] = 'keep'
    return df

def create_tags(dataframe, periods):
    """
    Función para clasificar los registros de cierre como
    subidas/bajadas fuertes, subidas/bajada, o matiene de
    un día con respecto a otro.

    Parámetros:
    ----------
        df pd.DataFrame:
            DataFrame que se quiere clasificar.
        periods int:
            Número de días con respecto al cual se quiere
            clasificar.

    Return:
    ------
        pd.DataFrame:
            Dataframe con los datos clasificados.
    """
    COLS_STATS = ['symbol', 'date', f'pct_change_{periods}',
                  'sign', 'p10', 'p60', 'p90', 'iqr_min', 'iqr_max']
    dataframe = pct_change(dataframe, periods)
    df = stats(dataframe.to_numpy(), sign, relativedelta(months=4))
    data = pd.DataFrame(df, columns=COLS_STATS)
    df = tagging(data, periods)
    df = df.dropna()
    return df[['symbol', 'date', f'pct_change_{periods}', f'tag_{periods}']].set_index(['symbol', 'date'])
