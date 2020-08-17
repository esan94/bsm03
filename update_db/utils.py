"""
En este archivo se encuentran funciones auxiliares usadas para
actualizar día a día los datos.
"""
from datetime import datetime, timedelta
import os
import time
import logging


from alpha_vantage.techindicators import TechIndicators
import pandas as pd
import numpy as np
import pandas_datareader.data as web

import constants as c

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

def get_symbols():
    """
    Lee los índices bursátiles del archivo aux_symbols.

    Return:
    -------
        np.array:
            Array con los índices cargados
    """
    df = pd.read_csv(f"{c.PATH_SYMBOL}aux_symbols.csv", names=['description', 'symbol'])
    return df['symbol'].values

def failed_symbols(symbol):
    """
    Esta función es para guardar en un archivo los índices bursátiles que han fallado.

    Parámetros:
    ----------
        symbol str:
            Nombre de la acción en bolsa.
    """

    df = pd.DataFrame({'symbol': [symbol]})
    if os.path.isfile(f"{c.PATH_SYMBOL}failed_symbols.csv"):
        df.to_csv(f"{c.PATH_SYMBOL}failed_symbols.csv", index=False, header=False, mode='a')
    else:
        df.to_csv(f"{c.PATH_SYMBOL}failed_symbols.csv", index=False, header=False)


def get_daily(symbol, end=None, is_save=False):
    """
    Esta función es para descargar todos los indicadores técnicos.
    Si usamos una API_KEY gratuita de ALPHA VANTAGE tenemos que
    descomentar los time.sleep() de la función para que no de un
    problema de peticiones; en el caso de una API_KEY premium no haría
    falta descomentarlo.

    Parámetros:
    ----------
        symbol str:
            Nombre de la acción en bolsa.
        end str or None:
            Parámetro para decidir si descargar todos los datos hasta el día
            de hoy o hasta una fecha dada. Por defecto es None lo que indica
            que se descarga hasta el día de hoy.
        is_save bool:
            Booleano para decidir si guardar o no los datos
            descargados. Por defecto False no guarda los
            datos descargados
    """

    # Si existe el archivo tomar la ultima fecha de datos para actualizarlo
    # en caso negativo abrir un archivo nuevo
    if os.path.isfile(f"{c.PATH_SAVE_CLOSE}{symbol}.csv"):
        df = pd.read_csv(f"{c.PATH_SAVE_CLOSE}{symbol}.csv", names=c.COLUMNS)
        df = df[df['symbol'] == symbol]
        df['date'] = pd.to_datetime(df['date'])
        ld = df['date'].tail(1)
        start = datetime(ld.dt.year, ld.dt.month, ld.dt.day) + timedelta(days=1)
    else:
        start = datetime(2006, 1, 1)

    # Si tomar como ultima fecha de datos el día de hoy o uno dado por
    # parámetro
    if end is None:
        end = datetime.today()
    else:
        pass

    # Algoritmo para obtener los precios de cierre diarios desde un inicio
    # hasta una fecha final
    data = web.DataReader(symbol, "av-daily", start=start, end=end, api_key=c.ALPHA_VAN_KEY)
    data.reset_index(inplace=True)
    data['symbol'] = symbol
    data.rename(columns={'index': 'date'}, inplace=True)
    data['date'] = pd.to_datetime(data['date'])

    # Guardar o no los resultados
    if is_save:
        data[c.COLUMNS].to_csv(f"{c.PATH_SAVE_CLOSE}{symbol}.csv", mode='a', index=False, header=False)


def get_technical(symbol, is_save=False):
    """
    Esta función es para descargar todos los indicadores técnicos.
    Si usamos una API_KEY gratuita de ALPHA VANTAGE tenemos que
    descomentar los time.sleep() de la función para que no de un
    problema de peticiones; en el caso de una API_KEY premium no haría
    falta descomentarlo.

    Parámetros:
    ----------
        symbol str:
            Nombre de la acción en bolsa.
        is_save bool:
            Booleano para decidir si guardar o no los datos
            descargados. Por defecto False no guarda los
            datos descargados
    """
    try:
        # Comprueba si ya existe o no el archivo y en el caso de que
        # si exista guarda solo los días que no estén descargados
        if os.path.isfile(f"{c.PATH_SAVE_TECH}{symbol}.csv"):
            df = pd.read_csv(f"{c.PATH_SAVE_TECH}{symbol}.csv", names=c.COLUMNS)
            df = df[df['symbol'] == symbol]
            df['date'] = pd.to_datetime(df['date'])
            ld = df['date'].tail(1)
            last = datetime(ld.dt.year, ld.dt.month, ld.dt.day)
        else:
            last = None

        techindc = list()

        # Descarga los datos de indicadores técnicos.
        ti = TechIndicators(key=c.ALPHA_VAN_KEY, output_format='pandas')
        init = time.time()
        macd = ti.get_macd(symbol, interval='daily')[0]
        techindc.append(macd)

        stoch = ti.get_stoch(symbol, interval='daily')[0]
        techindc.append(stoch)

        ad = ti.get_ad(symbol, interval='daily')[0]
        techindc.append(ad)

        obv = ti.get_obv(symbol, interval='daily')[0]
        techindc.append(obv)

    #    time.sleep(60)

        sma = ti.get_sma(symbol, interval='daily', time_period=50)[0]
        sma.columns = [f"{c}50"for c in sma.columns]
        techindc.append(sma)

        bbands = ti.get_bbands(symbol, interval='daily', time_period=28)[0]
        bbands.columns = [f"{c}28"for c in bbands.columns]
        techindc.append(bbands)

        for tp in c.TIME_PERIOD:

            rsi = ti.get_rsi(symbol, interval='daily', time_period=tp)[0]
            rsi.columns = [f"{c}{tp}"for c in rsi.columns]
            techindc.append(rsi)

            adx = ti.get_adx(symbol, interval='daily', time_period=tp)[0]
            adx.columns = [f"{c}{tp}"for c in adx.columns]
            techindc.append(adx)

    #        time.sleep(60)

            cci = ti.get_cci(symbol, interval='daily', time_period=tp)[0]
            cci.columns = [f"{c}{tp}"for c in cci.columns]
            techindc.append(cci)

            aroon = ti.get_aroon(symbol, interval='daily', time_period=tp)[0]
            aroon.columns = [f"{c}{tp}"for c in aroon.columns]
            techindc.append(aroon)

        df_techindc = pd.concat(techindc, axis=1, join='inner')
        df_techindc.reset_index(inplace=True)
        df_techindc['symbol'] = symbol

        if last is not None:
            df_techindc = df_techindc[df_techindc['date'] > last]

        # Guardar los datos
        if is_save:
            df_techindc[c.COLUMNS_TECH].to_csv(f"{c.PATH_SAVE_TECH}{symbol}.csv", mode='a', index=False, header=False)
    except:
        LOGGER.warning(f"Ticker {symbol} ha fallado.")
