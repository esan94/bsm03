"""
En este archivo se encuentran funciones de preprocesado
de datos.
"""
import pickle

import constants as c


def robust_scaler_model(X, days2pred):
    """
    Esta función carga el escalado robusto de los datos
    previamente entrenado para aplicarlo a nuevos datos de
    cada día diferente.

    Parámetros:
    ----------
        X np.array:
            Datos de los índices bursátiles en un array de datos.
        days2pred str:
            Días sobre los que se quiere hacer la predicción.

    Return:
    -------
        np.array:
            Array con los datos transformados según el escalado
            robusto.
    """

    # Ruta donde se encuentra el preprocesado
    path_rs = f"{c.PATH_MODEL}robust_scaler_{days2pred}.pkl"

    # Cargar el modelo
    scl = pickle.load(open(path_rs, 'rb'))

    # Transformar los nuevos datos
    X_scl = scl.transform(X)
    return X_scl

def pca(X, days2pred):
    """
    Esta función carga el PCA, previamente entrenado, usado para
    reducir la dimensionalidad de los datos a 4 dimensiones con
    el fin de luego comparar con los datos de los 6 meses
    anteriores y ver cuáles son los vectores de datos que más
    se asemejan.

    Parámetros:
    ----------
        X np.array:
            Datos de los índices bursátiles en un array de datos.
        days2pred str:
            Días sobre los que se quiere hacer la predicción.

    Return:
    -------
        np.array:
            Array con los datos transformados según el escalado
            robusto.
    """

    # Ruta donde se encuentra el modelo
    path_pca = f"{c.PATH_MODEL}pca_{days2pred}.pkl"

    # Carga del modelo
    pca = pickle.load(open(path_pca, 'rb'))

    # Transformar los datos
    X_pca = pca.transform(X)
    return X_pca
