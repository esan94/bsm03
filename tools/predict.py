"""
En este archivo se encuentra la función usada para predecir
si un índice va a subir o bajar.
"""
import pickle

import constants as c


def predict(X, days2pred):
    """
    Esta función carga el modelo (Random Forest) previamente entrenado
    para predecir los valores sobre los días seleccionados.

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
    path_rf = f"{c.PATH_MODEL}rf_{days2pred}.pkl"

    # Carga del modelo
    clf_rf = pickle.load(open(path_rf, 'rb'))

    # Predecir los valores
    y_pred_rf = clf_rf.predict(X)
    return y_pred_rf
