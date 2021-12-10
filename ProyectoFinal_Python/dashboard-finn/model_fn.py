#Modulos nativos de Python
import datetime as dt
import warnings

#Modulos para gestionar dataframes, cálculos con vectores y matrices
import numpy  as np
import pandas as pd

#Modulos para la generación de modelos
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

#Función que me entrena el modelo lineal
def OLS_temp(df, horizon):
    #Calculando train y test size
    n = len(df.Close); n_test = horizon # This can be changed
    train_size = n - n_test

    #Creando cada dataframe para train y test
    train_Y = df.Close.iloc[:train_size]
    train_X = df.Date.iloc[:train_size].apply(lambda x: int(x.timestamp()))
    test_Y = df.Close.iloc[train_size:train_size + n_test] 
    dates_test = df.Date.iloc[train_size:train_size + n_test] 

    #Añadiendo la constante al modelo de regresión
    train_X = sm.add_constant(train_X)

    #Entrenando el modelo
    model = sm.OLS(train_Y, train_X).fit()

    return(train_X, train_Y, dates_test, test_Y, model)

#Función que calcula las métricas de error
def forecast_accuracy(forecast, actual, str_name):
    
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual)) # MAPE
    mae = np.mean(np.abs(forecast - actual))                 # MAE
    rmse = np.mean((forecast - actual)**2)**.5               # RMSE
    mse = np.mean((forecast - actual)**2)                    # MSE
    
    df_acc = pd.DataFrame({'MAE': [mae],
                           'MSE': [mse],
                           'MAPE': [mape],
                           'RMSE': [rmse]},
                          index=[str_name])
    
    return df_acc

#Función que me genera las predicciones con el modelo lineal
def linear_rolling(train_X, train_Y, dates_test, test_Y, model):
    predictions = list()

    for t, c_p in zip(dates_test, test_Y):
        #Transformando los datos para predecir
        t1 = int(t.timestamp())
        
        #Parametros del modelo
        intercept = model.params[0]
        pend = model.params[1]
        
        #Prediciendo los valores
        pred = intercept + pend*t1

        #Guardando la predicción en la lista
        predictions.append(pred)

        #Actualizando train_X, train_Y, y re-entrenando el modelo
        train_X = train_X.append({"const": 1,"Date":t1}, ignore_index=True)
        train_Y[len(train_Y)] = c_p
        #print(train_Y.tail())

        model = sm.OLS(train_Y, train_X).fit()
    return predictions

#Definiendo la función para entrenar los modelos ARIMA
def ARIMA_model(df, horizon, metric="aic", pq_rng=range(5), d_rng=range(3)):
    #Metric indicará la métrica a utilizar, aic o bic.
    #Calculando train y test size
    n = len(df.Close); n_test = horizon # This can be changed
    train_size = n - n_test

    #Creando cada dataframe para train y test
    train = df.Close.iloc[:train_size]
    test = df.Close.iloc[train_size:train_size + n_test] 
    dates_test = df.Date.iloc[train_size:train_size + n_test] 

    #Encontrando el mejor modelo usando la métrica seleccionada
    best_metric = np.inf
    best_order = None
    best_mdl = None
    #exceptions = []

    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = ARIMA(train, order=(i,d,j), enforce_stationarity=False).fit(method='innovations_mle')
                    if metric=="bic":
                        tmp_metric = tmp_mdl.bic
                    else:
                        tmp_metric = tmp_mdl.aic                
                    
                    if tmp_metric < best_metric:
                        best_metric = tmp_metric
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except Exception as e: 
                    #print(e)
                    #exceptions.append(e)
                    continue

    #Regresando los valores que se encontraron, junto con el train y el test set
    return(train, test, dates_test, best_mdl, best_order, best_metric)#, exceptions)

#Función para generar las predicciones
def arima_rolling(history, test, best_order):
    
    predictions = list()
    for t in range(len(test)):
        model_fit = ARIMA(history, order=best_order, enforce_stationarity=False).fit(method='innovations_mle')
        output = model_fit.forecast()
        yhat = output[0]
        #print(yhat)
        predictions.append(yhat)
        obs = test[t]
        #print(obs)
        history.append(obs)
        #print('predicted=%f, expected=%f' % (yhat, obs))
    return predictions