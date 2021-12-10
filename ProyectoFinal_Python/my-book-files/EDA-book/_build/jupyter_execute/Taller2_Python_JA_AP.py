#!/usr/bin/env python
# coding: utf-8

# # Taller 2. Visualización de Datos en Python y R.
# ## Jorge Arteaga y Adriana Palacio.
# ## Maestría en Estadística Aplicada.
# ### Universidad del Norte.

# ## Ejercicio 2.1
# 
# Trabajaremos con los conjuntos de datos **Renewable Energy Consumption and Production** de Our World in Data (https://ourworldindata.org/renewable-energy). Estos conjuntos de datos están disponibles en el repositorio **GitHub** del curso, púedes ser importados con las **URL**: [share-of-electricity-production-from-renewable-sources.csv](https://raw.githubusercontent.com/lihkir/Uninorte/main/AppliedStatisticMS/DataVisualizationRPython/Lectures/Python/PythonDataSets/share-of-electricity-production-from-renewable-sources.csv), [renewable-energy-consumption-by-country.csv](https://raw.githubusercontent.com/lihkir/Uninorte/main/AppliedStatisticMS/DataVisualizationRPython/Lectures/Python/PythonDataSets/renewable-energy-consumption-by-country.csv)
# 
# Su tarea consiste en crear mapas coropléticos de la producción y el consumo totales de energía renovable y el consumo en diferentes países del mundo animados en base a los años de años de producción/consumo entre (excluyendo) 2007 y 2017.
# 
# - **Pasos principales**
#     1. Cargar el conjunto de datos de producción de energía renovable.
#     2. Ordenar el `DataFrame` de producción en función de la característica *Year*.
#     3. Generar un mapa de coropletas para la producción de energía renovable utilizando el módulo `plotly express` animado en función del *Year*.
#     4. Actualice el diseño para incluir un estilo de proyección adecuado y un texto de título, y luego muestre la figura.
#     5. Cargue el conjunto de datos de consumo de energía renovable.
#     6. Convierta el `DataFrame` de consumo a un formato adecuado para su visualización.
#     7. Ordenar el `DataFrame` de consumo en base a la característica *Year*.
#     8. Generar un mapa coroplético para el consumo de energía renovable utilizando el módulo `plotly express` animado en función del año.
#     9. Actualice el diseño para incluir un estilo de proyección adecuado y un texto de título, y luego muestre la figura.

# Primero, cargaremos los modulos que se usarán en todo el taller:

# In[1]:


import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from time import mktime
import time
import plotly.express as px
from plotly.offline import plot as px_off_plot
import plotly.graph_objects as go
import json
from urllib.request import urlopen
from IPython.display import IFrame, display, HTML

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
from numpy import log

import psycopg2
from psycopg2 import Error
from sqlalchemy import create_engine

import finnhub
import warnings
warnings.filterwarnings("ignore")


# Primero, leamos los archivos de la **URL**, y carguemoslos en un **DataFrame** de **pandas**:

# In[2]:


energy_prod_url = "https://raw.githubusercontent.com/lihkir/Uninorte/main/AppliedStatisticMS/DataVisualizationRPython/Lectures/Python/PythonDataSets/share-of-electricity-production-from-renewable-sources.csv"
prod_df = pd.read_csv(energy_prod_url)
prod_df.head()


# In[21]:


energy_cons_url = "https://raw.githubusercontent.com/lihkir/Uninorte/main/AppliedStatisticMS/DataVisualizationRPython/Lectures/Python/PythonDataSets/renewable-energy-consumption-by-country.csv"
comsup_df = pd.read_csv(energy_cons_url)
comsup_df.head()


# Primero, revisemos si el dataframe de producción de energía renovable tiene valores faltantes.

# In[14]:


(prod_df.isnull().sum()/prod_df.shape[0])*100


# Como se puede observar, solo la columna de `Code` tiene valores faltantes, con el 5.54% de valores faltantes. Si bien, lo ideal sería imputar estos datos, para el ejercicio que realiceramos no afectará los cálculos o nivel de producción. Esto nos afectará solamente en la presentación del código, por lo cual, por lo pronto, dejaremos el dataframe así.
# 
# Ordenemos ahora el dataframe relacionado con la producción de energía renovable en función del año.

# In[4]:


prod_df = prod_df.sort_values(by="Year",ascending=True)
prod_df.head()


# Generemos un mapa de cloropletas para la producción de energía renovable utilizando el módulo de `plotly express`, con un slider para mostrar el cambio a través de los años de manera dinámica:

# In[6]:


fig = px.choropleth(prod_df, locations="Code", # colunm containing ISO 3166 country codes
                    color = "Renewable electricity (% electricity production)", # column by which to color-code
                    hover_name = "Country", # column to display in hover information
                    color_continuous_scale = px.colors.sequential.Plasma,
                    animation_frame="Year")

fig.update_layout(
    # add a title text for the plot
    title_text = 'Reneweable energy production per Country (% Electricity Prod)',
    # set projection style for the plot
    geo = dict(projection={'type':'natural earth'}) # by default, projection type is set to 'equirectangular'
)
fig.show(renderer="svg", width=900, height=410)


# In[7]:


#Para generar la visualización a tráves de los años
px_off_plot(fig)


# Como se puede observar en la figura, en el año 1990, los países que producen más energía renovable son mayoritariamente de Latinoamérica y África, entre ellos, Paraguay, Costa Rica, Honduras, Brasil, ambos Congos, Zambia, Camerún, entre otros. En Europa, los países con más producción son Noruega e Islandia. Se puede observar que los países que tienen menos producción de energía renovable están ubicados en Asia-Pacífico, Europa, el Norte de África y Estados Unidos.
# 
# 
# En 2014, se observa que las regiones que producían más energía renovable han bajado considerablemente su producción de energía renovable, siguen liderando a nivel mundial la producción (África central y Latinoamérica). De los países que siguen con un porcentaje del 99% de producción, Paraguay, República Democrática del Congo y Zambia siguen liderando.
# 
# En Europa, se nota un aumento en la producción de energía renovable, siendo Noruega e Islandia los países que siguen teniendo la mayor producción de energía renovable. Las regiones con menor producción de energía renovable siguen siendo las mismas que hace 14 años, Asia-Pacífico, el Norte de África y Estados Unidos.

# Miremos ahora el dataframe relacionado con el consumo de energía renovable en el mundo. Para esto, primero lo transformaremos. Primero imputaremos los valores faltantes.
# 
# Como estamos hablando de consumo de energía, asumiremos que el hecho de que no haya información sobre una energía renovable específica, implica que no hay capacidad de generación de dicha energía en el país, por lo que podríamos reemplazar estos valores por cero sin ningún problema.

# In[22]:


#Primero revisando las columnas que tienen valores faltantes
(comsup_df.isnull().sum()/comsup_df.shape[0])*100


# Como se observa, las columnas de `Traditional Biofuels` y `Hydropower` tienen el 100% de valores faltantes. Estas columnas no nos están aportando ninguna información relevante, por lo cual las borraremos.

# In[23]:


comsup_df = comsup_df.drop(columns=["Traditional biofuels", "Hydropower"])
comsup_df.describe()


# Como se puede observar, contamos con cinco columnas. Una que representa el año, tres que representan distintos tipos de energía renovable, y la última columna que nos indica el total de consumo de energía renovable. Para el gráfico que realizaremos, usaremos en primera instancia el total de consumo de energía renovable.
# 
# Primero organicemos el dataframe por año.

# In[24]:


comsup_df = comsup_df.sort_values(by="Year",ascending=True)
comsup_df.head()


# Prosigamos ahora con el gráfico para el consumo total de energía renovable por país a través de los años.

# In[27]:


fig = px.choropleth(comsup_df, locations="Code", # colunm containing ISO 3166 country codes
                    color = "Total", # column by which to color-code
                    hover_name = "Country", # column to display in hover information
                    color_continuous_scale = px.colors.sequential.Plasma,
                    animation_frame="Year")

fig.update_layout(
    # add a title text for the plot
    title_text = 'Reneweable energy comsuption per Country (TWh)',
    # set projection style for the plot
    geo = dict(projection={'type':'natural earth'}) # by default, projection type is set to 'equirectangular'
)
fig.show(renderer="svg", width=900, height=410)


# In[28]:


#Para generar la visualización offline a tráves de los años
px_off_plot(fig)


# Como se puede observar en la animación, poco países consumían energía renovable en 1965, y la cantidad era muy pequeña. El país que más consumía renovables era Italia con 2.6TWh. Al pasar de los años, más países empiezan a consumir energía renovable, y la cantidad de la energía consumida aumenta considerablemente. En 2016, los países con más consumo de energía renovable son China, Estados Unidos, y Alemania, con 380TWh, 370TWh y 167TWh, respectivamente.

# ## Ejercicio 2.2
# 
# -  Considere la serie de tiempo asociada con los futuros de la criptomoneda **Bitcoin** desde que comenzó a tradearse hasta la fecha del día de hoy. Utilice la **API** de `finnhub` para obtener esta serie de tiempo. Recuerde que para esto debe crear primero una cuenta, y así obtener su **API Key** para realizar la conexión de forma efectiva.
# 
# - Repita **TODOS** los pasos indicados en esta sección para encontrar modelos **ARIMA** para predecir el precio de **Bitcoin** con los siguientes horizontes: **7 días, 14 días, 21 días, 28 días**. Utilizar siempre predicciones usando *rolling* con ventana de predicción continua de un día. Cualquier cantidad de pasos extra para enriqucer su anális predictivo serán aceptados siempre y cuando sean acordes con lo que indica la teoría de análisis de series de tiempo. 
# 
# - Realice una tabla de error utlizando las métricas: `MAPE, MAE, RMSE, MSE`. 
# 
# - Repita el análisis desarrollado en los pasos anteriores considerando ahora el criterio de **inferencia Bayesiana (BIC)** para encontrar el mejor modelo **ARIMA** y compare los errores con aquellos obtenidos con el criterio de **Akaike**.
# 
# - Escriba en cada paso las conclusiones y análisis estadísticos asociados con los resultados obtenidos. Realice un test de normalidad para los residuales obtenidos para cada predicción, en cada caso agregue las correspondientes conclusiones estadísticas. Figuras y algoritmos que no estén acompañados de una conclusión, descripción y análisis estadístico, no serán tenidas en cuenta.

# Los modulos que vamos a usar fueron cargados al principio del taller, ahora solo cargaremos el tema de paper, para que las gráficas se vean mejor.

# In[31]:


sns.set_theme()
sns.set_context("paper")


# Para conectarnos a la API de FinnHub a través de Python, debemos tener una cuenta creada para usar la API key de dicha cuenta. En la siguiente celda, creamos una instancia del cliente de FinnHub con nuestra API key.

# In[32]:


finnhub_client = finnhub.Client(api_key="c5tpk32ad3i9n9aj0u3g")


# El símbolo que vamos a usar para extraer la información es `BINANCE:BTCUSDT`:

# In[36]:


print(finnhub_client.symbol_lookup('BINANCE:BTCUSDT'))


# Teniendo el cliente instanciado, y el código a utilizar, ahora extraeremos toda la información asociada al par `BTC:USDT`:

# In[67]:


#Par BTCUSDT
stock = 'BINANCE:BTCUSDT'
resolution = 'D'

#Tiempo a extraer
end_date = datetime.now()
start_date = datetime(year=2017, month=8, day=14)

#FinnHub maneja tiempo en Unix, por lo cual convertiremos el tiempo de datetime a unix, y llamaremos la función stock_candles
end = int(end_date.timestamp())
start = int(start_date.timestamp())

BTC_res = finnhub_client.stock_candles(stock, resolution, start, end)
BTC_df = pd.DataFrame(BTC_res)

BTC_df.head()


# In[68]:


BTC_df.shape


# In[69]:


#Función utilizada para convertir t a un formato legible
def date_format(date_col):
    return datetime.fromtimestamp(date_col).strftime('%Y-%m-%d')


# In[70]:


date_format(BTC_df.t.max())


# Como se puede observar, incluso aunque coloquemos las fechas correctamente, FinnHub tiene una restricción sobre el número de columnas que nos puede dar por consulta, de 500. Lo que haremos entonces es hacer múltiples consultas para completar toda la historia transaccional del BTC.

# In[71]:


#Cambiando la fecha inicial
start_date = datetime(year=2018, month=12, day=29)
start = int(start_date.timestamp())

BTC_res = finnhub_client.stock_candles(stock, resolution, start, end)
BTC_df2 = pd.DataFrame(BTC_res)

date_format(BTC_df2.t.max())


# In[72]:


#Cambiando la fecha inicial
start_date = datetime(year=2020, month=5, day=12)
start = int(start_date.timestamp())

BTC_res = finnhub_client.stock_candles(stock, resolution, start, end)
BTC_df3 = pd.DataFrame(BTC_res)

date_format(BTC_df3.t.max())


# In[73]:


#Cambiando la fecha inicial
start_date = datetime(year=2021, month=9, day=24)
start = int(start_date.timestamp())

BTC_res = finnhub_client.stock_candles(stock, resolution, start, end)
BTC_df4 = pd.DataFrame(BTC_res)

date_format(BTC_df4.t.max())


# Ahora procedemos a unir todos los dataframes en uno solo con toda la información:
# 

# In[76]:


BTC_df = pd.concat([BTC_df, BTC_df2, BTC_df3, BTC_df4], ignore_index=True, axis=0)
BTC_df.head()


# In[77]:


date_format(BTC_df.t.min())


# In[78]:


date_format(BTC_df.t.max())


# Una vez completada la extracción de los datos, cambiemos los nombres de las columnas, eliminemos la columna `s` que no nos aportará nada al modelo, y cambiemos también el formato del tiempo.

# In[79]:


#Cambiando el formato del tiempo
BTC_df['t'] = BTC_df['t'].apply(date_format)

#Cambiando los nombres de las columnas, y eliminando la columna s
BTC_df.rename(columns = {'c':'Close','h':'High','l':'Low','o':'Open','t':'Date','v':'Volume'}, inplace = True)
BTC_df.drop('s', axis=1, inplace = True)
BTC_df.head()


# Demosle un vistazo a la historia del BTC, y grafiquemos la serie temporal:

# In[83]:


plt.figure(figsize = (12,7))
ax = sns.lineplot(BTC_df.Date, BTC_df.Close)
freq = int(40)
xtix = ax.get_xticks()
ax.set_xticks(xtix[::freq])
plt.tight_layout()
plt.xticks(rotation=50);


# Se puede observar que desde que se empezó a tranzar, el precio de Bitcoin empezó alrededor de los USD\$5.000 y tuvó una tendencia alcista hasta casi que alcanzar los USD\$20.000 en diciembre de 2017. Luego de esto, tuvo una fuerte caída y volvió a los mismos niveles. Desde 2018, estuvo por debajo de los USD\$10.000, hasta mediados de 2019 donde supero levemente esa barrera. Luego de superarla, tuvo unas cuantas subidas y bajadas, pero el precio se mantenía alrededor de este valor, hasta finales de 2020, donde tuvo un crecimiento rápido, hasta alcanzar los USD\$62.000 aproximadamente, en Marzo de 2021. Luego de eso, tuvo una corrección, pero volvió a subir para alcanzar su máximo histórico situado en torno a los USD\$68.000 en Octubre de 2021. Actualmente, presenta una tendencia bajista, y se sitúa en torno a los USD\$53.000.

# Para continuar, antes de iniciar los modelos, realicemos una prueba estadística para verificar si la serie temporal asociada al par BTC:USDT es estacionaria o no. La prueba que usaremos es la de `Dickey-Fuller`.

# In[86]:


result = adfuller(BTC_df.Close)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


# Como se puede observar, con un nivel de significancia del 5%, el P-valor es mayor que el alpha, por lo cual, la hipótesis nula no se rechaza. Esto significa que no hay evidencia suficiente para afirmar que la serie temporal es estacionaria. Miremos ahora las figuras de autocorrelación, para verificar cuál es el orden de integración necesario para llevar nuestra serie de tiempo no estacionaria a una estacionaria. 

# In[90]:


plt.rcParams.update({'figure.figsize': (15,15)})

fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(BTC_df.Close); axes[0, 0].set_title('Original Series')
plot_acf(BTC_df.Close, ax=axes[0, 1], lags = 1500);

axes[1, 0].plot(BTC_df.Close.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(BTC_df.Close.diff().dropna(), ax=axes[1, 1], lags = 1500);

axes[2, 0].plot(BTC_df.Close.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(BTC_df.Close.diff().diff().dropna(), ax=axes[2, 1], lags = 1500);


# Se puede notar el decaimiento geométrico en la primera figura de autocorrelación (la serie original), que baja desde la parte positiva con una tendencia lineal. Esto también nos muestra y nos confirma la prueba de `Dickey-Fuller`, que la serie no es estacionaria. También se puede ver en el segundo gráfico de autocorrelación, correspondiente a la segunda derivada, que el retardo entra rápidamente en la zona negativa del gráfico, por lo cual nos indica que la serie fue sobrediferenciada.
# 
# Prosigamos ahora si con los modelos **ARIMA** para predecir el precio del **Bitcoin**. Como usaremos varios horizontes, crearemos una función general, que calcule el mejor modelo, y tendremos como entrada el horizonte.

# In[115]:


#Definiendo la función para entrenar los modelos ARIMA
def ARIMA_model(horizon, metric="aic", pq_rng=range(5), d_rng=range(3)):
    #Metric indicará la métrica a utilizar, aic o bic.
    #Calculando train y test size
    n_BTC = len(BTC_df.Close); n_test = horizon # This can be changed
    train_size = n_BTC - n_test

    #Creando cada dataframe para train y test
    train = BTC_df.Close.iloc[:train_size]
    test = BTC_df.Close.iloc[train_size:train_size + n_test] 
    dates_test = BTC_df.Date.iloc[train_size:train_size + n_test] 

    #Encontrando el mejor modelo usando la métrica seleccionada
    best_metric = np.inf
    best_order = None
    best_mdl = None
    #exceptions = []

    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = ARIMA(train, order=(i,d,j)).fit(method='innovations_mle')
                    if metric=="bic":
                        tmp_metric = tmp_mdl.bic
                    else:
                        tmp_metric = tmp_mdl.aic                
                    
                    if tmp_metric < best_metric:
                        best_metric = tmp_metric
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except Exception as e: 
                    #exceptions.append(e)
                    continue

    #Regresando los valores que se encontraron, junto con el train y el test set
    return(train, test, dates_test, best_mdl, best_order, best_metric)#, exceptions)


# También usaremos la función mostrada en clase por el profesor, para calcular el error de predicción cometido en las predicciones, usando las métrica `MAPE, MAE, RMSE, MSE`:

# In[93]:


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


# Calculemos primero, los mejores modelos ARIMA, usando horizontes de 7, 14, 21 y 28 días, y usando el criterio de Akaike. Mantendremos constantes los rangos usados en clase para calcular el mejor orden.

# In[116]:


horizons = [7, 14, 21, 28]

best_models = []

for horizon in horizons:
    best_m = ARIMA_model(horizon=horizon)
    best_models.append(best_m)


# Revisemos para cada horizonte, cuál es el mejor modelo usando el criterio de Akaike:

# In[121]:


for model in best_models:
    print('aic: {:6.5f} | order: {}'.format(model[5], model[4]))


# Con los mejores modelos según el criterio de Akaike, podemos usarlos para predecir los valores futuros usando rolling. Para esto, usaremos los modelos guardados en la variable `best_models`. Adicionalmente, usaremos también la función utilizada en clase, con unos pequeños cambios (Ya calculamos el modelo y lo entrenamos, así que solo haremos las predicciones).

# In[133]:


def arima_rolling(history, test, model_fit):
    
    predictions = list()
    for t in range(len(test)):
        output = model_fit.forecast()
        yhat = output.values[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        #print('predicted=%f, expected=%f' % (yhat, obs))
        
    return predictions


# In[136]:


#Variable para guardar las predicciones de cada modelo
predictions_list = []

#for loop para recorrer los modelos y generar las predicciones
for var in best_models:
    pred = arima_rolling(var[0].to_list(), var[1].to_list(), var[3])
    predictions_list.append(pred)

#Variable para guardar las métricas
df_metricas = pd.DataFrame()

#Usamos la función forecast_accuracy para calcular las métricas de cada modelo
for i, pred in enumerate(predictions_list):
    df_temp = forecast_accuracy(np.array(pred),np.array(best_models[i][1]), "Model for {} days".format(7*(i+1)))
    df_metricas = pd.concat([df_metricas, df_temp], axis=0)

df_metricas


# Como se puede observar, tenemos errores bastantes altos para los modelos. El error más bajo fue para el modelo de 7 días. Esto se debe a la naturaleza de la serie temporal que estamos analizando, ya que por su alta volatilidad, es díficil predecir precios a más tiempo. También, podríamos tratar de tunear mejor los modelos, al usar más parámetros, pero esto aumentaría considerablemente el tiempo de cómputo.
# 
# Ahora, calculemos la misma tabla y los mismos modelos, pero esta vez teniendo en cuenta el criterio de inferencia Bayesiana para la selección de los mejores modelos. De esta manera podremos comparar las métricas arrojadas por la selección de cada criterio:

# In[137]:


horizons = [7, 14, 21, 28]

best_models_bic = []

for horizon in horizons:
    best_m = ARIMA_model(horizon=horizon, metric="bic")
    best_models_bic.append(best_m)


# In[138]:


for model in best_models_bic:
    print('bic: {:6.5f} | order: {}'.format(model[5], model[4]))


# Si bien, con el criterio Bayesiano estamos obteniendo valores un poquito más altos que con el criterio de Akaike, a la final, con los dos criterios se obtiene el mismo orden para cada horizonte evaluado, por lo cual, las predicciones de ambos modelos serán las mismas, y de esta manera, sus errores también lo serán. Por esta razón, no calcularemos la tabla de las métricas para los modelos generados mediante el criterio de inferencia bayesiana, ya que obtuvimos modelos del mismo orden.
# 
# Por último, grafiquemos entonces los resultados de las predicciones:

# In[149]:


fig, ax=plt.subplots(2,2, figsize=(15,15))

sns.lineplot(best_models[0][2], best_models[0][1], label="Test",ax=ax[0,0]) 
sns.lineplot(best_models[0][2], predictions_list[0], label="Forecast",ax=ax[0,0]) #Modelo con horizonte de siete días
ax[0,0].set_xlabel("Fecha")
ax[0,0].set_ylabel("Precio de Cierre")
ax[0,0].set_title("Forecast BTC:USDT - 7 días")

sns.lineplot(best_models[1][2], best_models[1][1], label="Test",ax=ax[0,1]) 
sns.lineplot(best_models[1][2], predictions_list[1], label="Forecast",ax=ax[0,1]) #Modelo con horizonte de catorce días
ax[0,1].set_xlabel("Fecha")
ax[0,1].set_ylabel("Precio de Cierre")
ax[0,1].set_title("Forecast BTC:USDT - 14 días")
ax[0,1].tick_params(axis='x', rotation=60)

sns.lineplot(best_models[2][2], best_models[2][1], label="Test",ax=ax[1,0]) 
sns.lineplot(best_models[2][2], predictions_list[2], label="Forecast",ax=ax[1,0]) #Modelo con horizonte de veintiun días
ax[1,0].set_xlabel("Fecha")
ax[1,0].set_ylabel("Precio de Cierre")
ax[1,0].set_title("Forecast BTC:USDT - 21 días")
ax[1,0].tick_params(axis='x', rotation=60)

sns.lineplot(best_models[3][2], best_models[3][1], label="Test",ax=ax[1,1]) 
sns.lineplot(best_models[3][2], predictions_list[3], label="Forecast",ax=ax[1,1]) #Modelo con horizonte de veintiocho días
ax[1,1].set_xlabel("Fecha")
ax[1,1].set_ylabel("Precio de Cierre")
ax[1,1].set_title("Forecast BTC:USDT - 28 días")
ax[1,1].tick_params(axis='x', rotation=60)

plt.tight_layout()
plt.show()


# Como habíamos vaticinado al principio, las predicciones son muy malas. Para mejorarlas, necesitamos seguir iterando sobre los hiper-parámetros (en este caso, el orden del modelo), o probar con otros parámetros del modelo ARIMA, para ver si encontramos una combinación que funcione mejor.

# ## Ejercicio 2.3
# 
# Determine las características que se visualizan en un gráfico de dispersión. Se le da el conjunto de datos co2.csv y se le pide que proporcione información sobre él, como por ejemplo, qué tipo de patrones existen, si hay tendencias entre las características, etc. Es necesario que su visualización final transmita información significativa. Para conseguirlo, va a crear visualizaciones para diferentes emparejamientos de características para entender están correlacionadas y, por lo tanto, vale la pena visualizarlas.
# 
# **Pasos principales**
# 
# - Importe las bibliotecas necesarias.
# - Vuelva a crear el `DataFrame`. Desde el `DataFrame` `gm` incluya las columnas `population`, `fertility`, y `life`
# - Visualiza la relación entre el `co2` y `life` utilizando un gráfico de dispersión, con el nombre del país como información en la herramienta hover y el año como deslizador.
# - Comprueba la correlación entre `co2` y `life`.
# - Visualiza la relación entre `co2` y `fertility` mediante un gráfico de dispersión, con el nombre del país como información en la herramienta hover y el año como deslizador.
# - Comprueba la correlación entre `co2` y `fertility`

# Cargamos el conjuto de datos co2.csv y gapminder.csv

# In[3]:


url_co2 = 'https://raw.githubusercontent.com/lihkir/Uninorte/main/AppliedStatisticMS/DataVisualizationRPython/Lectures/Python/PythonDataSets/co2.csv'
co2 = pd.read_csv(url_co2)
co2.head()


# In[4]:


url_gm = 'https://raw.githubusercontent.com/lihkir/Uninorte/main/AppliedStatisticMS/DataVisualizationRPython/Lectures/Python/PythonDataSets/gapminder.csv'
gm = pd.read_csv(url_gm)
gm.head()


# La información de los dos dataframes anteriores, los unificaremos dejando solo las columnas de interés:

# In[5]:


df_gm = gm[['Country', 'region']].drop_duplicates()
df_w_regions = pd.merge(co2, df_gm, left_on='country', right_on='Country', how='inner')
df_w_regions = df_w_regions.drop('Country', axis='columns')

new_co2 = pd.melt(df_w_regions, id_vars=['country', 'region'])
columns = ['country', 'region', 'year', 'co2']
new_co2.columns = columns
df_co2 = new_co2[new_co2['year'].astype('int64') > 1963]
df_co2 = df_co2.sort_values(by=['country', 'year'])
df_co2['year'] = df_co2['year'].astype('int64')

df_gdp = gm[['Country', 'Year', 'fertility', 'life', 'population']]
df_gdp.columns = ['country', 'year', 'fertility', 'life', 'population']

data = pd.merge(df_co2, df_gdp, on=['country', 'year'], how='left')
data = data.dropna()

data.head()


# Visualicemos la relación entre el co2 y life. El  coeficiente de correlación nos indica que hay una correlación positiva moderada.

# In[6]:


np_co2 = np.array(data['co2'])
np_life = np.array(data['life'])
np.corrcoef(np_co2, np_life)


# In[7]:


xmin, xmax = min(data.life), max(data.life)
ymin, ymax = min(data.co2), max(data.co2)
fig = px.scatter(data, 
                 x="life", y="co2", 
                 animation_frame="year",
                 animation_group="country",
                 color="region",
                 facet_col="region",
                 hover_name="country",
                 size="co2",
                 width=1500, height=400, 
                 log_x=True,
                 range_x=[xmin,xmax],
                 range_y=[ymin,ymax])
fig.show()


# Si vemos por ejemplo el año 2013 tenemos que para "East Asia & Pacific" tenemos que las mayores emisiones de co2 se dan cuanto se tiene una esperanza de vida mayor, este mismo coportamiento se aprecia en "America". Mientras que en "South Asia" no hay gran diferencia entre las emisiones de co2 y la esperanza de vida, se puede decir que son independientes.
# 
# Si miramos las emisiones de co2 para "South Asia" a lo largo de los años (1964 - 2013) vemos como las emisiones de co2 se mantienen en el mismo rango.

# Al revisar ahora la relación entre co2 y fertility ,tenemos que el hay una correlación negativa moderada según el valor que nos arroja el coeficiente de correlación

# In[8]:


np_fertility = np.array(data['fertility'])
np.corrcoef(np_co2, np_fertility)


# In[9]:


xmin, xmax = min(data.fertility), max(data.fertility)
fig = px.scatter(data, 
                 x="fertility", y="co2", 
                 animation_frame="year",
                 hover_name="country",
                 color="region",
                 facet_col="region",
                 size="co2",
                 width=1400, height=400, 
                 log_x=True,
                 range_x=[xmin,xmax],
                 range_y=[ymin,ymax])
fig.show()


# Si nos enfocamos por ejemplo en el año 2000, vemos que las emisiones de calor más grande se tienen en "Middle East & North Africa".
# 
# Graficamente se nota por ejemplo que en "East Asia & Pacific" y "America" a menos fertilidad mayores emisiones de c02, a lo largo de todos los años analizados 

# In[10]:


xmin, xmax = min(data.fertility), max(data.fertility)
fig = px.scatter(data, 
                 x="fertility", y="co2", 
                 animation_frame="year",
                 hover_name="country",
                 color="region",
                 size="co2",
                 width=1000, height=400, 
                 log_x=True,
                 range_x=[xmin,xmax],
                 range_y=[ymin,ymax])
fig.show()


# Al comparar las emisiones de c02 entre las regiones vemos que "Middle East & North Africa" a lo largo de los años ha tenido las mayores emisiones. 

# ## Ejercicio 2.4
# - La primera tarea es poner a funcionar todo lo realizado en esta sección y entenderlo. Luego de esto creará una nueva tabla que llamaremos `links`, la cual contendrá información acerca de link de páginas web. La tabla debe ser creada siguiendo las siguientes instrucciones. En la siguiente revisaremos con detalle el significado de cada consulta `SQL`
# 
# ``` mysql
# CREATE TABLE links (
# 	id SERIAL PRIMARY KEY,
# 	url VARCHAR(255) NOT NULL,
# 	name VARCHAR(255) NOT NULL,
# 	description VARCHAR (255),
#     last_update DATE
# );
# ```
# - Luego insertará datos a la tabla usando las siguientes ordenes. En cada caso debe mostrar el output generado por **pgAdmin**, así como también la actualización de **Heroku Postgres**
# 
# ``` mysql
# INSERT INTO links (url, name)
# VALUES('https://www.postgresqltutorial.com','PostgreSQL Tutorial');
# ```
# 
# ``` mysql
# INSERT INTO links (url, name)
# VALUES('http://www.oreilly.com','O''Reilly Media');
# ```
# 
# ``` mysql
# INSERT INTO links (url, name)
# VALUES('http://www.oreilly.com','O''Reilly Media');
# ```
# 
# ``` mysql
# INSERT INTO links (url, name, last_update)
# VALUES('https://www.google.com','Google','2013-06-01');
# ```
# 
# ``` mysql
# INSERT INTO links (url, name)
# VALUES('http://www.postgresql.org','PostgreSQL') 
# RETURNING id;
# ```
# 
# - Importe los datos del precio de **Cardano USD (ADA-USD)** en su base de datos **Herokus Postgres**, teniendo en cuenta lo explicado. Luego dibuje un gráfico de candlestick para la criptomoneda. En el siguiente link encontrará el **CSV** de **Cardano**: [Cardano USD (ADA-USD)](https://raw.githubusercontent.com/lihkir/Uninorte/main/AppliedStatisticMS/DataVisualizationRPython/Lectures/Python/PythonDataSets/ADA-USD.csv)

# Primero creamos conexión con nuestra base de datos PostgresSql que se encuentra en Heroku.
# 
# ![](images/0tablas.png)
# ![](images/0estadoheroku.png)

# In[15]:


connection = psycopg2.connect(user="uqtxfaqjjcxggw",
                                  password="916d311356954de6a99118d13578bb9d1b47bdc86cb8360a60b9606293bd882d",
                                  host="ec2-3-229-43-149.compute-1.amazonaws.com",
                                  port="5432",
                                  database="d41lsl8qgestjf")
cursor = connection.cursor()


# Procedemos a crear una nueva tabla como sigue:

# In[16]:


cursor.execute('''DROP TABLE IF EXISTS links''')
cursor.execute('''CREATE TABLE links(
id SERIAL PRIMARY KEY NOT NULL,
url VARCHAR(255) NOT NULL,
name VARCHAR(255) NOT NULL,
description VARCHAR (255),
last_update DATE);
''')
print("Table created successfully")
   
connection.commit()


# ![](images/1create_links.png)
# ![](images/1estadoheroku.png)

# Ahora, insertaremos registros en la tabla:

# In[17]:


cursor.execute("INSERT INTO links (url, name) VALUES ('https://www.postgresqltutorial.com','PostgreSQL Tutorial')");

cursor.execute("INSERT INTO links (url, name) VALUES ('http://www.oreilly.com','O''Reilly Media')");

cursor.execute("INSERT INTO links (url, name) VALUES ('http://www.oreilly.com','O''Reilly Media')");

cursor.execute("INSERT INTO links (url, name,last_update) VALUES ('https://www.google.com','Google','2013-06-01')");

cursor.execute("INSERT INTO links (url, name) VALUES ('http://www.postgresql.org','PostgreSQL')");

connection.commit()
print("Records created successfully")
connection.close()


# ![](images/2insertlinks.png)
# ![](images/2estadoheroku.png)

# Importamos los datos del precio de Cardano USD (ADA-USD)

# In[18]:


df_cardano = pd.read_csv("https://raw.githubusercontent.com/lihkir/Uninorte/main/AppliedStatisticMS/DataVisualizationRPython/Lectures/Python/PythonDataSets/ADA-USD.csv")
df_cardano.head()


# Cargamos el dataset en nuestra base  de  datos en Heroku

# In[19]:


engine = create_engine('postgresql://uqtxfaqjjcxggw:916d311356954de6a99118d13578bb9d1b47bdc86cb8360a60b9606293bd882d@ec2-3-229-43-149.compute-1.amazonaws.com:5432/d41lsl8qgestjf')
df_cardano.to_sql('cardano', engine, if_exists = 'replace', index=False, method='multi')


# ![](images/3insertcardano.png)
# ![](images/3estadoheroku.png)

# Leemos los datos desde nuestra tabla "cardano" de la base de datos

# In[20]:


connection = psycopg2.connect(user="uqtxfaqjjcxggw",
                                  password="916d311356954de6a99118d13578bb9d1b47bdc86cb8360a60b9606293bd882d",
                                  host="ec2-3-229-43-149.compute-1.amazonaws.com",
                                  port="5432",
                                  database="d41lsl8qgestjf")

cursor = connection.cursor()
cursor.execute("SELECT * from cardano;")
close = cursor.fetchall()
cardano_sql = pd.DataFrame(close)
display(cardano_sql.head())


# Vemos que nuestro dataframe no tiene los nombres de las columnas, entonces procedemos a ajustarlo de acuerdo a los nombres de las columnas en nuestra tabla dentro de la base  de  datos.

# In[21]:


cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='cardano';")
names = cursor.fetchall()

print("Operation done successfully")
connection.close()

names_list = [name[0] for name in names]
cardano_sql.columns = names_list 
cardano_sql = cardano_sql.sort_values(by = "Date")
cardano_sql.head()


# Por último realizamos el candlestick para la criptomoneda

# In[22]:


fig = go.Figure(data=[go.Candlestick(x = cardano_sql.Date,
                                     open = cardano_sql.Open, 
                                     high = cardano_sql.High,
                                     low = cardano_sql.Low, 
                                     close = cardano_sql.Close)
                     ])
fig.update_layout(
    title="Cardano USD (ADA-USD)",
    xaxis_title="Day",
    yaxis_title="ADA-USD",
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="RebeccaPurple"
    )
)
fig.update_layout(xaxis_rangeslider_visible=False)
fig.show()


# Vemos como por ejemplo, los precios de Cardano, tuvieron una caída luego de enero de 2018 y su precio se mantuvo muy cercano a los 0.03 USD hasta julio de 2020 donde empezó a subir su precio, logrando alcanzar casi los 3 USD para septiembre de 2021. Actualmente el precio de Cardano se encuentra a la baja con valroes al rededor de los 2.5 USD.
# 
# Ahora si nos enfocamos en el mes de agosto de 2021, vemos que es mayor el volumen de compras (velas verdes) que las ventas (velas rojas).
# 
# Si revisamos el último registro que tenemos, vemos que para el 14 de Octubre de 2021, el precio de Cardano abrió en 2.195 USD y cerró 2.179 USD, con valores mínimo y maximo de 2.162 y 2.221, respectivamente.
