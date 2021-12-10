#!/usr/bin/env python
# coding: utf-8

# # Análisis Exploratorio de Datos - Proyecto Final Python
# 
# Para este proyecto, se analizarán las cinco empresas colombianas cotizadas en la bolsa de valores de Nueva York. En esta primera fase, se descargaron los datos históricos del comportamiento de las acciones durante los últimos cinco años de Yahoo Finance. Las empresas seleccionadas fueron:
# 
# * Ecopetrol.
# * Bancolombia.
# * Grupo Aval.
# * Avianca.
# * Tecnoglass.
# 
# Para la fase final, se espera conectar el dashboard a la API de FinnHub para tener los datos más recientes, y descargar esta información en la base de datos de Heroku Postgres. Por ahora, continuemos con el análisis exploratorio de los datos.

# ## Modulos utilizados
# 
# Primero, carguemos los modulos que necesitamos:

# In[1]:


#Modulos nativos de Python
from datetime import datetime, timedelta
import pickle
import warnings

#Modulos de visualización
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

#Modulos para gestionar dataframes, cálculos con vectores y matrices
import numpy  as np
import pandas as pd

#Modulos para las pruebas estadísticas para series de tiempo
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

#Modulos para la generación de modelos
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

#Modulos para manejar la base de datos
import psycopg2
from sqlalchemy import create_engine

#Modulos para obtener datos de las acciones actualizadas
import finnhub

warnings.filterwarnings("ignore")


# Configuremos ahora el tema que se usará en las gráficas.

# In[2]:


sns.set_theme()
sns.set_context("paper")


# ## Fuente de datos históricos
# 
# Ahora cargaremos los archivos descargados de Yahoo Finance para cada una de las acciones:

# In[3]:


#Paths relativos para los archivos de cada acción
ec_path = "./files/EC.csv"  #Ecopetrol
aval_path = "./files/AVAL.csv"  #Grupo AVAL
bc_path = "./files/CIB.csv"  #Grupo Bancolombia
tgls_path = "./files/TGLS.csv"  #Tecnoglass
av_path = "./files/AVHOQ.csv"  #Avianca

#Leyendo los archivos csv con pandas
ec_df = pd.read_csv(ec_path).drop(columns=["Adj Close"])
aval_df = pd.read_csv(aval_path).drop(columns=["Adj Close"])
bc_df = pd.read_csv(bc_path).drop(columns=["Adj Close"])
tgls_df = pd.read_csv(tgls_path).drop(columns=["Adj Close"])
av_df = pd.read_csv(av_path).drop(columns=["Adj Close"])


# In[4]:


ec_df.head()


# Recordemos, que estas bases de datos contienen los últimos cinco años de los precios y volumenes de cada acción. Lo siguiente que haremos, será conectarnos a la base de datos de Heroku-Postgress, y crear tablas para cada dataframe:

# In[5]:


#Creando el motor para cargar los datos a nuestra base de datos
engine = create_engine('postgresql://snfifneqljoybc:e1aaa689afeaaff2f75921f4fd6ca3816f6ebf33bdaf0c3366b9db4e5fe218fe@ec2-3-95-130-249.compute-1.amazonaws.com:5432/d3go0j5dkhm396')


# In[6]:


ec_df.to_sql('EC', engine, if_exists = 'replace', index=False, method='multi')
aval_df.to_sql('AVAL', engine, if_exists = 'replace', index=False, method='multi')
bc_df.to_sql('CIB', engine, if_exists = 'replace', index=False, method='multi')
tgls_df.to_sql('TGLS', engine, if_exists = 'replace', index=False, method='multi')
av_df.to_sql('AVHOQ', engine, if_exists = 'replace', index=False, method='multi')


# Ya tenemos las bases de datos creadas en Heroku, así que seguiremos trabajando con los DataFrames originales. 
# 
# ![](images/01_Heroku_Postgres.png)
# 
# 
# Verifiquemos ahora las fechas iniciales y finales para cada DataFrame:

# In[6]:


df = [ec_df, aval_df, bc_df, tgls_df, av_df]

for d in df:
    print(d.Date.min())
    print(d.Date.max())


# Cómo se observa, la última fecha del DataFrame no es la fecha actual, por lo cual, crearemos una función para actualizar la base de datos de Heroku. Primero, recordemos que FinnHub nos entrega la información de la fecha en Unix, por lo cual necesitaremos de una función para transformar la fecha a un formato legible:

# In[8]:


#Función utilizada para convertir t a un formato legible
def date_format(date_col):
    return datetime.fromtimestamp(date_col).strftime('%Y-%m-%d')


# In[8]:


def update_stock(stock, finnhub_client, sql_connection, engine):
    '''Función que actualizará automáticamente la tabla seleccionada con la información del último día disponible.
    stock: Tabla a actualizar. Solo puede tomar cinco valores ['EC','AVAL', 'CIB', 'TGLS','AVHOQ']
    finnhub_client: Cliente de FinnHub que extraerá la información.
    sql_connection: Conexión a la base de datos para leer las tablas.
    Engine: Conexión a la base de datos usada para actualizar las tablas.
    '''
    #Primero nos conectaremos a la base de datos, y obtendremos la última fecha 
    cursor = sql_connection.cursor()

    #Obteniendo la fecha máxima de la tabla seleccionada
    cursor.execute('SELECT MAX("Date") FROM "'+stock+'";')
    max_date = cursor.fetchall()
    max_date = max_date[0][0]
    print(max_date)

    #Tiempo a extraer
    end_date = datetime.now()
    start_date = datetime.strptime(str(max_date), '%Y-%m-%d')+ timedelta(days=1)
    print(start_date)
    print(end_date)

    #Transformando a Unix
    end = int(end_date.timestamp())
    start = int(start_date.timestamp())

    #Buscando la información en FinnHub
    res = finnhub_client.stock_candles(stock, 'D', start, end)
    df_res = pd.DataFrame(res).drop(columns=["s"], errors="ignore")

    if df_res.shape[0] == 0:
        return print("Tabla al día, no se actualizó.")
    else:
        #Cambiando el formato del tiempo
        df_res['t'] = df_res['t'].apply(date_format)

        #Cambiando los nombres de las columnas
        df_res.rename(columns = {'c':'Close','h':'High','l':'Low','o':'Open','t':'Date','v':'Volume'}, inplace = True)

        #Reorganizando el orden de las columnas
        cols = df_res.columns.to_list()
        cols = cols[4:5] + cols[3:4] + cols[1:3] + cols[0:1] + cols[-1:]
        df_res = df_res[cols]

        #Actualizando la tabla
        df_res.to_sql(stock, engine, if_exists = 'append', index=False, method='multi')
        return print("Tabla "+stock+ " actualizada.")


# Probemos la función, para esto necesitamos instanciar la conexión a Heroku y la conexión a FinnHub:

# In[5]:


#Conectandose a la base de datos
connection = psycopg2.connect(user="snfifneqljoybc",
                                  password="e1aaa689afeaaff2f75921f4fd6ca3816f6ebf33bdaf0c3366b9db4e5fe218fe",
                                  host="ec2-3-95-130-249.compute-1.amazonaws.com",
                                  port="5432",
                                  database="d3go0j5dkhm396")

#Configurando el cliente de FinnHub
finnhub_client = finnhub.Client(api_key="c5tpk32ad3i9n9aj0u3g")


# In[10]:


#Llamando a la función creada
stocks = ["EC", "AVAL", "AVHOQ", "TGLS", "CIB"]

for st in stocks:
    update_stock(st,finnhub_client,connection, engine)


# Con las tablas actualizadas en la base de datos de Heroku-Postgres, las leeremos nuevamente para proseguir con el EDA.

# In[6]:


ec_df = pd.read_sql('SELECT * FROM "EC"', connection)
aval_df = pd.read_sql('SELECT * FROM "AVAL"', connection)
bc_df = pd.read_sql('SELECT * FROM "CIB"', connection)
tgls_df = pd.read_sql('SELECT * FROM "TGLS"', connection)
av_df = pd.read_sql('SELECT * FROM "AVHOQ"', connection)


# La información descargada previamente de Yahoo Finance, y actualizada con FinnHub contiene el comportamiento diario de cada acción durante los últimos cinco años. Contiene la información del precio de apertura, precio máximo diario, precio mínimo diario, precio de cierre, precio ajustado de cierre y el volumen diario.

# In[6]:


ec_df.head()


# ## Análisis Exploratorio de Datos
# 
# Lo primero, es realizar un análisis básico, para esto usaremos la función describe de pandas, para generar las medidas de tendencia central más importantes para cada dataframe.

# In[31]:


ec_df.describe()


# In[32]:


aval_df.describe()


# In[33]:


bc_df.describe()


# In[34]:


tgls_df.describe()


# In[35]:


av_df.describe()


# In[36]:


[av_df.Date.min(), av_df.Date.max()]


# In[37]:


[aval_df.Date.min(), aval_df.Date.max()]


# In[38]:


[bc_df.Date.min(), bc_df.Date.max()]


# In[39]:


[ec_df.Date.min(), ec_df.Date.max()]


# In[40]:


[tgls_df.Date.min(), tgls_df.Date.max()]


# Observando las medidas de tendencia central para el valor de apertura de las cinco acciones durante los últimos cinco años (desde 2016-11-23 hasta 2021-12-09), podemos observar lo siguiente:
# 
# * Para Ecopetrol, el precio medio de apertura fue de US\$ 14.87, con una desviación de US\$ 4.62. Esto implica que el coeficiente de variación es de 0.311. El valor mínimo de apertura en los cinco años analizados fue de US\$ 6.72, y el valor máximo de apertura fue de US\$27.65.
# 
# * Para el Grupo Aval, el precio medio de apertura fue de US\$7.15, con una desviación de US\$1.44. Esto implica que el coeficiente de variación es de 0.201. El valor mínimo de apertura en los cinco años analizados fue de US\$3.35, y el valor máximo de apertura fue de US\$9.46.
# 
# * Para el Grupo Bancolombia, el precio medio de apertura fue de US\$40.06, con una desviación de US\$8.52. Esto implica que el coeficiente de variación es de 0.2123. El valor mínimo de apertura en los cinco años analizados fue de US\$17.07, y el valor máximo de apertura fue de US\$55.95.
# 
# * Para Tecnoglass, el precio medio de apertura fue de US\$9.78, con una desviación de US\$5.47. Esto implica que el coeficiente de variación es de 0.559. El valor mínimo de apertura en los cinco años analizados fue de US\$2.44, y el valor máximo de apertura fue de US\$34.32.
# 
# * Por último, para Avianca, el precio medio de apertura fue de US\$4.17, con una desviación de US\$3.084. Esto implica que el coeficiente de variación es de 0.739. El valor mínimo de apertura en los cinco años analizados fue de US\$0.04, y el valor máximo de apertura fue de US\$10.27.
# 
# Podemos observar que los coeficientes de variación de todas las acciones son menores que 1, así que se podrían considerar que el precio de apertura no tiene una alta varianza. Sin embargo, hay que tener cuidado con este resultado, ya que estamos agregando cinco años. Pudieron haber meses donde el coeficiente de variación perfectamente pudo haber superado el valor debido a grandes fluctuaciones en el precio de apertura de la acción.
# 
# Por esta razón, para revisar series de tiempo, lo mejor es revisar su comportamiento a lo largo del tiempo. Para esto, utilizaremos la función `lineplot` de `seaborn` para graficar el comportamiento de cada acción. En este caso, usaremos mejor el precio de cierre diario.
# 
# Antes de graficar, convertiremos la columna de *Date* a tipo `datetime`, para poder realizar operaciones más fácilmente.

# In[7]:


#Convirtiendo las fechas a tipo DateTime
ec_df.Date = ec_df.Date.apply(lambda x: datetime.strptime(str(x).strip(), '%Y-%m-%d')) 
bc_df.Date = bc_df.Date.apply(lambda x: datetime.strptime(str(x).strip(), '%Y-%m-%d'))
av_df.Date = av_df.Date.apply(lambda x: datetime.strptime(str(x).strip(), '%Y-%m-%d'))
aval_df.Date = aval_df.Date.apply(lambda x: datetime.strptime(str(x).strip(), '%Y-%m-%d'))
tgls_df.Date = tgls_df.Date.apply(lambda x: datetime.strptime(str(x).strip(), '%Y-%m-%d'))


# In[42]:


#Graficando el precio de cierre diario de cada acción
plt.figure(figsize = (10,6))
sns.lineplot(x=ec_df.Date, y=ec_df.Close)
sns.lineplot(aval_df.Date,aval_df.Close)
sns.lineplot(av_df.Date,av_df.Close)
sns.lineplot(bc_df.Date,bc_df.Close)
sns.lineplot(tgls_df.Date,tgls_df.Close)

plt.legend(labels=["Ecopetrol","Grupo Aval","Avianca", "Bancolombia", "Tecnoglass"])
plt.tight_layout()
plt.xticks(rotation=50)
plt.show()


# Como se observa en la gráfica, las diferentes acciones han tenido diferentes comportamientos en estos últimos cinco años. Como comportamiento común entre todas, se observa una caída en las acciones aproximadamente en marzo de 2020. Esto tiene sentido, ya que concuerda con el comienzo de la pandemia del Coronavirus, la cual tuvo un alto impacto en los precios de las acciones. Otras observaciones son las siguientes:
# 
# * Para la acción del Grupo Bancolombia, que fue la que tuvo el mayor precio promedio a lo largo de los cinco años, se observa que tuvo un crecimiento sostenido desde finales de 2016 hasta inicios de 2020, luego de la caída por la pandemia, empezó a recuperarse hasta alcanzar una precio alrededor de US\$40, para luego empezar una tendencia a la baja.
# 
# * Para la acción de Ecopetrol, también se observa un comportamiento similar, si bien tuvo un crecimiento desde finales de 2016 hasta 2018, luego experimentó un proceso de estabilización en el precio de US\$20, hasta la pandemia, donde cayó y empezó una lenta recuperación.
# 
# * Para la acción de Tecnoglass, se observa una tendencia a la baja desde finales de 2016 hasta 2020, donde tiene una fuerte caida por la pandemia. Luego de la pandemia, sufre una recuperación en el precio para en el 2021, dar un salto vertiginoso en el precio, al llegar a su máximo histórico en estos últimos cinco años.
# 
# * Para la acción de Grupo Aval, también se observa un comportamiento parecido a la acción de Ecopetrol, pero la estabilización ocurrió entre los US\$8 y US\$9 dólares. Luego de la pandemia, cayó y empezó una leve recuperación.
# 
# * Por último, para la acción de Avianca se observa una tendencia bajista desde finales de 2016 hasta 2020, donde la pandemia le acestó un tremendo golpe, ya que al ser aerolínea, estuvo fuertemente impactada por la situación. Luego de la pandemia la acción prácticamente vale cero, y no muestra signos de recuperación. 

# Además de poder revisar el comportamiento a través de los gráficos de lineas, también podemos usar otros métodos para observar las tendencias de las acciones. Empezemos con un gráfico de velas. El gráfico de velas nos permite ver en un mismo gráfico los precios de apertura y cierre de la acción. Para este caso, solo lo haremos con la acción de Grupo Bancolombia, a modo de ejemplo:

# In[45]:


fig = go.Figure(data=[go.Candlestick(x = bc_df.Date,
                                     open = bc_df.Open, 
                                     high = bc_df.High,
                                     low = bc_df.Low, 
                                     close = bc_df.Close)
                     ])
fig.update_layout(
    title="Bancolombia Stock (CIB)",
    xaxis_title="Day",
    yaxis_title="USD",
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="RebeccaPurple"
    )
)
fig.update_layout(xaxis_rangeslider_visible=False)
fig.show()


# También podemos ver la tendencia de los precios, con un gráfico de caja y bigotes:

# In[47]:


fig = go.Figure()

fig.add_trace(go.Box(
    y=bc_df["Close"],
    name='Bancolombia',
    marker_color='#3D9970'
))
fig.add_trace(go.Box(
    y=ec_df["Close"],
    name='Ecopetrol',
    marker_color='#FF4136'
))
fig.add_trace(go.Box(
    y=aval_df["Close"],
    name='Aval',
    marker_color='#FF851B'
))
fig.add_trace(go.Box(
    y=av_df["Close"],
    name='Avianca',
    marker_color='#00AAFF'
))
fig.add_trace(go.Box(
    y=tgls_df["Close"],
    name='Tecnoglass',
    marker_color='#572364'
))

fig.update_layout(
    title="Diagrama de Caja y Bigotes para el precio de Cierre",
    yaxis_title='USD',
    boxmode='group' # group together boxes of the different traces for each value of x
)
fig.show()


# Se puede observar la diferencia entre los precios y la distribución de cada acción. Bancolombia es la que tiene históricamente precios de cierre más altos, entre \$ 30 a \$ 45 USD, seguida de Ecopetrol (entre \$10 y \$18 USD). Se observa que el grupo Aval siempre ha mantenido unos precios bastante estables, de \$5 a \$10 USD. Avianca, por su lado, muestra los precios de cierre más bajos, entre \$0.5 y \$7 USD. Por último, si bien Tecnoglass tiene precios históricamente parecidos al grupo Aval, se observan un montón de datos atípicos. Estos datos atípicos coinciden con el excelente desempeño obtenido por la acción de Tecnoglass en 2021, dónde creció agresivamente.
# 
# Otra forma de revisar la distribución de los precios de cierre, es realizando un histograma. Miremos la distribución para cada acción:

# In[52]:


fig = make_subplots(rows=2, cols=3, shared_yaxes=True,       subplot_titles=["Ecopetrol", "Bancolombia", "Grupo Aval", "Avianca", "Tecnoglass"])

fig.add_trace(go.Histogram(x=ec_df["Close"], name='Ecopetrol', marker_color='#FF4136'),row=1, col=1)
fig.add_trace(go.Histogram(x=bc_df["Close"], name='Bancolombia', marker_color='#3D9970'),row=1, col=2)
fig.add_trace(go.Histogram(x=aval_df["Close"], name='Grupo Aval', marker_color='#FF851B'),row=1, col=3)
fig.add_trace(go.Histogram(x=av_df["Close"], name='Avianca', marker_color='#00AAFF'),row=2, col=1)
fig.add_trace(go.Histogram(x=tgls_df["Close"], name='Tecnoglass', marker_color='#572364'),row=2, col=2)

fig.update_layout(showlegend=False)
fig.show()


# Se pueden observar las distintas distribuciones que tiene el precio de cierre para cada acción. Tenemos acciones donde la distribución es más uniforme, como Bancolombia, y otras donde se observan máximos locales en la distribución, como es el caso de Avianca, Tecnoglass y Ecopetrol.
# 
# Observemos ahora como se comporta el volumen de transacciones con respecto al precio de cierre para cada acción:

# In[63]:


# Create figure with secondary y-axis
fig = make_subplots(rows=5, cols=1,     specs=[
            [{"secondary_y": True}], 
            [{"secondary_y": True}], 
            [{"secondary_y": True}], 
            [{"secondary_y": True}],
            [{"secondary_y": True}]
          ],
    subplot_titles=["Ecopetrol", "Bancolombia", "Grupo Aval", "Avianca", "Tecnoglass", None])

############################
########## Add ECOPETROL
fig.add_trace(
    go.Scatter(x=ec_df["Date"], y=ec_df["Close"], name="Precio Cierre"),
    secondary_y=False, row=1, col=1
)

fig.add_trace(
    go.Scatter(x=ec_df["Date"], y=ec_df["Volume"], name="Volumen"),
    secondary_y=True, row=1, col=1
)

# Set x-axis title
fig.update_xaxes(title_text="Fecha")

# Set y-axes titles
fig.update_yaxes(title_text="USD", secondary_y=False)
fig.update_yaxes(title_text="Número de Transacciones", secondary_y=True)


############################
########## Add Bancolombia
fig.add_trace(
    go.Scatter(x=bc_df["Date"], y=bc_df["Close"], name="Precio Cierre"),
    secondary_y=False, row=2, col=1
)

fig.add_trace(
    go.Scatter(x=bc_df["Date"], y=bc_df["Volume"], name="Volumen"),
    secondary_y=True, row=2, col=1
)

# Set x-axis title
fig.update_xaxes(title_text="Fecha")

# Set y-axes titles
fig.update_yaxes(title_text="USD", secondary_y=False)
fig.update_yaxes(title_text="Número de Transacciones", secondary_y=True)


############################
########## Add Grupo Aval
fig.add_trace(
    go.Scatter(x=aval_df["Date"], y=aval_df["Close"], name="Precio Cierre"),
    secondary_y=False, row=3, col=1
)

fig.add_trace(
    go.Scatter(x=aval_df["Date"], y=aval_df["Volume"], name="Volumen"),
    secondary_y=True, row=3, col=1
)

# Set x-axis title
fig.update_xaxes(title_text="Fecha")

# Set y-axes titles
fig.update_yaxes(title_text="USD", secondary_y=False)
fig.update_yaxes(title_text="Número de Transacciones", secondary_y=True)


############################
########## Add Avianca
fig.add_trace(
    go.Scatter(x=av_df["Date"], y=av_df["Close"], name="Precio Cierre"),
    secondary_y=False, row=4, col=1
)

fig.add_trace(
    go.Scatter(x=av_df["Date"], y=av_df["Volume"], name="Volumen"),
    secondary_y=True, row=4, col=1
)

# Set x-axis title
fig.update_xaxes(title_text="Fecha")

# Set y-axes titles
fig.update_yaxes(title_text="USD", secondary_y=False)
fig.update_yaxes(title_text="Número de Transacciones", secondary_y=True)

############################
########## Add Tecnoglass
fig.add_trace(
    go.Scatter(x=tgls_df["Date"], y=tgls_df["Close"], name="Precio Cierre"),
    secondary_y=False, row=5, col=1
)

fig.add_trace(
    go.Scatter(x=tgls_df["Date"], y=tgls_df["Volume"], name="Volumen"),
    secondary_y=True, row=5, col=1
)

# Set x-axis title
fig.update_xaxes(title_text="Fecha")

# Set y-axes titles
fig.update_yaxes(title_text="USD", secondary_y=False)
fig.update_yaxes(title_text="Número de Transacciones", secondary_y=True)


# Add figure title
fig.update_layout(
    title_text="Gráfica lineal comparativa entre volumen de transacciones y precio de cierre",
    width=1000, height=1400
)

fig.show()


# Como se puede observar, movimientos fuertes en el precio de cierre, ya sean alcistas o bajistas, siempre están correlacionados con un aumento de las transacciones diarias. Esto se puede observar en casi todas las acciones, exceptuando la del Grupo Aval que por alguna razón, el volumen es invariante a los cambios drásticos en el precio.
# 
# Por último, analizemos las relaciones que pueden haber entre los precios de diferentes acciones. En este caso, analizaremos las acciones de los dos grupos bancarios que tenemos:

# In[65]:


fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=bc_df["Close"], y=aval_df["Close"],
                    mode='markers'))

# Add figure title
fig.update_layout(
    title_text="Gráfica de dispersión Precio de Cierre Grupo Aval vs Bancolombia"
)

# Set x-axis title
fig.update_xaxes(title_text="Bancolombia (USD)")

# Set y-axes titles
fig.update_yaxes(title_text="Grupo Aval (USD)")

fig.show()


# Se observa que hay una correlación positiva, ya que a medida que el precio de cierre de la acción de grupo Aval aumenta, así lo hace el precio de cierre de la acción del grupo Bancolombia. Esto tiene sentido, ya que las dos empresas hacen parte del sector financiero del país, y muchos factores que podrían afectar a una acción, también podrían afectar a la otra. 
# 
# Comparemos ahora las acciones de Ecopetrol y Tecnoglass:

# In[66]:


fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=ec_df["Close"], y=tgls_df["Close"],
                    mode='markers'))

# Add figure title
fig.update_layout(
    title_text="Gráfica de dispersión Precio de Cierre Tecnoglass vs Ecopetrol"
)

# Set x-axis title
fig.update_xaxes(title_text="Ecopetrol (USD)")

# Set y-axes titles
fig.update_yaxes(title_text="Tecnoglass (USD)")

fig.show()


# En el caso del precio de cierre de Tecnoglass vs Ecopetrol, se puede observar que no hay una tendencia clara en lo que respecta a su comportamiento. Si bien, ambas empresas dependen del precio de dólar, ya que son exportadoras, no exportan el mismo tipo de producto, por lo cual, los factores que podrían afectar el precio podrían ser diferentes.

# ## Test de Estacionalidad en las series temporales
# 
# Ahora, verifiquemos si las series de tiempo son estacionarias o no. Para esto, usaremos la prueba de `Dickey-Fuller`. Para hacer uso de esta prueba importamos la función `adfuller` de la clase `statsmodels.tsa.stattools`.

# In[67]:


#Prueba de Dickey-Fuller para Ecopetrol
result = adfuller(ec_df.Close)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


# In[68]:


#Prueba de Dickey-Fuller para Grupo aval
result = adfuller(aval_df.Close)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


# In[69]:


#Prueba de Dickey-Fuller para Avianca
result = adfuller(av_df.Close)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


# In[70]:


#Prueba de Dickey-Fuller para Bancolombia
result = adfuller(bc_df.Close)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


# In[71]:


#Prueba de Dickey-Fuller para Tecnoglass
result = adfuller(tgls_df.Close)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


# Si consideramos un nivel de significancia del 5%, tenemos entonces que para todos los precios de cierre de las acciones el P-valor es mayor que este nivel, por lo cual, la hipótesis nula no se rechaza. Esto significa que no hay evidencia suficiente para afirmar que las series temporales analizadas son estacionarias. 
# 
# Como las series de tiempo no son estacionarias, calculemos las figuras de autocorrelación, para verificar el orden de integración necesario para convertirlas a estacionarias.

# In[72]:


#Graficando la autocorrelación de la serie original de Ecopetrol y sus primeras y segundas derivadas
plt.rcParams.update({'figure.figsize': (15,15)})

fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(ec_df.Close); axes[0, 0].set_title('Original Series')
plot_acf(ec_df.Close, ax=axes[0, 1], lags = 1200);

axes[1, 0].plot(ec_df.Close.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(ec_df.Close.diff().dropna(), ax=axes[1, 1], lags = 1200);

axes[2, 0].plot(ec_df.Close.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(ec_df.Close.diff().diff().dropna(), ax=axes[2, 1], lags = 1200);


# In[73]:


#Graficando la autocorrelación de la serie original de Grupo Aval y sus primeras y segundas derivadas
plt.rcParams.update({'figure.figsize': (15,15)})

fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(aval_df.Close); axes[0, 0].set_title('Original Series')
plot_acf(aval_df.Close, ax=axes[0, 1], lags = 1200);

axes[1, 0].plot(aval_df.Close.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(aval_df.Close.diff().dropna(), ax=axes[1, 1], lags = 1200);

axes[2, 0].plot(aval_df.Close.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(aval_df.Close.diff().diff().dropna(), ax=axes[2, 1], lags = 1200);


# In[74]:


#Graficando la autocorrelación de la serie original de Bancolombia y sus primeras y segundas derivadas
plt.rcParams.update({'figure.figsize': (15,15)})

fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(bc_df.Close); axes[0, 0].set_title('Original Series')
plot_acf(bc_df.Close, ax=axes[0, 1], lags = 1200);

axes[1, 0].plot(bc_df.Close.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(bc_df.Close.diff().dropna(), ax=axes[1, 1], lags = 1200);

axes[2, 0].plot(bc_df.Close.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(bc_df.Close.diff().diff().dropna(), ax=axes[2, 1], lags = 1200);


# In[75]:


#Graficando la autocorrelación de la serie original de Avianca y sus primeras y segundas derivadas
plt.rcParams.update({'figure.figsize': (15,15)})

fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(av_df.Close); axes[0, 0].set_title('Original Series')
plot_acf(av_df.Close, ax=axes[0, 1], lags = 1200);

axes[1, 0].plot(av_df.Close.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(av_df.Close.diff().dropna(), ax=axes[1, 1], lags = 1200);

axes[2, 0].plot(av_df.Close.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(av_df.Close.diff().diff().dropna(), ax=axes[2, 1], lags = 1200);


# In[76]:


#Graficando la autocorrelación de la serie original de Tecnoglass y sus primeras y segundas derivadas
plt.rcParams.update({'figure.figsize': (15,15)})

fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(tgls_df.Close); axes[0, 0].set_title('Original Series')
plot_acf(tgls_df.Close, ax=axes[0, 1], lags = 1200);

axes[1, 0].plot(tgls_df.Close.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(tgls_df.Close.diff().dropna(), ax=axes[1, 1], lags = 1200);

axes[2, 0].plot(tgls_df.Close.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(tgls_df.Close.diff().diff().dropna(), ax=axes[2, 1], lags = 1200);


# Se puede observar el decaimiento geométrico en la primera figura de autocorrelación en todas las series de tiempo analizadas, que baja desde la parte positiva con una tendencia lineal. Esto confirma lo observado con la prueba de `Dickey-Fuller`, ya que nos indica que son series de tiempo no estacionarias.
# 
# Adicionalmente, se observa también un rápido decaimiento a la parte negativa en la gráfica de autocorrelación de la segunda derivada, lo que indicaría que hubo una sobrediferenciación. Esto indicaría que con la primera derivada es suficiente para transformar los datos para el modelo ARIMA, si quisieramos usar series estacionales.

# ## Entrenamiento y Test de Modelos
# 
# En este apartado, probaremos dos modelos. Un modelo básico, el modelo de regresión lineal, y el modelo dado en clase, el cual es más apropiado para estos casos, el modelo ARIMA.
# 
# Empezemos primero con el modelo de regresión lineal. Se entrenará un modelo para cada acción. Como estamos hablando de series temporales, mantener el orden es de suma importancia. Por esta razón, se decidirá hacer una partición de training/test donde el test sea de un mes (30 días). El resto de la información la usaremos como training.

# Con los valores definidos de train y test, procedemos a entrenar el modelo de regresión, usando la función de statsmodels.api.OLS. Para este caso, crearemos una función que entrene el modelo de regresión y nos entregue el modelo entrenado, junto con las métricas asociadas.

# In[9]:


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


# Ahora entrenamos los modelos de regresión lineal:

# In[10]:


#Horizonte utilizado
horizon = 30

#Dataframes de las acciones
df_all = [ec_df, aval_df, bc_df, tgls_df, av_df]

#Lista que guardará los modelos entrenados
linear_models = []

#Llamando la función para cada dataframe
for df in df_all:
    best_m = OLS_temp(df, horizon=horizon)
    linear_models.append(best_m)


# Observemos la salida de uno de los modelos, por ejemplo, para la acción de Ecopetrol:

# In[11]:


linear_models[0][4].summary()


# Se puede observar que para la regresión lineal, el coeficiente relacionado con la fecha no es significativo (Tomando un nivel de significancia del 5%). Particularmente este tipo de modelo no es muy bueno para predecir series temporales, ya que usualmente las series temporales no tienen comportamientos o tendencia lineales.

# Con los modelos lineales entrenados, revisaremos el error de predicción que arroja cada modelo. Para esto, usaremos la función mostrada en clase por el profesor, para calcular el error de predicción cometido en las predicciones, usando las métrica `MAPE, MAE, RMSE, MSE`:

# In[12]:


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


# Verifiquemos los errores de predicción para cada acción. Para esto, realizaremos las predicciones de los últimos 30 días:

# In[13]:


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


# In[14]:


#Nombres
stocks = ["EC", "AVAL","CIB", "TGLS", "AVHOQ"]

#Variable para guardar las predicciones de cada modelo
predictions_list_linear = []

#for loop para recorrer los modelos y generar las predicciones
for var in linear_models:
    pred = linear_rolling(*var)
    predictions_list_linear.append(pred)

#Variable para guardar las métricas
df_metricas_l = pd.DataFrame()

#Usamos la función forecast_accuracy para calcular las métricas de cada modelo
for i, pred in enumerate(predictions_list_linear):
    df_temp = forecast_accuracy(np.array(pred),np.array(linear_models[i][3]), "Model for {} stock".format(stocks[i]))
    df_metricas_l = pd.concat([df_metricas_l, df_temp], axis=0)

df_metricas_l


# Como se puede observar, aunque se obtienen errores de predicción bajos según las métricas, esto no necesariamente indica que el modelo tiene un buen desempeño. Grafiquemos las predicciones obtenidas para cada acción con el modelo de regresión lineal:

# In[15]:


fig, ax=plt.subplots(3,2, figsize=(15,15))

####### Ecopetrol
x = linear_models[0][2]
sns.lineplot(x, linear_models[0][3], label="Test",ax=ax[0,0]) 
sns.lineplot(x, predictions_list_linear[0], label="Forecast",ax=ax[0,0])
ax[0,0].set_xlabel("Fecha")
ax[0,0].set_ylabel("Precio de Cierre")
ax[0,0].set_title("Forecast Ecopetrol - 30 días")
ax[0,0].legend(loc="best")

####### Modelo Grupo Aval
x = linear_models[1][2]
sns.lineplot(x, linear_models[1][3], label="Test",ax=ax[0,1]) 
sns.lineplot(x, predictions_list_linear[1],  label="Forecast",ax=ax[0,1]) 
ax[0,1].set_xlabel("Fecha")
ax[0,1].set_ylabel("Precio de Cierre")
ax[0,1].set_title("Forecast Grupo Aval - 30 días")
ax[0,1].tick_params(axis='x', rotation=60)
ax[0,1].legend(loc="best")

####### Modelo Bancolombia
x = linear_models[2][2]
sns.lineplot(x, linear_models[2][1],label="Test",ax=ax[1,0]) 
sns.lineplot(x, predictions_list_linear[2],  label="Forecast",ax=ax[1,0]) 
ax[1,0].set_xlabel("Fecha")
ax[1,0].set_ylabel("Precio de Cierre")
ax[1,0].set_title("Forecast Bancolombia - 30 días")
ax[1,0].tick_params(axis='x', rotation=60)
ax[1,0].legend(loc="best")

######## Modelo Tecnoglass
x = linear_models[3][2]
sns.lineplot(x, linear_models[3][1],  label="Test",ax=ax[1,1]) 
sns.lineplot(x, predictions_list_linear[3], label="Forecast",ax=ax[1,1]) 
ax[1,1].set_xlabel("Fecha")
ax[1,1].set_ylabel("Precio de Cierre")
ax[1,1].set_title("Forecast Tecnoglass - 30 días")
ax[1,1].tick_params(axis='x', rotation=60)
ax[1,1].legend(loc="best")

######## Modelo Avianca
x = linear_models[4][2]
sns.lineplot(x, linear_models[4][1], label="Test",ax=ax[2,0]) 
sns.lineplot(x, predictions_list_linear[4],  label="Forecast",ax=ax[2,0]) 
ax[2,0].set_xlabel("Fecha")
ax[2,0].set_ylabel("Precio de Cierre")
ax[2,0].set_title("Forecast Avianca - 30 días")
ax[2,0].tick_params(axis='x', rotation=60)
ax[2,0].legend(loc="best")

plt.tight_layout()
plt.show()


# Como se puede observar, la regresión lineal  en algunos casos está cerca de los valores reales los primeros días, pero cuando la tendencia cambia y el precio de cierre baja, la regresión no es tan sensible a estos cambios. En otros casos, la regresión lineal está muy por debajo o muy por encima de los valores reales.
# 
# Ahora, proseguiremos con los modelos ARIMA. Para esto, usaremos la función que se había creado para el taller #2, con una pequeña modificación. En este caso, aumentaremos el rango de los órdenes, para encontrar el mejor modelo para cada acción. También usaremos el criterio de Akaike, el cual es asintóticamente eficiente (menor error de predicción en nuevos datos).

# In[16]:


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
                    #exceptions.append(e)
                    continue

    #Regresando los valores que se encontraron, junto con el train y el test set
    return(train, test, dates_test, best_mdl, best_order, best_metric)#, exceptions)


# Corremos el modelo ARIMA para cada acción:

# In[17]:


#Test seleccionado
horizon = 30

#Comentando para poder cargar el notebook
#Variable para guardar los resultados de la función
#best_models = []
#
#for df in df_all:
#    best_m = ARIMA_model(df, horizon=horizon, pq_rng=range(10), d_rng=range(10))
#    best_models.append(best_m)


# Revisemos los ordenes de cada modelo:

# In[32]:


#Cargando desde el pickle 
with open("./data/arima_models.pickle", "rb") as f:
    best_models = pickle.load(f)

for model in best_models:
    print('aic: {:6.5f} | order: {}'.format(model[5], model[4]))


# Con los mejores modelos según el criterio de Akaike, podemos usarlos para predecir los valores futuros usando rolling. Para esto, usaremos los modelos guardados en la variable `best_models`. Adicionalmente, usaremos también la función utilizada en clase, con unos pequeños cambios (Ya calculamos el modelo y lo entrenamos, así que solo haremos las predicciones).

# In[19]:


def arima_rolling(history, test, best_order):
    
    predictions = list()
    for t in range(len(test)):
        model_fit = ARIMA(history, order=best_order, enforce_stationarity=False).fit(method='innovations_mle')
        output = model_fit.forecast()
        #print(output)
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        #print('predicted=%f, expected=%f' % (yhat, obs))
        
    return predictions


# In[20]:


#Nombres
stocks = ["EC", "AVAL","CIB", "TGLS", "AVHOQ"]

#Variable para guardar las predicciones de cada modelo
predictions_list = []

#for loop para recorrer los modelos y generar las predicciones
for var in best_models:
    pred = arima_rolling(var[0].to_list(), var[1].to_list(), var[4])
    predictions_list.append(pred)

#Variable para guardar las métricas
df_metricas = pd.DataFrame()

#Usamos la función forecast_accuracy para calcular las métricas de cada modelo
for i, pred in enumerate(predictions_list):
    df_temp = forecast_accuracy(np.array(pred),np.array(best_models[i][1]), "Model for {} stock".format(stocks[i]))
    df_metricas = pd.concat([df_metricas, df_temp], axis=0)

df_metricas


# In[22]:


#Generando las predicciones para el dashboard (Evitar error H12)
#dashboard_arima = {}
#
#horizons = [7, 14, 21, 28]
#stocks_name = ["EC", "AVAL", "CIB", "TGLS", "AVHOQ"]
#prop = ["x","y_true", "y_pred"]
#
##Diccionario para los modelos ARIMA (calculados acá)
#best_order = dict(zip(
#    ["EC", "AVAL", "CIB", "TGLS", "AVHOQ"],
#    [(6,1,9),(5,0,4),(0,1,7),(9,2,9),(7,1,7)]
#))
#
##Usamos df_all generado arriba, tiene el mismo orden de stocks_name
#
#for i, stock in enumerate(stocks_name):
#    st = dict()
#    df = df_all[i]
#    for horizon in horizons:
#        #Calculando train y test size
#        n = len(df.Close); n_test = horizon # This can be changed
#        train_size = n - n_test
#        
#        #Creando cada dataframe para train y test
#        train = df.Close.iloc[:train_size]
#        y_true = df.Close.iloc[train_size:train_size + n_test] 
#        x = df.Date.iloc[train_size:train_size + n_test] 
#
#        #Calculando las predicciones
#        y_pred = arima_rolling(train.to_list(), y_true.to_list(), best_order[stock])
#
#        #Guardando los resultados en el diccionario
#        st[horizon] = {"x": x, "y_true": y_true, "y_pred": y_pred}
#    
#    #Guardando los resultados de la acción en el diccionario general
#    dashboard_arima[stock] = st


# In[27]:


#Exporting data to pickle
#with open('./data/pred_arima.pickle', 'wb') as f:
#    pickle.dump(dashboard_arima, f)


# Como se puede observar, tenemos errores bastantes bajos para los modelos. Grafiquemos los test y las predicciones de cada acción:

# In[33]:


fig, ax=plt.subplots(3,2, figsize=(15,15))

sns.lineplot(best_models[0][2], best_models[0][1], label="Test",ax=ax[0,0]) 
sns.lineplot(best_models[0][2], predictions_list[0], label="Forecast",ax=ax[0,0]) #Modelo Ecopetrol
ax[0,0].set_xlabel("Fecha")
ax[0,0].set_ylabel("Precio de Cierre")
ax[0,0].set_title("Forecast Ecopetrol - 30 días")

sns.lineplot(best_models[1][2], best_models[1][1], label="Test",ax=ax[0,1]) 
sns.lineplot(best_models[1][2], predictions_list[1], label="Forecast",ax=ax[0,1]) #Modelo Grupo Aval
ax[0,1].set_xlabel("Fecha")
ax[0,1].set_ylabel("Precio de Cierre")
ax[0,1].set_title("Forecast Grupo Aval - 30 días")
ax[0,1].tick_params(axis='x', rotation=60)

sns.lineplot(best_models[2][2], best_models[2][1], label="Test",ax=ax[1,0]) 
sns.lineplot(best_models[2][2], predictions_list[2], label="Forecast",ax=ax[1,0]) #Modelo Bancolombia
ax[1,0].set_xlabel("Fecha")
ax[1,0].set_ylabel("Precio de Cierre")
ax[1,0].set_title("Forecast Bancolombia - 30 días")
ax[1,0].tick_params(axis='x', rotation=60)

sns.lineplot(best_models[3][2], best_models[3][1], label="Test",ax=ax[1,1]) 
sns.lineplot(best_models[3][2], predictions_list[3], label="Forecast",ax=ax[1,1]) #Modelo Tecnoglass
ax[1,1].set_xlabel("Fecha")
ax[1,1].set_ylabel("Precio de Cierre")
ax[1,1].set_title("Forecast Tecnoglass - 30 días")
ax[1,1].tick_params(axis='x', rotation=60)

sns.lineplot(best_models[4][2], best_models[4][1], label="Test",ax=ax[2,0]) 
sns.lineplot(best_models[4][2], predictions_list[4], label="Forecast",ax=ax[2,0]) #Modelo Avianca
ax[2,0].set_xlabel("Fecha")
ax[2,0].set_ylabel("Precio de Cierre")
ax[2,0].set_title("Forecast Avianca - 30 días")
ax[2,0].tick_params(axis='x', rotation=60)

plt.tight_layout()
plt.show()


# Se observa que el modelo ARIMA genera predicciones para las acciones muy parecidas a la realidad. Por último, se exportarán los modelos a pkl para cargarlos en el Dashboard:

# In[29]:


#Exporting models
#with open('linear_models.pickle', 'wb') as f:
#    pickle.dump(linear_models, f)
#
#with open('arima_models.pickle', 'wb') as f:
#    pickle.dump(best_models, f)


# ## Dashboard
# 
# Como primer acercamiento, se habían presentado los siguientes bosquejos para los dashboards:

# ![Bosquejo Dashboard para el EDA.](images/Bosquejo_Dashboard_EDA_Python.png)
# 

# ![Bosquejo Predicción](./images/Bosquejo_Dashboard_Pred_Python.png)

# Los dashboards fueron construidos usando el módulo de `Dash`, y el resultado final fue el siguiente:
# 
# ![](./images/Dashboard_Final_EDA.png)

# Como se observa, en el dashboard del EDA, se puede observar que se tienen seis containers. En la primera fila, la primera columna es el centro de control del dashboard, y nos permite escoger las siguientes opciones:
# 
# * **Acción seleccionada:** En este dropdown podemos seleccionar una de las cinco acciones colombianas (Ecopetrol, Grupo Aval, Bancolombia, Tecnoglass o Avianca). Esta acción será la principal, y actualizará todos los gráficos del dashboard.
# * **Tiempo seleccionado:** Acá podremos seleccionar el tiempo que queremos analizar la acción, y sus posibles valores son 7, 30, 60, 170 días, un año, y todo el tiempo disponible en el DataFrame (cinco años).
# * **Fecha de la DB:** Este valor no es seleccionable, pero, cuando se escoge la acción, nos muestra el tiempo disponible en la base de datos.
# * **Actualizar DB:** Este botón nos sirve para actualizar las bases de datos con la información de FinnHub.
# 
# Una vez seleccionada la acción principal, los diferentes gráficos cambiarán dependiendo del tiempo seleccionado, para mostrarnos los resultados. También se calcularán los KPIs presentes en la segunda columna. Los KPIs son los siguientes:
# 
# * **Precio de cierre promedio:** Este KPI nos muestra el precio de cierre promedio para la acción durante el tiempo seleccionado.
# * **Precio de cierre mínimo:** Este KPI nos muestra el precio de cierre mínimo para la acción durante el tiempo seleccionado.
# * **Precio de cierre máximo:** Este KPI nos muestra el precio de cierre máximo para la acción durante el tiempo seleccionado.
# * **P-value Dickey-Fuller Test:** Este KPI nos muestra el P-valor del test de `Dickey-Fuller` para la acción durante el tiempo seleccionado.
# 
# Las gráficas que se actualizan automáticamente son las siguientes:
# 
# * **Diagrama de dispersión**: Este diagrama compara el precio de cierre de la acción principal seleccionada, con el precio de cierre de otra acción que escojamos. La acción secundaria se puede seleccionar en el dropdown.
# * **Histograma del precio de cierre**: El histograma nos muestra la distribución del precio de cierre de la acción principal durante el tiempo seleccionado.
# * **Diagrama de velas**: El diagrama de velas muestra la tendencia histórica del precio de la acción principal durante el tiempo seleccionado. Adicionalmente, este gráfico posee una opción para poder visualizar las últimas 24h con una resolución de minutos, utilizando la API de FinnHub.
# * **Gráfico de líneas para el volumen de transacciones y el precio de cierre:** Este diagrama nos muestra en el mismo plano, el volumen de transacciones y el precio de cierre de la acción principal durante el tiempo seleccionado.
# 
# Por último, debajo de las gráficas se observa un link que nos redirige hacia el dashboard predictivo. El dashboard predictivo tiene la siguiente estructura:

# ![](./images/Dashboard_Final_Predictivo.png)

# Nuestro dashboard predictivo, también posee un centro de control. El centro de control posee las siguientes opciones:
# * **Acción seleccionada:** En este dropdown podemos seleccionar una de las cinco acciones colombianas (Ecopetrol, Grupo Aval, Bancolombia, Tecnoglass o Avianca). Esta acción será la que actualizará todos las métricas y el gráfico del dashboard.
# * **Horizonte seleccionado:** Acá podremos seleccionar la ventana de tiempo que queremos predecir, y sus posibles valores son 7, 14, 21 y 28 días.
# * **Tipo de Modelo:** Este botón radial nos permite escoger entre generar un modelo lineal o un modelo ARIMA. Los modelos usarán los mejores parámetros encontrados en el tunning de los hiper-parámetros. 
# * **Parámetros del Modelo:** Este valor no es seleccionable, pero se actualiza cuando se selecciona el tipo de modelo y la acción. Nos muestra los parámetros del modelo que genera las predicciones.
# 
# En el apartado de los resultados, contamos con las cuatro métricas principales para medir el error de predicción, calculado para el horizonte y la acción seleccionada, y en la gráfica de líneas se comparan los valores reales de la acción con las predicciones realizadas por el modelo.
# 
# Por último, también se tiene un link para regresar al dashboard inicial.

# El dashboard fue desplegado usando Heroku:
# 
# ![](./images/Heroku-App.png)
# 
# Los links donde se pueden observar los dashboards son los siguientes:
# * EDA: https://dashboard-finn.herokuapp.com/dash/
# * Modelos: https://dashboard-finn.herokuapp.com/model/

# Si bien, con Heroku tenemos la opción de tener un servidor gratis para poder desplegar el dashboard, este servidor (en su versión gratuita) es limitado. De manera local, se puede correr y generar rápidamente las predicciones para el modelo ARIMA (approx. 1-2 minutos). 
# 
# Pero, cuando se intentan generar las predicciones en el dashboard desplegado en Heroku, no se alcanza a generar por la restricción que tiene Heroku (entregar los resultados de las peticiones en menos de 30 segundos). Se intentó resolver este problema, reduciendo el train size del modelo, para reducir la carga y así agilizar el proceso, pero seguía ocurriendo el mismo problema.
# 
# Una manera de solucionar este problema es aumentar la capacidad del servidor, pero esto ya es algo que queda fuera del scope del curso.
# 
# ![](./images/H12_Error_Timeout.png)

# Al final, se decidió realizar la predicción de los modelos offline, para que el tiempo de carga fuera más rápido.
