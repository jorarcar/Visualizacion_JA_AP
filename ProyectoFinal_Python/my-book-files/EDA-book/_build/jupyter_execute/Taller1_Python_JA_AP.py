#!/usr/bin/env python
# coding: utf-8

# # Taller 1. Visualización de Datos en Python y R.
# ## Jorge Arteaga y Adriana Palacio.
# ## Maestría en Estadística Aplicada.
# ### Universidad del Norte.

# ## Ejercicio 1.1
# 
# Trabajaremos con el conjunto de datos de 120 años de historia olímpica adquirido por Randi Griffin en [Randi-Griffin](https://www.sports-reference.com/) y puesto a disposición en [athlete_events](https://raw.githubusercontent.com/lihkirun/AppliedStatisticMS/main/DataVisualizationRPython/Lectures/Python/PythonDataSets/athlete_events.csv). La tarea consiste en identificar los cinco deportes más importantes según el mayor número de medallas otorgadas en el año 2016, y luego realizar el siguiente análisis:
# 
# 1. Genere un gráfico que indique el número de medallas concedidas en cada uno de los cinco principales deportes en 2016.
# 2. Trace un gráfico que represente la distribución de la edad de los ganadores de medallas en los cinco principales deportes en 2016.
# 3. Descubre qué equipos nacionales ganaron el mayor número de medallas en los cinco principales deportes en 2016.
# 4. Observe la tendencia del peso medio de los atletas masculinos y femeninos ganadores en los cinco principales deportes en 2016
# 
# - Pasos principales.
# 
# 1. Descargue el conjunto de datos y formatéelo como un DataFrame de pandas.
# 2. Filtra el **DataFrame** para incluir solo las filas correspondientes a los ganadores de medallas de 2016.
# 3. Descubre las medallas concedidas en 2016 en cada deporte.
# 4. Enumera los cinco deportes más importantes en función del mayor número de medallas concedidas. Filtra el **DataFrame** una vez más para incluir solo los registros de los cinco deportes principales en 2016.
# 5. Genere un gráfico de barras con los recuentos de registros correspondientes a cada uno de los cinco deportes principales.
# 6. Generar un histograma para la característica Edad de todos los ganadores de medallas en los cinco deportes principales (2016).
# 7. Genera un gráfico de barras que indique cuántas medallas ganó el equipo de cada país en los cinco deportes principales en 2016.
# 8. Genere un gráfico de barras que indique el peso medio de los jugadores, clasificados en función del género, que ganaron en los cinco principales deportes en 2016.

# Primero, cargaremos los modulos que se usarán en todo el taller:

# In[1]:


import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import plotly.express as px
from plotly.offline import plot as px_off_plot
import altair as alt
alt.renderers.enable('html')
from altair import pipe, limit_rows, to_values
t = lambda data: pipe(data, limit_rows(max_rows=10000), to_values)
alt.data_transformers.register('custom', t)
alt.data_transformers.enable('custom')


# Primero, leamos los archivos de la **URL**, y carguemoslos en un **DataFrame** de **pandas**:

# In[43]:


athlete_link = "https://raw.githubusercontent.com/lihkirun/AppliedStatisticMS/main/DataVisualizationRPython/Lectures/Python/PythonDataSets/athlete_events.csv"
athlete_df = pd.read_csv(athlete_link)
athlete_df.head()


# Filtremos ahora el DataFrame para solo mostrar a los ganadores de Medallas:

# In[44]:


#Primero miremos que valores tiene la columna Medal
athlete_df["Medal"].unique()


# In[45]:


#Filtrando por Ganadores de Medallas y año 2016
athlete_df = athlete_df[athlete_df.Medal.notna() & (athlete_df.Year == 2016)]
athlete_df.head()


# Ahora, revisemos cuales son los deportes con más medallas concedidas:

# In[46]:


#Agrupando por deporte, y luego contando el número de medallas concedidas
#Luego, se ordenan entre los deportes que más tienen medallas
important_sport = athlete_df.groupby(["Sport"])["Medal"].count().sort_values(ascending = False).head(5)
important_sport = pd.DataFrame(data=important_sport).reset_index()
important_sport


# Se puede observar que atletismo se lleva el primer lugar con 192 medallas concedidas, luego natación con 191, Remo con 144, Fútbol con 106 y Hockey con 99 medallas. Filtremos entonces el DataFrame únicamente con estos cinco deportes.

# In[47]:


#Filtrando por los deportes con más medallas
athlete_df = athlete_df[athlete_df.Sport.isin(["Athletics","Swimming", "Rowing", "Football", "Hockey"])]
athlete_df.head()


# Generemos ahora el gráfico que indica el número de medallas concedidas en cada uno de los cinco principales deportes en 2016:

# In[48]:


fig, ax1 = plt.subplots(figsize=(12,6))
plt.rcParams.update({'font.size': 10})

sns.barplot(data = important_sport, x="Sport", y="Medal")
ax1.set_xlabel('Deportes')
ax1.set_ylabel('Número de Medallas Concedidas')
ax1.tick_params(axis='y')
ax1.set_title("Número de medallas concedidas en los cinco deportes principales en 2016")
plt.show()


# Se observa la gráfica generada con la información de la tabla anteriormente mencionada. Ahora, miremos como se encuentra distribuida la variable edad entre los ganadores de Medallas.

# In[49]:


fig, ax=plt.subplots(1,2, figsize=(15,8))

sns.histplot(x="Age", data=athlete_df, ax=ax[0]) #Histograma de la edad con el dataFrame ya filtrado
ax[0].set_xlabel("Edad")
ax[0].set_ylabel("Frecuencia")
ax[0].set_title("Histograma para la Edad")

sns.boxplot(y="Age", data=athlete_df, ax=ax[1])  #Diagrama de cajas y bigotes para el diagrama ya filtrado
ax[1].set_ylabel("Edad")
ax[1].set_title("Diagrama de Cajas y Bigotes para la Edad")

plt.show()


# In[50]:


athlete_df.describe()


# Como se puede observar en el histograma y en el diagrama de cajas y bigotes, la mayoría de los ganadores de medallas en 2016 tienen entre 22 y 29 años (primer y tercer cuantil). Esto representa el 50% de la población. La media y la mediana tienen valores similares, y se ubican entre los 25-26 años. Esto se observa en el histograma, donde se puede ver una gráfica algo simétrica. El valor mínimo fue de 16 años, y el valor máximo de 40 años (El cual es un dato extremo). También se observa que el 75% de los ganadores de medallas en los cinco deportes principales, tenían menos de 29 años.

# Ahora, miremos cuántas medallas ganó el equipo de cada país en los cinco deportes principales de 2016:

# In[51]:


#Agrupando por Equipo, y luego contando el número de medallas concedidas
#Luego, se ordenan entre los equipos que más medallas obtuvieron
team_medal = athlete_df.groupby(["Team"])["Medal"].count().sort_values(ascending = False)
team_medal = pd.DataFrame(data=team_medal).reset_index()
team_medal.head(5)


# In[52]:


fig, ax1 = plt.subplots(figsize=(12,6))
plt.rcParams.update({'font.size': 9})

sns.barplot(data = team_medal, x="Team", y="Medal")
ax1.set_xlabel('Países competidores en 2016')
ax1.set_ylabel('Número de Medallas Ganadas')
ax1.set_title("Países ganadores de medallas en los cinco deportes principales en 2016")
plt.xticks(rotation=80)
ax1.tick_params(axis='y')
plt.show()


# Como se observa, el país con más medallas ganadas en 2016 en los cinco principales deportes es Estados Unidos con 127 medallas, seguido por Alemania con 88, Reino Unido con 69, Canadá con 45 y Australia con 43.

# In[53]:


team_medal[team_medal["Team"]=="Colombia"]


# In[54]:


athlete_df[athlete_df["Team"]== "Colombia"]


# Nuestro país sólo obtuvo una medalla en 2016 para los cinco deportes principales. La medalla fue de oro en atletismo, y la ganó Catherine Ibarguén en Salto Triple.

# Por último, revisemos la tendencia del peso medio de los atletas masculinos y femeninos ganadores en los cinco principales deportes en 2016.

# In[55]:


#Agrupando por Sexo, y luego calculando el promedio del peso medio 
weight_df = athlete_df.groupby(["Sex"])["Weight"].mean().sort_values(ascending = False)
weight_df = pd.DataFrame(data=weight_df).reset_index()
weight_df.head(5)


# In[56]:


fig, ax1 = plt.subplots(figsize=(12,6))
plt.rcParams.update({'font.size': 9})

sns.barplot(data = weight_df, x="Sex", y="Weight")
ax1.set_xlabel('Sexo')
ax1.set_ylabel('Peso Promedio')
ax1.set_title("Peso promedio por sexo para los atletas ganadores de medallas en los cinco deportes principales en 2016")
ax1.tick_params(axis='y')
plt.show()


# Se observa una diferencia entre los pesos promedios de ambos grupos, el peso promedio para los atletas masculinos ganadores de medallas fue de 81.80 kg, mientras que el peso promedio para las atletas femeninos ganadoras de medallas fue de 65.23kg.
# 
# Miremos si esta tendencia se conserva en los diferentes deportes mencionados anteriormente:

# In[57]:


#Agrupando por Deporte y Sexo, y luego calculando el promedio del peso medio 
weight_team_df = athlete_df.groupby(["Sport","Sex"])["Weight"].mean().sort_values(ascending = False)
weight_team_df = pd.DataFrame(data=weight_team_df).reset_index()
weight_team_df.head(10)


# In[58]:


fig, ax1 = plt.subplots(figsize=(12,6))
plt.rcParams.update({'font.size': 9})

sns.barplot(data = weight_team_df, x="Sport", y="Weight", hue="Sex")
ax1.set_xlabel('Deporte')
ax1.set_ylabel('Peso Promedio')
ax1.set_title("Peso promedio por deporte y sexo para los atletas ganadores de medallas en los cinco deportes principales en 2016")
ax1.tick_params(axis='y')
plt.show()


# Como se observa, la tendencia general (el peso promedio en general es mayor en los atletas masculinos ganadores de medallas que en las atletas femeninas) se mantiene en cada deporte. 

# ## Ejercicio 1.2
# 
# **Estadísticas:** Seguiremos trabajando con el conjunto de datos de 120 años de historia olímpica adquirido por Randi Griffin en [Randi Griffin](https://www.sports-reference.com/).
# 
# Como especialista en visualización, su tarea consiste en crear dos parcelas para los ganadores de medallas de 2016 de cinco deportes: atletismo, natación, remo, fútbol y hockey.
# 
# - Crea un gráfico utilizando una técnica de visualización adecuada que presente de la mejor manera posible el patrón global de las características de **height** y **weight** de los ganadores de medallas de 2016 de los cinco deportes.
# 
# - Crea un gráfico utilizando una técnica de visualización adecuada que presente de la mejor manera posible la estadística de resumen para la altura y el peso de los jugadores que ganaron cada tipo de medalla (oro/plata/bronce) en los datos.
# 
# **Pasos importantes**
# 
# - Descargue el conjunto de datos y formatéelo como un **pandas** DataFrame
# - Filtrar el DataFrame para incluir únicamente las filas correspondientes a los ganadores de medallas de 2016 en los deportes mencionados en la descripción de la actividad.
# - Observe las características del conjunto de datos y anote su tipo de datos: ¿son categóricos o numéricos?
# - Evaluar cuál sería la visualización adecuada para que un patrón global represente las características de **height** y **weight**
# - Evaluar cuál sería la visualización adecuada para representar las estadísticas resumidas de las características de **height** y **weight** en función de las medallas, separadas además por género de los atletas.

# Cargamos entonces los datos nuevamente:

# In[59]:


athlete_link = "https://raw.githubusercontent.com/lihkirun/AppliedStatisticMS/main/DataVisualizationRPython/Lectures/Python/PythonDataSets/athlete_events.csv"
athlete_df = pd.read_csv(athlete_link)
athlete_df.head()


# In[60]:


#Filtrando por Ganadores de Medallas y año 2016
best_df = athlete_df[athlete_df.Medal.notna() & (athlete_df.Year == 2016)]

#Filtrando por los deportes con más medallas
best_df = best_df[best_df.Sport.isin(["Athletics","Swimming", "Rowing", "Football", "Hockey"])]
best_df.head()


# In[61]:


best_df.describe()


# In[62]:


best_df.dtypes


# Se tiene entonces que hay diferentes tipos de datos, tenemos datos numéricos como la edad, altura, peso, y el año. Tenemos datos categóricos como el nombre, sexo, y equipo, entre otros. 
# 
# Para poder representar el patrón global de las características de altura y peso de los ganadores de medallas de 2016 de los cinco deportes, utilizaremos un diagrama de dispersión hexagonal (hexbin):

# In[63]:


plt.figure(figsize=(10, 10))
plt.rcParams.update({'font.size': 9})

sns.set(style="ticks")
sns.jointplot(x=best_df.Weight, y=best_df.Height, kind="hex", color="#4CB391")

plt.xlabel('Peso')
plt.ylabel('Altura')
plt.show()


# Se puede observar que hay una correlación positiva entre la altura y el peso, ya que a medida que la altura va aumentando, el peso también lo hace. También se observa que es una correlación bastante lineal. La mayoría de los atletas se concentran en el área comprendida entre 62 y 84 kgs de peso, y 172 y 187 cms de altura. Esto corresponde aproximadamente con el primer y tercer cuantil de cada variable, es decir, que se observa aproximadamente un 50% de los atletas dentro de esta región.

# Para poder visualizar la estadística de resumen para la altura y el peso de los jugadores que ganaron cada tipo de medalla en los datos filtrados separados por género, podemos tener dos posibilidades. La primera, es usar el mismo diagrama de dispersión pero usando la opción de hue para ver como se comportan las dos variables en diferentes grupos.

# In[64]:


#Gráfico 1, donde se muestra el comportamiento de las variables altura y peso para cada género
sns.jointplot(x=best_df.Weight, y=best_df.Height, color="#4CB391", hue=best_df.Sex)

#Gráfico 2, donde se muestra el comportamiento de las variables altura y peso para cada medalla
sns.jointplot(x=best_df.Weight, y=best_df.Height, color="#4CB391", hue=best_df.Medal)


# Se puede observar en el primer gráfico de dispersión la misma correlación positiva entre peso y altura para ambos grupos. Adicionalmente, se observa que para los atletas masculinos los valores de peso y altura tienden a ser mayores que para las atletas femeninas, como se había observado en el punto 1.1. 
# 
# En el segundo gráfico, mirando ambas variables por medallas, si bien se mantiene la correlación positiva entre ambas, no hay una diferencia significativa entre los atletas que ganaron medallas de oro, plata o bronce.
# 
# Verifiquemos esto con un diagrama de violín:

# In[65]:


fig, ax=plt.subplots(1,2, figsize=(15,7))

sns.violinplot(x='Sex', y='Height', data=best_df, hue='Medal', ax=ax[0])
ax[0].set_xlabel("Sexo")
ax[0].set_ylabel("Altura")
ax[0].set_title("Diagrama de Violines para la Altura")

sns.violinplot(x='Sex', y='Weight', data=best_df, hue='Medal', ax=ax[1])
ax[1].set_xlabel("Sexo")
ax[1].set_ylabel("Peso")
ax[1].set_title("Diagrama de Violines para el Peso")

plt.show()


# Cómo se puede observar en los diagramas de violín, la tendencia general se mantiene, es decir, que en general el peso y altura promedio son mayores en los atletas masculinos que las atletas femeninas. Pero no hay una diferencia significativa entre los atletas ganadores de las diferentes medallas (oro, plata y bronce).

# ## Ejercicio 1.3
# 
# En esta actividad, utilizaremos los archivos `co2.csv` y `gapminder.csv`. El primero consiste en las emisiones de dióxido de carbono por persona por año y por país, mientras que el segundo consiste en el PIB por año y por país. Es importante que se prueben varios tipos de visualización para para determinar la visualización que mejor transmite el mensaje que está tratando de dar con sus datos. Vamos a crear algunas visualizaciones interactivas utilizando la biblioteca **`Plotly Express`** para determinar cuál es la que mejor se adapta a nuestros datos.
# 
# 1. Vuelve a crear el **DataFrame** de las emisiones de dióxido de carbono y del PIB.
# 2. Crea un gráfico de dispersión con los ejes **x** e **y** como **year** y **co2** respectivamente. Añada un histograma para los valores de **co2** con el parámetro **marginaly_y**.
# 3. Crea un gráfico de caja para los valores del PIB con el parámetro **marginal_x**. Añada los parámetros de parámetros de animación en la columna del año.
# 4. Crea un gráfico de dispersión con los ejes **x** e **y** como **gdp** y **co2** respectivamente.
# 5. Cree un contorno de densidad con los ejes **x** e **y** como **gdp** y **co2** respectivamente.

# Se cargan los datos de las emisiones de dióxido de carbono y del PIB.

# In[66]:


url_co2 = 'https://raw.githubusercontent.com/lihkir/Uninorte/main/AppliedStatisticMS/DataVisualizationRPython/Lectures/Python/PythonDataSets/co2.csv'
co2 = pd.read_csv(url_co2)
co2.head()


# In[67]:


url_gm = 'https://raw.githubusercontent.com/lihkir/Uninorte/main/AppliedStatisticMS/DataVisualizationRPython/Lectures/Python/PythonDataSets/gapminder.csv'
gm = pd.read_csv(url_gm)
gm.head()


# Se realiza la misma transformación hecha en clase para tener todos los datos agregados en una misma tabla:

# In[68]:


df_gm = gm[['Country', 'region']].drop_duplicates()
df_w_regions = pd.merge(co2, df_gm, left_on='country', right_on='Country', how='inner')
df_w_regions = df_w_regions.drop('Country', axis='columns')

new_co2 = pd.melt(df_w_regions, id_vars=['country', 'region'])
columns = ['country', 'region', 'year', 'co2']
new_co2.columns = columns
df_co2 = new_co2[new_co2['year'].astype('int64') > 1963]
df_co2 = df_co2.sort_values(by=['country', 'year'])
df_co2['year'] = df_co2['year'].astype('int64')

df_gdp = gm[['Country', 'Year', 'gdp']]
df_gdp.columns = ['country', 'year', 'gdp']

data = pd.merge(df_co2, df_gdp, on=['country', 'year'], how='left')
data = data.dropna()

data.head()


# Generemos un gráfico de dispersión con los ejes **x** e **y** como **year** y **co2** respectivamente. Para observar de mejor manera las tendencias, primero agregaremos los datos por región.

# In[69]:


#Primero, realizamos un groupby con la region y año, para sumar el co2 por región
co2_gdp_agg = data.groupby(["region","year"])["co2","gdp"].sum().reset_index()
co2_gdp_agg.head()


# In[70]:


fig = px.scatter(co2_gdp_agg, x="year", y="co2", marginal_y="histogram", color="region")
fig.show(renderer='notebook')


# Se observa que en regiones como Europa y Asia Central, y Oriente Medio y el Norte de África, si bien la tendencia de las emisiones de dióxido de carbono por persona era alcista entre 1970 y 1980, luego tuvo una estabilización (en el caso de Europa y Asia Central), y una bajada en el caso de Oriente Medio y el Norte de África. En los últimos años, ambas regiones registraron tendencias a la baja.
# 
# El resto de regiones, si bien, algunas han tenido cambios en las tendencias entre 1970 y 1980, como es el caso de América y APAC, en general, han venido subiendo las emisiones constantemente desde 1980, en una tendencia alcista.
# 
# Observemos ahora el cambio de la distribución del PIB a través de los años para los países:

# In[71]:


fig = px.box(data, x="gdp", animation_frame="year", hover_data=["country","gdp"])
fig.show()


# In[72]:


#Para generar la visualización a tráves de los años
px_off_plot(fig)


# A través de la animación, se puede observar el cambio en la distribución del PIB a lo largo de los años. En general, se observa un crecimiento sostenido, con bajadas en algunos años debido a algunas crisis económicas. Esto se puede observar por ejemplo, en los años de 1982 - 1984 (crisis del petróleo en Medio Oriente), en los años de 1990 a 1992 (caída de la Unión Soviética), y en los años 2008 a 2009 (Crisis Subprime en USA). 
# 
# Generemos ahora un gráfico de dispersión con los ejes **x** e **y** como **gdp** y **co2** respectivamente.

# In[73]:


fig = px.scatter(data, x="gdp", y="co2")
fig.show()


# Como se observa en el gráfico de dispersión, hay una tendencia clara entre la cantidad de emisiones de CO2 emitidas y el PIB de un país, en general se observa un correlación positiva. A mayor PIB, mayores emisiones de CO2. 
# 
# Observemos ahora el mismo gráfico, pero agregado por regiones:

# In[74]:


fig = px.scatter(co2_gdp_agg, x="gdp", y="co2", color="region", hover_data=["year"])
fig.show()


# Cómo se observa, la misma tendencia se mantiene para la mayoría de regiones, aunque para la región de Europa y Asia Central, hay una especie de estabilización, cuando la economía alcanza cierto nivel de PIB, no necesariamente emite más CO2.
# 
# Por último, generemos un contorno de densidad para las mismas variables:

# In[75]:


fig = px.density_contour(data, x="gdp", y="co2", range_x=[-2000,10000], range_y=[-2, 4])
fig.update_traces(contours_coloring="fill", contours_showlabels = True)
fig.show(renderer='notebook')


# Se observa en el gráfico que la mayoría de contornos están en la esquina inferior izquierda del gráfico, que representa los países con bajo PIB y bajas emisiones de CO2. Haciendo zoom en la gráfica, se observan seis contornos.

# ## Ejercicio 1.4
# Trabajaremos con el conjunto de datos de Google Play Store Apps alojado en googleplaystore.csv. Su tarea es crear una visualización con:
# - Un gráfico de barras de un número de aplicaciones estratificado por cada categoría Content Rating (calificado por Everyone/Teen).
# - Un mapa de calor que indica el número de aplicaciones estratificadas por app Category y rangos de rangos segmentados por Rating. El usuario debe poder interactuar con el gráfico seleccionando cualquiera de los tipos de Content Rating y el cambio correspondiente debería reflejarse en el mapa de calor para incluir sólo el número de aplicaciones en la categoría Content Rating.
# 
# Pasos principales
# - Descargue el conjunto de datos [googleplaystore.csv](https://raw.githubusercontent.com/lihkir/Uninorte/main/AppliedStatisticMS/DataVisualizationRPython/Lectures/Python/PythonDataSets/googleplaystore.csv) y formatéelo como un `pandas` `DataFrame`
# - Elimina las entradas del `DataFrame` que tienen valores de característica de `NA`.
# - Cree el gráfico de barras necesario del número de aplicaciones en cada categoría **Content Rating**
# - Cree el mapa de calor necesario indicando el número de aplicaciones en la app en rangos **Category** y **Rating**
# - Combine el código del gráfico de barras y del mapa de calor y cree una visualización con ambos gráficos vinculados dinámicamente entre sí.
# - Interprete cada visualización

# Primero importamos el archivo de datos que contiene el detalle de aplicaciones de Google Play Store.

# In[76]:


url_app = 'https://raw.githubusercontent.com/lihkirun/AppliedStatisticMS/main/DataVisualizationRPython/Lectures/Python/PythonDataSets/googleplaystore.csv'
apps = pd.read_csv(url_app, sep=',')
print(apps.shape)
apps.head()


# Al verificar qué columnas tienen valores de características NA, encontramos que rating, type, content rating, current version y android version.

# In[77]:


apps.isna().sum(axis=0)


# Procedemos entonces a eliminar dichas entradas:

# In[78]:


apps=apps.dropna()
apps.reset_index(drop=True, inplace=True)
apps.isna().sum(axis=0)
print(apps.shape)
apps.head()


# Por Content Rating, podemos ver que el mayor número de aplicaciones corresponde a la categoría Everyone, seguido de Teen. Las categorias Everyone 10+ y Mature 17+ tienen número de aplicaciones muy cercanas y no hay aplicaciones en las categorías Adults only 18+ y Unrated.

# In[79]:


alt.Chart(apps).mark_bar().encode(
    x = 'Content Rating:N',
    y = 'count():Q'
).properties(width=400).interactive()


# Al revisar el número de aplicaciones estratificadas por Category y Rating, vemos que el mayor número se encuentra en la categoría de familia y rating de 4.0 con una cantidad superior a 600.

# In[80]:


heatmap = alt.Chart(apps).mark_rect().encode(
    alt.Y('Rating:Q', bin = True),
    alt.X('Category:N'),
    alt.Color('count():Q', scale = alt.Scale(scheme='greenblue'), legend = alt.Legend(title='Total Apps'))
)

circles = heatmap.mark_point().encode(
    alt.ColorValue('grey'),
    alt.Size('count()', legend = alt.Legend(title='Records in Selection'))
)

heatmap + circles


# Se puede observar que las categorías de las apps en el Google PlayStore con más rating son Familia, en donde se observan al menos 800 apps con un rating entre 4.0 y 4.5, y 600 apps con rating entre 4.5 y 5.
# 
# El segundo lugar es para la categoría de Juegos, con al menos 600 apps con rating entre 4.0 y 4.5, y 400 apps con rating entre 4.5 y 5.
# 
# El tercer lugar se lo lleva la categoría de herramientas ("Tools"), con 400 apps con rating entre 4.0 y 4.5.
# 
# Realicemos el mismo heatmap, pero teniendo en cosideración el Content Rating:

# In[81]:


selected_region = alt.selection(type="single", encodings=['x'])

heatmap = alt.Chart(apps).mark_rect().encode(
    alt.Y('Rating:Q', bin = True),
    alt.X('Category:N'),
    alt.Color('count()', scale = alt.Scale(scheme = 'greenblue'), legend = alt.Legend(title = 'Total Apps'))
).properties(
    width=600
)

circles = heatmap.mark_point().encode(
    alt.ColorValue('grey'),
    alt.Size('count()', legend = alt.Legend(title='Records in Selection'))
).transform_filter(
    selected_region
)

bars = alt.Chart(apps).mark_bar().encode(
    x = 'Content Rating:N',
    y = 'count():Q',
    color = alt.condition(selected_region, alt.ColorValue("steelblue"), alt.ColorValue("grey"))
).properties(
    width=600
).add_selection(selected_region)

heatmap + circles | bars


# Al hacer clic en cada gráfico de barras, el mapa de calor se actualiza con la información de las descargas por categoría y rango correspondientes al Content Rating Seleccionado.
# 
# Si miramos el Content Rating de "Everyone", más o menos se mantienen las mismas relaciones encontradas en el heatmap general, siendo la primera categoría Familia, seguida de Juegos y Herramientas.
# 
# En cambio, al seleccionar "Mature 17+", se observa que los records seleccionados cambian, siendo la categoría principal las apps de citas, seguidas de las apps de compras.
