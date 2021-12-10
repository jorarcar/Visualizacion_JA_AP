# Proyecto Final Python - Curso de Visualización en Python y R
## Jorge Arteaga y Adriana Palacio.
### Maestría en Estadística Aplicada. Universidad del Norte. 2021.

El presente Jupyter-Book contiene los dos talleres entregados, junto con el informe final del proyecto de Python para la materia de Visualización en Python y R. El proyecto escogido analiza las cinco principales acciones colombianas cotizadas en la bolsa de Nueva York. Además de brindar un análisis sobre los datos históricos de las acciones, también se implementará un modelo ARIMA predictivo para predecir valores futuros.

En el menú, se encuentra el informe análisis descriptivo realizado a las acciones. Las acciones seleccionadas fueron las siguientes:

* Ecopetrol.
* Grupo Bancolombia.
* Grupo Aval.
* Tecnoglass.
* Avianca.

Se tomó en primera instancia, los últimos cinco años de los datos históricos de dichas acciones. Esta información fue extraída de Yahoo Finance. Se cargaron los datos históricos en Heroku-Postgres, y luego, se actualizaron con información de la FinnHub API. 

El objetivo general del proyecto final es encontrar como están distribuidos los precios de cierre, volumen de transacciones y las tendencias históricas de las acciones. Para esto, se realizaran los siguientes objetivos específicos:

* Realizar un análisis exploratorio inicial de las acciones.
* Generar modelos de ML (regresión lineal y ARIMA) para generar predicciones en las acciones seleccionadas.
* Crear un dashboard con `Dash` para mostrar los resultados obtenidos.


