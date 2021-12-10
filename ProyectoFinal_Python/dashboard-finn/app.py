########################################################################################################################
############################################## MODULOS PARA EL DASHBOARD ###############################################
########################################################################################################################

# -*- coding: utf-8 -*-

#Modulos nativos de Python
import datetime as dt

#Modulos para gestionar dataframes, cálculos con vectores y matrices
import numpy  as np
import pandas as pd

#Modulos de visualización
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px
import seaborn as sns

#Modulos del Dash para generar el dashboard
from dash import Dash, dcc, html, Input, Output, State, callback_context
from flask import Flask
server = Flask(__name__)

#Modulos para las pruebas estadísticas para series de tiempo
from statsmodels.tsa.stattools import adfuller

#Modulos para la generación de modelos
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

#Modulos para manejar la base de datos
import psycopg2
from sqlalchemy import create_engine

#Modulos para obtener datos de las acciones actualizadas
import finnhub

#Tema de los gráficos
sns.set_theme()
sns.set_context("paper")

######################## Importando funciones creadas para el funcionamiento del DashBoard ###############

#Importando funciones para la actualización de la base de datos
import update_database as up_db

#Importando funciones para gráficar
import graphs as gph

#importing model functions
import model_fn as md

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#app = dash.Dash(__name__, server=server, routes_pathname_prefix='/dash/',
#                external_stylesheets=external_stylesheets)
#
#app2 = dash.Dash(__name__, server=server, routes_pathname_prefix='/model/',
#                external_stylesheets=external_stylesheets)

#Local server
app = Dash(__name__, server=server, routes_pathname_prefix='/dash/',
                external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)

app2 = Dash(__name__, server=server, routes_pathname_prefix='/model/',
                 external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)


#######################################################################################################################
#########################################  Conexión a la base de datos ################################################
#######################################################################################################################

#Creando el motor para leer los datos de nuestra base de datos
engine = create_engine('postgresql://snfifneqljoybc:e1aaa689afeaaff2f75921f4fd6ca3816f6ebf33bdaf0c3366b9db4e5fe218fe@ec2-3-95-130-249.compute-1.amazonaws.com:5432/d3go0j5dkhm396')

#Conectandose a la base de datos
connection = psycopg2.connect(user="snfifneqljoybc",
                                  password="e1aaa689afeaaff2f75921f4fd6ca3816f6ebf33bdaf0c3366b9db4e5fe218fe",
                                  host="ec2-3-95-130-249.compute-1.amazonaws.com",
                                  port="5432",
                                  database="d3go0j5dkhm396")

#Configurando el cliente de FinnHub
finnhub_client = finnhub.Client(api_key="c5tpk32ad3i9n9aj0u3g")

#Creando función para cargar las bases de datos
def cargar_db():
    #Leyendo la base de datos
    ec_df = pd.read_sql('SELECT * FROM "EC"', connection)
    aval_df = pd.read_sql('SELECT * FROM "AVAL"', connection)
    bc_df = pd.read_sql('SELECT * FROM "CIB"', connection)
    tgls_df = pd.read_sql('SELECT * FROM "TGLS"', connection)
    av_df = pd.read_sql('SELECT * FROM "AVHOQ"', connection)
    
    #Convirtiendo las fechas a tipo DateTime
    ec_df.Date = ec_df.Date.apply(lambda x: dt.datetime.strptime(str(x).strip(), '%Y-%m-%d')) 
    bc_df.Date = bc_df.Date.apply(lambda x: dt.datetime.strptime(str(x).strip(), '%Y-%m-%d'))
    av_df.Date = av_df.Date.apply(lambda x: dt.datetime.strptime(str(x).strip(), '%Y-%m-%d'))
    aval_df.Date = aval_df.Date.apply(lambda x: dt.datetime.strptime(str(x).strip(), '%Y-%m-%d'))
    tgls_df.Date = tgls_df.Date.apply(lambda x: dt.datetime.strptime(str(x).strip(), '%Y-%m-%d'))

    #Acciones que se mostrarán, con su dataframe correspondiente
    stocks = dict(zip(["EC", "AVAL", "CIB", "TGLS", "AVHOQ"],
        [ec_df, aval_df, bc_df, tgls_df, av_df]
    ))
    return stocks

#stocks = cargar_db()

#Diccionario para las opciones de tiempo
time_stocks = dict(zip(["7", '30', '60', "180", "1Y", "ALL"],
    [7, 30, 60, 180, 360, "ALL"]
))

#Diccionario para los modelos ARIMA (calculados en el EDA)
best_order = dict(zip(
    ["EC", "AVAL", "CIB", "TGLS", "AVHOQ"],
    [(6,1,9),(5,0,4),(0,1,7),(9,2,9),(7,1,7)]
))

#Graph config
config_graph = {'displayModeBar': True,
              'scrollZoom': False,
              'autosizable':True,
              'showAxisDragHandles':False}

#Dummy fig
data_canada = px.data.gapminder().query("country == 'Canada'")
fig = px.bar(data_canada, x='year', y='pop')

######################################################################################################################
#############################################   Dashboard Layout  ####################################################
######################################################################################################################

# Create dashboard app layout
app.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),

        #First row (Image and Title)
        html.Div(
            [
                #Image Uninorte
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("assets/1200px-Logo_uninorte_colombia.jpg"),
                            id="plotly-image",
                            style={
                                "height": "60px",
                                "width": "auto",
                                "margin-bottom": "25px",
                            },
                        )
                    ],
                    className="one column",
                ),
                #Container for Title
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Proyecto Final DataViz - Acciones Colombianas más importantes en la bolsa de Nueva York",
                                    style={"margin-bottom": "0px"},
                                ),

                            ]
                        )
                    ],
                    className="eleven columns",
                    id="title",
                ),

            ],
            id="header",
            className="row flex-display",
        ),
        #Second container (filter, KPIs, scatter and histogram)
        html.Div(
            [
                #Column Filter
                html.Div(
                    [
                        html.P(
                            "Acción seleccionada:",
                            className="control_label",
                        ),
                        dcc.Dropdown(
                            id="stock1",
                            options=[
                                {"label": "ECOPETROL", "value":"EC"},
                                {"label": "GRUPO AVAL", "value": "AVAL"},
                                {"label": "BANCOLOMBIA", "value": "CIB"},
                                {"label": "TECNOGLASS", "value": "TGLS"},
                                {"label": "AVIANCA", "value": "AVHOQ"}
                            ],
                            multi=False,
                            value="EC",
                            className="dcc_control",
                        ),
                        html.P(
                            "Tiempo seleccionado:",
                            className="control_label",
                        ),
                        dcc.RadioItems(
                            id='time_radio',
                            options=[{'label': i, 'value': i} for i in ["7", '30', '60', "180", "1Y", "ALL"]],
                            value='30',
                            labelStyle={'display': 'inline-block'}
                        ),
                        html.Br(),
                        html.Br(),

                        html.P(
                            "Fechas de la DB: "
                        ),
                        html.P(
                            "[-,-]",
                            id ="Fechas_DB"
                        ),
                        html.Button(
                            id="update_db_button",
                            n_clicks=0, children='Actualizar DB',
                            className="dcc_control",
   
                        ),
                        html.Br(),
                        html.Br(),
                        html.P(
                            "Base de datos no cargada.",
                            id="action_performed_db"
                        ),
                        html.Br(),
                        html.Br(),
                    ],
                    className="pretty_container two columns",
                    id="cross-filter-options",
                ),
                #Container for KPIs
                html.Div(
                    [
                        html.Div(
                            [html.H6(id='mean_price'), html.P("Precio de cierre promedio")],
                            className="mini_container"
                        ),
                        html.Div(
                            [html.H6(id='min_price'), html.P("Precio de cierre mínimo")],
                            className="mini_container",
                        ),
                            html.Div(
                            [html.H6(id='max_price'), html.P("Precio de cierre máximo")],
                            className="mini_container",
                        ),
                        html.Div(
                            [html.H6(id='adf_test'), html.P("P-Value: ADF Fuller Test")],
                            className="mini_container",
                        ),
                    ],
                    id="right-column",
                    className="pretty_container two columns",
                ),

                #Container for scatter plot 
                html.Div(
                    [
                        html.Div([
                               html.P(
                                    "Acción a comparar: ",
                                    className="control_label",
                                ),
                                dcc.Dropdown(
                                    id="stock2",
                                    options=[
                                        {"label": "ECOPETROL", "value": "EC"},
                                        {"label": "GRUPO AVAL", "value": "AVAL"},
                                        {"label": "BANCOLOMBIA", "value": "CIB"},
                                        {"label": "TECNOGLASS", "value": "TGLS"},
                                        {"label": "AVIANCA", "value": "AVHOQ"}
                                    ],
                                    multi=False,
                                    value="AVAL",
                                    className="dcc_control",
                                ),
                        ]),
                        html.Div(
                                [html.P("Gráfico de dispersión"),
                                    dcc.Graph(id="scatter_plot",config = config_graph)
                                    ],
                                className="pretty container",
                    )],
                    id="scatt-column",
                    className="pretty_container four columns",
                ),

                #Container for histogram plot 
                html.Div(
                    [
                        html.Div(
                            [html.P("Histograma para el precio de cierre"),
                                dcc.Graph(id="histogram",config = config_graph)
                                ],
                            className="pretty container",
                        ),
                    ],
                    id="hist-column",
                    className="pretty_container four columns",
                ),
            ],
            className="row flex-display",
        ),
        #Container for candle sticks and volume vs price
        html.Div(
            [
                #Container for candle sticks
                html.Div(
                    [
                        html.Div([
                            dcc.RadioItems(
                                id='candle_radio',
                                options=[{'label': i, 'value': i} for i in ['DB', 'Latest 24h']],
                                value='DB',
                                labelStyle={'display': 'inline-block'}
                            ),
                        ], 
                        className="row container-display"
                        ),

                        html.Div([
                            dcc.Graph(id='candle_stick', figure=fig, config = config_graph)
                        ])
                        ],
                        id="candle_container",
                        className="pretty_container six columns",
                        ),
                #Container for Volume vs Price
                html.Div(
                    [   html.P("Gráfico de líneas para el volumen y precio de cierre"),
                        dcc.Graph(id='volume_price', figure=fig,config = config_graph)],
                        id="volume_container",
                        className="pretty_container six columns",
                        ),
            ],
            className="row flex-display",
        ),

    #html.A('Go to predictor Model', href='http://team16.pythonanywhere.com/model/'),
    html.A('Go to forecasting', href='/model/'),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)

########################## Model Layout ##########################

app2.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        #First row (Image and Title)
        html.Div(
            [
                #Image Uninorte
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("assets/1200px-Logo_uninorte_colombia.jpg"),
                            id="plotly-image",
                            style={
                                "height": "60px",
                                "width": "auto",
                                "margin-bottom": "25px",
                            },
                        )
                    ],
                    className="one column",
                ),
                #Container for Title
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Proyecto Final DataViz - Modelos Predictivos",
                                    style={"margin-bottom": "0px"},
                                ),

                            ]
                        )
                    ],
                    className="eleven columns",
                    id="title",
                ),

            ],
            id="header",
            className="row flex-display",
        ),
       #Second container (filter, KPIs, scatter and histogram)
        html.Div(
            [
                #Column Filter
                html.Div(
                    [
                        html.P(
                            "Acción seleccionada:",
                            className="control_label",
                        ),
                        dcc.Dropdown(
                            id="stock1_model",
                            options=[
                                {"label": "ECOPETROL", "value":"EC"},
                                {"label": "GRUPO AVAL", "value": "AVAL"},
                                {"label": "BANCOLOMBIA", "value": "CIB"},
                                {"label": "TECNOGLASS", "value": "TGLS"},
                                {"label": "AVIANCA", "value": "AVHOQ"}
                            ],
                            multi=False,
                            value="EC",
                            className="dcc_control",
                        ),
                        html.P(
                            "Horizonte seleccionado:",
                            className="control_label",
                        ),
                        dcc.RadioItems(
                            id='time_radio2',
                            options=[{'label': str(i), 'value': i} for i in [7, 14, 21, 28]],
                            value=14,
                            labelStyle={'display': 'inline-block'}
                        ),
                        html.Br(),
                        html.Br(),

                        html.P(
                            "Tipo de Modelo:",
                            className="control_label",
                        ),
                        dcc.RadioItems(
                            id='model_radio',
                            options=[{'label': i, 'value': i} for i in ["Lineal", "ARIMA"]],
                            value="Lineal",
                            labelStyle={'display': 'inline-block'}
                        ),
                        html.Br(),
                        html.Br(),
                        html.P(
                            "Parámetros del modelo: "
                        ),
                        html.P(
                            "[-,-]",
                            id ="param_model"
                        ),
                        html.Br(),
                        html.Br(),                    
                    ],
                    className="pretty_container two columns",
                    id="cross-filter-options",
                ),
                #Container for KPIs
                html.Div(
                    [
                        html.Div([
                            html.Div(
                                [html.H6(id='mae'), html.P("MAE")],
                                className="pretty_container three columns mini_container"
                            ),
                            html.Div(
                                [html.H6(id='mse'), html.P("MSE")],
                                className="pretty_container three columns mini_container",
                            ),
                                html.Div(
                                [html.H6(id='mape'), html.P("MAPE")],
                                className="pretty_container three columns mini_container",
                            ),
                            html.Div(
                                [html.H6(id='rmse'), html.P("RMSE")],
                                className="pretty_container three columns mini_container",
                            ),
                        ],),
                        #Container for line plot 
                        html.Br(),
                        html.Br(),
                        html.Div(
                                [   
                                    html.Br(),
                                    html.H6("Predicción del modelo para la acción seleccionada"),
                                    dcc.Graph(id="pred_model",config = config_graph)
                                ], id="pred-column",
                            ),
                    ],
                    id="right-column2",
                    className="pretty_container ten columns",
                ),
            ],
            className="row flex-display",
        ),
    #html.A('Go to Dashboard', href='http://team16.pythonanywhere.com/dash/'),
    html.A('Go to Dashboard', href='/dash/'),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)

# Create callbacks
@app.callback(
    [
        Output(component_id='Fechas_DB', component_property='children'),
        Output(component_id='action_performed_db', component_property='children'),
        Output(component_id='mean_price', component_property='children'),
        Output(component_id='min_price', component_property='children'),
        Output(component_id='max_price', component_property='children'),
        Output(component_id='adf_test', component_property='children'),
        Output('scatter_plot', 'figure'),
        Output('histogram', 'figure'),
        Output('candle_stick', 'figure'),
        Output('volume_price', 'figure'),
    ],
    [
        Input(component_id='stock1', component_property='value'),
        Input(component_id="time_radio", component_property="value"),
        Input(component_id='update_db_button', component_property='n_clicks'),
        Input(component_id='stock2', component_property='value'),
        Input(component_id='candle_radio', component_property='value'),
    ],
)
def update_output_div(stock_value,stock_time, update_db_but, stock_value2, candle_opt):

    #cargando la base de datos
    stocks = cargar_db()

    #Seleccionando el dataframe correspondiente a los dropdown
    df = stocks[stock_value]
    df2 = stocks[stock_value2]

    #Mensaje inicial
    message_DB = "Base de datos cargada exitosamente."

    #Actualizando los valores de las fechas del DB
    dates_DB = "[ "+dt.datetime.strftime(df.Date.min(), '%Y-%m-%d')+", "+dt.datetime.strftime(df.Date.max(), '%Y-%m-%d')+"]"

    #Revisar si el botón de update_DB fue clickeado
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if "update_db_button" in changed_id:
        for st in stocks.keys():
            message_DB = up_db.update_stock(st,finnhub_client,connection, engine)
        stocks = cargar_db()
        df = stocks[stock_value]
        df2 = stocks[stock_value2]
        dates_DB = "[ "+dt.datetime.strftime(df.Date.min(), '%Y-%m-%d')+", "+dt.datetime.strftime(df.Date.max(), '%Y-%m-%d')+"]"

    #Seleccionando el tiempo
    t = time_stocks[stock_time]

    if isinstance(t, (int, float)):
        n = len(df.Close)
        begin_size = n - t

        df = df.iloc[begin_size:begin_size + t]
        df2 = df2.iloc[begin_size:begin_size + t]
    else:
        pass
    

    #Calculando los valores máximos, minimos, promedios y realizando el test de AD Fuller
    mean_price = np.round(df.Close.mean(),2)
    min_price = np.round(df.Close.min(),2)
    max_price = np.round(df.Close.max(),2)
    p_value = np.round(adfuller(df.Close)[1],2)

    #Generando los gráficos
    scatter = gph.scatter_plot(df, df2, stock_value,stock_value2)
    hist = gph.hist_close_price(df, stock_value)

    if candle_opt=="DB":
        candle = gph.candle_sticks(df, stock_value)
    else:
        #Buscando la información en tiempo real de las últimas 24 horas, por minuto
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(hours=24)

        #Transformando a Unix
        end = int(end_date.timestamp())
        start = int(start_date.timestamp())

        #Buscando la información en FinnHub
        res = finnhub_client.stock_candles(stock_value, 1, start, end)
        df_res = pd.DataFrame(res).drop(columns=["s"], errors="ignore")

        #Cambiando los nombres de las columnas
        df_res.rename(columns = {'c':'Close','h':'High','l':'Low','o':'Open','t':'Date','v':'Volume'}, inplace = True)

        #transformando a datetime
        df_res.Date = df_res.Date.apply(lambda x: dt.datetime.fromtimestamp(x)) 

        #Actualizando los KPIs
        mean_price = np.round(df_res.Close.mean(),2)
        min_price = np.round(df_res.Close.min(),2)
        max_price = np.round(df_res.Close.max(),2)
        p_value = np.round(adfuller(df_res.Close)[1],2)

        #Llamando a la función
        candle = gph.candle_sticks(df_res, stock_value, flag=1)

    volume = gph.volume_price(df, stock_value)

    return dates_DB, message_DB, mean_price, min_price, max_price, p_value, scatter, hist, candle, volume

#Create callbacks of model
@app2.callback(
    [
        Output(component_id='param_model', component_property='children'),
        Output(component_id='mae', component_property='children'),
        Output(component_id='mse', component_property='children'),
        Output(component_id='mape', component_property='children'),
        Output(component_id='rmse', component_property='children'),        
        Output('pred_model', 'figure'),
    ],
    [
        Input(component_id='stock1_model', component_property='value'),
        Input(component_id="time_radio2", component_property="value"),
        Input(component_id='model_radio', component_property='value'),
    ],
)
def update_forecast(stock_value, horizon, model_type):

    #Cargando la base de datos
    stocks = cargar_db()
    #Seleccionando el dataframe
    df = stocks[stock_value]
    if model_type == "Lineal":
        linear_model = md.OLS_temp(df, horizon)
        #Actualizando los parámetros del modelo
        params = "Intercepto: "+str(linear_model[4].params[0])+"\n Pendiente: "+str(linear_model[4].params[1])
        #print(params)
        predictions = md.linear_rolling(*linear_model) 
        x = linear_model[2]
        y_true = linear_model[3]
    else:
        #Cargando el mejor orden encontrado
        order = best_order[stock_value]
        params = str(order)
        n = len(df.Close)
        train_size = n - horizon
        #Creando cada dataframe para train y test
        train = df.Close.iloc[:train_size]
        y_true = df.Close.iloc[train_size:train_size + horizon] 
        x = df.Date.iloc[train_size:train_size + horizon]
        predictions = md.arima_rolling(train.to_list(), y_true.to_list(), order)
    #Prediciendo los errores de predicción
    metricas = md.forecast_accuracy(np.array(predictions), np.array(y_true),stock_value)

    #Graficando
    #print(predictions)
    #print(y_true)
    #print(x.shape)
    line_pred = gph.pred_plot(x, y_true, predictions, stock_value)
    return params, np.round(metricas["MAE"],2), np.round(metricas["MSE"],2), np.round(metricas["MAPE"],2), np.round(metricas["RMSE"],2), line_pred


# Main
#if __name__ == "__main__":
#    app.run_server(debug=True)

#Main
if __name__ == "__main__":
    app.run_server(debug=True, port=80, host='0.0.0.0')
