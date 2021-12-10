#Modulos nativos de Python
import warnings
import datetime as dt

#Modulos de visualización
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

#Modulos para gestionar dataframes, cálculos con vectores y matrices
import numpy  as np
import pandas as pd

warnings.filterwarnings("ignore")

#Función para el diagrama de dispersión con dos acciones
def scatter_plot(df, df2, st, st2):
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=df["Close"], y=df2["Close"],
                        mode='markers'))

    # Set x-axis title
    fig.update_xaxes(title_text=str(st)+" (USD)")

    # Set y-axes titles
    fig.update_yaxes(title_text=str(st2)+" (USD)")

    return fig

#Función para generar el histograma de precios
def hist_close_price(df, st):
    fig = go.Figure()
    
    #Add traces
    fig.add_trace(go.Histogram(
        x=df["Close"], name=st, marker_color='#FF4136'))
       
    # Set x-axis title
    fig.update_xaxes(title_text="Precio (USD)")

    # Set y-axes titles
    fig.update_yaxes(title_text="Frecuencia")    
    return fig

#Función para generar el diagrama de velas
def candle_sticks(df, st, flag=0):
    fig = go.Figure(data=[go.Candlestick(x = df.Date,
                                         open = df.Open, 
                                         high = df.High,
                                         low = df.Low, 
                                         close = df.Close)
                         ])
    
    if flag == 0:
        xaxis = "Day"
    else:
        xaxis = "Minutes"
    
    fig.update_layout(
        title=str(st),
        xaxis_title=xaxis,
        yaxis_title="USD",
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="RebeccaPurple"
        )
    )
    fig.update_layout(xaxis_rangeslider_visible=False)
    return fig

#Función para generar el diagrama de volumen y precio de cierre
def volume_price(df, st):
    fig = make_subplots(rows=1, cols=1, \
        specs=[
                [{"secondary_y": True}]
              ])

    #Adding traces
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["Close"], name="Precio Cierre"),
        secondary_y=False, row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["Volume"], name="Volumen"),
        secondary_y=True, row=1, col=1
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Fecha")

    # Set y-axes titles
    fig.update_yaxes(title_text="USD", secondary_y=False)
    fig.update_yaxes(title_text="Número de Transacciones", secondary_y=True)

    return fig

#Función para el diagrama de dispersión con dos acciones
def pred_plot(x, y_true, predictions, st):
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=x, y=y_true,
                        mode='line+markers',
                        name="Test"))

    fig.add_trace(go.Scatter(x=x, y=predictions,
                        mode='line+markers',
                        name="Forecast"))

    # Set x-axis title
    fig.update_xaxes(title_text="Fecha")

    # Set y-axes titles
    fig.update_yaxes(title_text=str(st)+" (USD)")

    return fig