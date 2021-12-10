#Modulos nativos de Python
import datetime as dt
import warnings

#Modulos para gestionar dataframes, cálculos con vectores y matrices
import numpy  as np
import pandas as pd

#Modulos para manejar la base de datos
import psycopg2
from sqlalchemy import create_engine

#Modulos para obtener datos de las acciones actualizadas
import finnhub

warnings.filterwarnings('ignore')

#Función utilizada para convertir t a un formato legible
def date_format(date_col):
    return dt.datetime.fromtimestamp(date_col).strftime('%Y-%m-%d')

#Función utilizada para actualizar la base de datos
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
    end_date = dt.datetime.now() - dt.timedelta(days=1)
    start_date = dt.datetime.strptime(str(max_date), '%Y-%m-%d')+ dt.timedelta(days=1)
    print(start_date)
    print(end_date)

    if start_date>end_date:
        return ("Tablas al día, no se actualizó.")
    else:
        #Transformando a Unix
        end = int(end_date.timestamp())
        start = int(start_date.timestamp())
    
        #Buscando la información en FinnHub
        res = finnhub_client.stock_candles(stock, 'D', start, end)
        df_res = pd.DataFrame(res).drop(columns=["s"], errors="ignore")

        #Cambiando el formato del tiempo
        df_res['t'] = df_res['t'].apply(date_format)

        #Cambiando los nombres de las columnas
        df_res.rename(columns = {'c':'Close','h':'High','l':'Low','o':'Open','t':'Date','v':'Volume'}, inplace = True)

        #Reorganizando el orden de las columnas
        cols = df_res.columns.to_list()
        cols = cols[4:5] + cols[3:4] + cols[1:3] + cols[0:1] + cols[-1:]
        df_res = df_res[cols]

        if df_res.shape[0] != 0:
            #Actualizando la tabla
            df_res.to_sql(stock, engine, if_exists = 'append', index=False, method='multi')
        return ("Tablas actualizadas.")