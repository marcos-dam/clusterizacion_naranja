import io
import re
import pandas as pd
import numpy as np
import sys
import os
from sklearn.ensemble import IsolationForest
from datetime import datetime
from time import gmtime, strftime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
try:
    import pyarrow.parquet as pq
except:
    print('Instalando Librerias')
    os.system('pip install pyarrow')
    import pyarrow.parquet as pq
    
    
#====================================================================================================================


def filtra_clientes_activos(original_contracts,data):
    
    """Filtramos clientes dejando aquellos que estan activos (no tienen fecha de baja) 
    y quitamos clientes repetidos del archivo original_contracts. Finalmente se combina con los ids de los clientes
     en el archivo data para obtener un dataframe final denominado contracts. Se busca consistencia en cuanto a tener
     los mismos clientes en contracts y data"""
    
    clientes_activos = original_contracts[original_contracts['fecha_baja'].isna()].cod_cliente.unique()
    original_contracts = original_contracts[original_contracts['cod_cliente'].isin(clientes_activos)]
    contracts = original_contracts[ original_contracts.cod_cliente.isin(data.client_id.unique()) ]
    return contracts


#============================================================================================

def estructura_eventos(data):
    
    """Se crea un dataframe de eventos estructurados a partir 
    de la columna event_type_name del dataframe data."""
    
    structured_events = data[['client_id', 'event_type_name']].pivot_table(index='client_id', columns = ['event_type_name'], aggfunc = len, fill_value = 0)
    structured_events = structured_events.reset_index().rename_axis(None, axis = 1)
    return structured_events
    
#============================================================================================    
    
def crea_var_por_intervalo_de_time(data):   
    
    """Se crean variables por diferentes intervalos de tiempo a partir de la columna 'event_date del dataframe data
    y se agregan al mismo. Estos indican, segun el intervalo, en qué momento ocurrió el evento X."""
   
    #------------------------------------------------------------------------------
    # Parse event date
    data['event_date'] = pd.to_datetime(data.event_date, format = '%d-%b-%y %I.%M.%S.%f %p')
    data['hour'] = [x.hour for x in data.event_date]
    data['day'] = data.event_date.dt.day
    data['month'] = data.event_date.dt.month
    data['weekday'] = data.event_date.map(lambda x: x.weekday())
    data['time_of_month'] = pd.cut(data.day, bins = 4, labels = [1, 2, 3, 4]).astype('int32')
    data['part_of_week'] = data.weekday.isin([5, 6])
    data['part_of_day'] = pd.cut(data.hour, bins = 4, labels = [1, 2, 3, 4]).astype('int32')
    
    return data

#============================================================================================

def cuenta_frecuencia_por_intervalo(data):
    
    """Frecuencia de eventos en cada intervalo de tiempo creado por la funcion crea_var_por_intervalo_de_time().
    Además se renombran cada uno de los intervalos de tiempo dentro de cada columna. Por ej: para la columna
    part_of_day, se renombra cada intervalo (1,2,3,4) como ('dawn', 'morning', 'afternoon', 'evening'). Lo mismo
    para part_of_week y time_of_month. A continuación se cuentan cuántos eventos ocurrieron en cada intervalo."""
    
    # Watch intensity by part of day
    #------------------------------------------------------------------------------
    frec_x_day = data[['client_id', 'part_of_day']].\
                         pivot_table(index='client_id', columns = 'part_of_day', aggfunc = 'size', fill_value = 0).\
                         reset_index().rename_axis(None, axis=1)
    frec_x_day.columns = ['client_id', 'dawn', 'morning', 'afternoon', 'evening']
    # data_intensity_day.head()

    frec_x_week = data[['client_id', 'part_of_week']].\
                          pivot_table(index='client_id', columns = 'part_of_week', aggfunc = 'size', fill_value = 0).\
                          reset_index().rename_axis(None, axis=1)
    frec_x_week.columns = ['client_id', 'weekday', 'weekend']
    # data_intensity_week.head()

    frec_x_month = data[['client_id', 'time_of_month']].\
                           pivot_table(index='client_id', columns = ['time_of_month'], aggfunc = 'size', fill_value = 0).\
                           reset_index().rename_axis(None, axis=1)
    frec_x_month.columns = ['client_id', 'early_month', 'midearly_month', 'midlate_month', 'late_month']
    # data_intensity_month.head()

    frec_x_interv = frec_x_day.merge(frec_x_week, on = 'client_id', how = 'inner').\
                                     merge(frec_x_month, on = 'client_id', how = 'inner')
    
    # data_intensity.head()
    
    return frec_x_interv



#=================================================================================================


def agrupaciones_por_cliente(contracts):
    
    important_columns = [
        'client_id',
        #------------------------------
        # 'id_cliente',
        # 'cod_cliente',
        # 'ide_cliente',
        'codigo_subtipo_local',
        'codigo_regimen',
        # 'codigo_tipo_contrato',
        # 'codigo_tipo_panel',
        # 'codigo_momento_facturacion',
        'fecha_instalacion',
        # 'fecha_baja',
        # 'codigo_postal_inst',
        'precio_alta',
        # 'precio_total_rec',
        # 'precio_total_nrec',
        # 'ide_pais',
        # 'pais',
        # 'ide_contrato',
        'id_contrato',
        # 'cod_contrato',
        # 'contratopais',
        # 'ind_btc',
        # 'dt',
    ]
    
    # Client id column
    contracts['client_id'] = contracts.cod_cliente
    contracts.precio_alta = contracts.precio_alta.astype('float32')
    
    def mode(x):
        return x.value_counts().index[0]

    # Client contracts summarization
    client_contracts = contracts[important_columns].groupby('client_id').agg({
        'codigo_subtipo_local': mode,
        'codigo_regimen'      : mode,
        'fecha_instalacion'   : 'min',
        'precio_alta'         : ['mean','max','sum'],
        'id_contrato'         : 'count',
    })
    client_contracts.columns = client_contracts.columns.to_flat_index().map(lambda x: '_'.join(x))
    client_contracts = client_contracts.rename(columns={'id_contrato':'cant_contratos'})
    
    # Drop rows with na's
    client_contracts = client_contracts.dropna()
    
    # Codigo Subtipo local
    #------------------------------------------------------------------------------
    keep = client_contracts.codigo_subtipo_local_mode.value_counts().index[:2]
    client_contracts['codigo_subtipo_local_mode_other'] = np.where(client_contracts.codigo_subtipo_local_mode.isin(keep), client_contracts.codigo_subtipo_local_mode, 'OTHER')
    #client_contracts.codigo_subtipo_local_mode_other.value_counts()
    
    one_hot = pd.get_dummies(client_contracts.codigo_subtipo_local_mode_other).astype(float)
    one_hot.columns = [ 'codigo_subtipo_local_' + c.replace(' ','_').lower() for c in one_hot.columns ]
    #one_hot.head()
    
    client_contracts = pd.concat([client_contracts, one_hot ], axis=1)
    
    # Codigo Regimen
    #------------------------------------------------------------------------------
    client_contracts['codigo_regimen_venta'] = (client_contracts.codigo_regimen_mode == 'VENTA').astype(float)
    
    # Antiguedad
    #------------------------------------------------------------------------------
    client_contracts['antiguedad'] = client_contracts.fecha_instalacion_min.map(lambda d: d.toordinal())

    #Me quedo solo con las columnas dummies o con medias y conteos, porque las que tienen string no sirven para clusterizar
    transformed_columns = [
        'codigo_subtipo_local_en_altura',
        'codigo_subtipo_local_other',
        'codigo_subtipo_local_unifamiliar',
        'codigo_regimen_venta',
        'antiguedad',
        'precio_alta_mean',
        'precio_alta_max',
        'precio_alta_sum',
        'id_contrato_count'
    ]
    
    client_contracts = client_contracts[transformed_columns]
    return client_contracts

#============================================================================================

def merge_dataset(frec_x_interv, data_origin, structured_events, client_contracts):
    
    """Mergea 4 dataframes: frec_x_interv, data_origin, structured_events y client_contracts. Todos a partir de la columna
    client_id. Se obtiene un dataframe con datos a nivel."""
        
    structured_dataset = frec_x_interv.merge( data_origin,       on = 'client_id', how = 'inner').\
                                       merge( structured_events, on = 'client_id', how = 'inner').\
                                       merge( client_contracts,  on = 'client_id', how = 'inner')
    
    
    return structured_dataset

#===============================================================================================


def filtra_por_config(structured_dataset, config_file):
    
    """Aquí se introduce el dataframe con datos a nivel para filtrar las columnas(features) que se utilizarán en
    el clusterizado"""
    
    with open(config_file,'r') as file:
        config = file.read()
    config = re.findall("'([\w _\.]+)'\s+(\d)\n", config)
    columns = [ col for col,val in config if val == '1' ]
    
    structured_df = structured_dataset[columns]
    print(structured_df.shape)
    return structured_df


#====================================================================================================================

def escala_datos(method,data_final):
    if method == 'minmax':
        minmax = MinMaxScaler()
        data_minmax = data_final.set_index('client_id')
        data_minmax = minmax.fit_transform(data_minmax)
        return data_minmax
    
    elif method == 'standard':
        minmax = StandardScaler()
        data_scaled = data_final.set_index('client_id')
        data_scaled = minmax.fit_transform(data_scaled)
        return data_scaled
    
    else:
        return 'No se selecciono metodo. Puede usar minmax o standard'

#====================================================================================================================

def get_outliers_out(data_isolation):
    data_isolation = data_isolation.set_index('client_id')
    iso_forest = IsolationForest(contamination = 0.001, random_state = 1234, behaviour = "new")
    iso_forest.fit(data_isolation, 1)
    outlier_pred = iso_forest.predict(data_isolation)
    outlier_pred = [False if x == 1 else True for x in outlier_pred]
    print(pd.Series(outlier_pred).value_counts())
    dataisolet_plus_outlier = pd.DataFrame(list(zip(data_isolation.index,outlier_pred)))
    clientesadropear=dataisolet_plus_outlier[dataisolet_plus_outlier[1]==-1][0].to_list()
    data_isolation_dropped = data_isolation.reset_index()[~(data_isolation.reset_index()['client_id'].isin(clientesadropear))]
    
    
    return data_isolation_dropped
