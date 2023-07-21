import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
from prophet import Prophet
df = pd.read_csv('silo_feed_weights.csv')
columns_to_be_deleted = ['url','Unnamed: 0', 'silo_profile','pk','volume','food','valid','weight']
for col in columns_to_be_deleted:
  df = df.drop(col,axis=1)
#Silos com lotação de mais de 100% e menos de 0% não fazem sentido
df.drop(df[df['level'] > 100].index, inplace = True)
df.drop(df[df['level'] < 0].index, inplace = True)
#Passa a data pro formato de data do python, ordena em ordem cronológica e retira a informação de fuso 
df['datetime_fetched'] = pd.to_datetime(df['datetime_fetched'])
df.sort_values(by='datetime_fetched', inplace = True)
df['datetime_fetched'] = df['datetime_fetched'].dt.tz_localize(None)
#Criando um dataframe para cada silo
#!pip install hampel
#from hampel import hampel
from statsmodels.tsa.stattools import adfuller as adf
from scipy import signal
dataframe_collection = {}
max_list = []
min_list = []
dataframe_total = pd.DataFrame()
i = 0
for x in df.silo.unique():
  #Cria um dicionário com um dataframe para cada silo 
  dataframe_collection[i] = df[df.silo == x].reset_index(drop=True)
  atual = dataframe_collection[i]
  atual = atual.drop('silo',axis=1)
  #Transforma a data no index do dataframe e faz com que os dados estejam de hora em hora, que é o período médio observado nos dados
  atual.index = atual['datetime_fetched']
  del atual['datetime_fetched']
  atual = atual.resample('H').mean()
  #Interpola linearmente os valores nulos
  atual['level'] = atual['level'].interpolate(option='linear')
  #Mesmo com a interpolação ainda há alguns saltos, então interpola-se novamente nestes saltos
  atual['diff'] = atual['level'].diff()
  atual['level'].loc[atual['diff'] > abs(15)]= np.nan
  atual['level'] = atual['level'].interpolate(option='linear')
  #Silos com lotação de mais de 100% e menos de 0% não fazem sentido
  atual.drop(atual[atual['level'] > 100].index, inplace = True)
  atual.drop(atual[atual['level'] < 0].index, inplace = True)
  #guarda os máximos e mínimos para desfazer a normalização posteriormente 
  min = atual['level'].min()
  max = atual['level'].max()
  max_list.append(max)
  min_list.append(min)
  #Normalização e diferenciação dos dados
  atual['level_norm'] = (atual['level'] - min) / (max- min)
  atual['diff'] = atual['level_norm'].diff()
  if atual['level'].std() > 0.4:
    dataframe_collection[i] = atual
  else:
    del dataframe_collection[i]
  i+=1


def predict(nivel_desejado,silo):
  min = min_list[i]
  max = max_list[i]
  df_atual = dataframe_collection[i]
  df_atual=df_atual.rename(columns={ 'diff':'y'})
  df_atual['ds'] = df_atual.index
  horas = 1
  previsao=0
  m = Prophet(interval_width=0.95)
  m.fit(df_atual)
  while previsao < nivel_desejado:
    future = m.make_future_dataframe(periods=horas, freq='H', include_history = True)
    forecast = m.predict(future)
  #Desfaz a diferenciação e a normalização
    forecast['yhat'] = np.r_[df_atual['level_norm'].iloc[0],forecast['yhat'].iloc[1:]].cumsum()
    forecast['yhat'] = forecast['yhat']*(max-min) + min
  #Novamente limita os valores a 100 e 0 %
    forecast.loc[forecast['yhat'] < 0, 'yhat'] = 0
    forecast.loc[forecast['yhat'] > 100, 'yhat'] = 100
    previsao=forecast['yhat'].iloc[-1]
    data_prevista = forecast['ds'].iloc[-1]
    delta_time = data_prevista - df_atual['ds'].iloc[-1]
    horas+=1
    if horas > 300:
      break
  return previsao, forecast['ds'], forecast['yhat'], delta_time

#Pegando um silo aleatório
i, value = random.choice(list(dataframe_collection.items()))
df_atual = dataframe_collection[i]
plt.figure(figsize = (10,10))
plt.scatter(df_atual.index, df_atual.level)
plt.xlabel("Data")
plt.ylabel("Nível")
plt.show()
ult_previsao, datas,previsoes ,delta_time = predict(50,i)
plt.figure(figsize = (10,10))
plt.plot(datas,previsoes,"r-")
plt.xlabel("Data")
plt.ylabel("Nível")
print(delta_time,i)
