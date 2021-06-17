import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from joblib import dump, load
from sklearn import ensemble
from sklearn import metrics
import numpy as np


def main():
  GrBosstClass = load_GRB()
  Bagging = load_Bagging()
  data = loadSet()
  defaulData = loadDefaultSet()
  currentModelSelected = 0
  vl15H = 0
  speedWind = 0
  cloydy3H = 0
  vl9H = 0
  rainToday = 1
  sun = 0
  tempAir15H = 0
  dav15H = 0
  minTemp = 0
  maxTemp = 0
  
  st.title("Модуль С")
  st.markdown("<style> .big-font {font-size:30px !important; }</style>", unsafe_allow_html=True)
  st.sidebar.markdown('<p class="big-font"> <b>Модели</b> </p>', unsafe_allow_html=True)
  modelSelect = st.sidebar.radio("Выберите модель машинного обучения", ["Градиентный бустинг", "Бэггиннг"])

  st.sidebar.markdown('<p class="big-font"> <b>Дата сеты</b> </p>', unsafe_allow_html=True)
  dataSetSelect = st.sidebar.radio("Выберите дата сет", ["Исходный", "После препроцесса"])

  if dataSetSelect == "После препроцесса":
    '''В датасете после препроцесса остались только самые важные десять полей. Также перекодированы столбцы: локация, направление ветра и был ли сегодня дождь.'''
    st.write(data.head(10))

  if dataSetSelect == "Исходный":
    ''' Исходный датасет имеет полный список полей. Также в нём не заполнены пустые значения и ни один признак не перекодирован.'''
    st.write(defaulData.head(10))
      

  if modelSelect == "Градиентный бустинг":
    currentModelSelected = GrBosstClass
    "Предсказание градиентного бустинга"
    "Градиентный бустинг предсказывает с точностью в 85%"


  if modelSelect == "Бэггиннг":
    currentModelSelected = Bagging
    "Предсказание бэггиннга"
    "Бэггинг предсказывает с точностью в 84%"
  "Для предсказания дождя на завтрашний день исопльзуются основные 10 параметров."
  ""  
  inpType = st.radio("Как будете вводить данные?", ['Строкой', "Буду заполнять каждое поле отдельно!"])
  if inpType == "Буду заполнять каждое поле отдельно!":
    vl15H = st.number_input("Влажность воздуха в 15:00")
    speedWind = st.number_input("Скорость порыва ветра")
    cloydy3H = st.number_input("Облачность в 15:00 (от 0 до 9)")
    vl9H = st.number_input("Влажность в 9 утра")
    rainTodayStr = st.radio("Был ли сегодня дождь?", ["Да" , "Нет"])
    rainToday = 1
    if rainTodayStr =="Да":
      rainToday = 1
    else:
      rainToday = 0
    sun = st.number_input("Солнечный свет (от 0 до 13,9)")
    tempAir15H = st.number_input("Температура воздуха в 15:00")
    dav15H = st.number_input("Давление в 15:00")
    minTemp = st.number_input("Минимальная температура воздуха")
    maxTemp = st.number_input("Максимальная температура воздуха")
  if inpType == "Строкой":
    a = st.text_input('Ввести данные строкой:', help= "Ввдетие числа через запятую. Без точек и запятых в конце строки!")
    if a:
      a = a.split(',')
      a = [float(i) for i in a]
      polya = a
  else:
    polya = int(currentModelSelected.predict([[vl15H,speedWind,cloydy3H,vl9H,rainToday,sun,tempAir15H,dav15H,minTemp,maxTemp]]))
  "Введенные данные: " + str(vl15H) + ", " + str(speedWind) + ", " + str(cloydy3H) + ", " + str(vl9H) + ", " +str(rainToday) + ", " + str(sun) + ", " + str(tempAir15H) + ", " + str(dav15H) + ", " + str(minTemp) + ", " + str(maxTemp) + "."
  if st.button("Рассчитать прогноз", key=None, help="Предсказать будет ли дождь завтра при помощи метода " + modelSelect):
    res = polya 
    if res == 0:
      st.markdown("<p style=\"font-size:30px\">Скорее всего завтра не будет дождя &#128516;</p>", unsafe_allow_html=True)
    else:
      st.markdown("<p style=\"font-size:30px\">Скорее всего завтра будет дождь... &#128532;</p>", unsafe_allow_html=True)
  
@st.cache
def loadSet():
  df = pd.read_csv("data/watherPreProcces.csv")
  return df

### models

@st.cache
def load_GRB():
  with open('models/GrBosstClass.pkl', 'rb') as pkl_file:
    grb = pickle.load(pkl_file)
  return grb


@st.cache(allow_output_mutation=True)
def load_Bagging():
  with open('models/Bagging.pkl', 'rb') as pkl_file:
    Bagging = pickle.load(pkl_file)
  return Bagging

@st.cache
def loadDefaultSet():
  df = pd.read_csv("data/weather.csv")
  return df

if __name__ == "__main__":
  main()
