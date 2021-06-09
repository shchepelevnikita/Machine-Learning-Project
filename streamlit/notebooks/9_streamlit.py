import pandas as pd
import streamlit as st
import numpy as np
import builtins
import matplotlib.pyplot as plt
import seaborn as sbn
import os 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def show_title_with_subtitle():
    # Заголовок и подзаголовок
    st.title("Покемоны")
    st.write("# Задача классификации")


def show_info_page():
    st.write("### Задача")
    st.write(
        "Построить, обучить и оценить модель для решения задачи классификации - получения высокоточных предсказаний того, является ли покемон легендарным "
        "по совокупности множества описывающих признаков, влияющих на статус покемона.")
    st.write("### Целевой признак")
    st.write(
        "Legendary")
    st.image("https://i.ytimg.com/vi/yNUYjyzA47o/maxresdefault.jpg",
             use_column_width=True)
    st.write("### Краткое описание входных данных")
    st.write(
        "Данный датасет содержит в себе информацию о 800 покемонах, включая их номер, имя, первый и второй " "тип, а так же некоторые базовые характеристики. Данные характеристики влияют на то, сколько урона " 
        "нанесет покемон в игре."
        "Данные были собраны с нескольких сайтов, включая:"
        "\n"
        "- pokemon.com"
        "\n"
        "- pokemondb.net"
        "\n"
        "- bulbapedia.bulbagarden.net \n"
        "\n"
        "Данные, для которых необходимо получать предсказания, представляют собой подробное признаковое описание покемонов,"
        "включающее в себя такие факторы, как имя, тип покемона, второй тип покемона, сумма всех статистик "
        "покемона, очки здоровья, базовый модификатор атак, базовое сопротивление урону, специальная способность, защита от специальных способностей, скорость, поколение и легендарность.")
    st.write("### Выбранная модель классификации")
    st.write(
        "В результате анализа метрик качества нескольких композиционных моделей классификации выбрана модель "
        "GradientBoostClassifier, обеспечивающая "
        "высокое качество предсказаний легендарности покемонов.")

def show_predictions_page():
    file = st.file_uploader(label="Выберите csv файл с предобработанными данными для прогнозирования статуса покемонов",
                            type=["csv"],
                            accept_multiple_files=False)
    if file is not None:
        data = pd.read_csv(file)
        st.write("### Загруженные данные")
        st.write(data)
        make_predictions(get_model(), data)


def get_model():
    return GradientBoostingClassifier()


def make_predictions(model, data):
    st.write("### Предсказанные значения")
    y = data.Legendary
    data = data.drop(['Legendary'], axis=1)
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 2)
    model.fit(X_train, y_train)
    y_pred = pd.DataFrame(model.predict(X_test))
    y_pred = model.predict(X_test)
    st.write("### Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))
    st.write("### Classification Report")
    st.write(classification_report(y_test, y_pred))


def plot_hist(data):
    fig = plt.figure()
    sbn.histplot(data, legend=False)
    st.pyplot(fig)


def select_page():
    # Сайдбар для смены страницы
    return st.sidebar.selectbox("Выберите страницу", ("Информация", "Прогнозирование"))




def main():
    # Стиль для скрытия со страницы меню и футера streamlit
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # размещение элементов на странице
    show_title_with_subtitle()
    st.sidebar.title("Меню")
    page = select_page()
    
    if page == "Информация":
        show_info_page()
    else:
        show_predictions_page()
        
if __name__ == "__main__":
    main()