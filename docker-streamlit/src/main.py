import streamlit as st

import math
import random as rd
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import statsmodels.api as sm 
from statsmodels.formula.api import glm

from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split


#plt.rcParams['figure.figsize'] = (13, 7)

df = pd.read_csv('data/StudentsPerformance.csv')


def main():
    pages = {
        'Введение': intro,
        'Подготовка данных': data_processing,
        'Создание модели': model_fit
    }

    st.sidebar.title('Страницы')
    page = st.sidebar.radio('Выберите страницу', tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page]()


def intro():
    """
    Стартовая страница.
    """

    st.title('Что такое логистическая регрессия')

    st.write("""
    Логистическая регрессия или логит-модель (англ. logit model) — это статистическая модель, используемая для прогнозирования вероятности возникновения некоторого события путём его сравнения с логистической кривой. Эта регреcсия выдаёт ответ в виде вероятности бинарного события (1 или 0). 
    """)

    x = np.arange(-6, 6, .1)
    y = (1 + np.exp(-x))**(-1)
    
    fig, ax = plt.subplots()
    plt.plot(x, y)
    st.pyplot(fig)


def data_processing():
    """
    Страница предварительного анализа исходных данных.
    """

    st.title('Подготовка данных')

    st.write(df.head(10))

    fig, ax = plt.subplots(3, 2)
    fig.tight_layout()
    ax[2, 0].axis('off')

    sns.histplot(data=df, x='gender', ax=ax[0, 0])
    sns.histplot(data=df, x='ethnicity', ax=ax[0, 1])
    sns.histplot(data=df, x='parental_level_of_education', ax=ax[1, 0])

    ax[1, 0].xaxis.set_visible(True)
    for tick in ax[1, 0].get_xticklabels():
        tick.set_rotation(90)

    sns.countplot(data=df, x='lunch', ax=ax[1, 1])
    sns.countplot(data=df, x='test_preparation_course', ax=ax[2, 1])
    st.pyplot(fig)

    ax = sns.pairplot(df[['math_score', 'literature_score', 'rus_score']])
    st.pyplot(ax)

    fig = px.scatter_3d(df, x='math_score', y='rus_score', z='literature_score',
                  color='have_no_problem', height=600)
    st.plotly_chart(fig)


def model_fit():
    """
    Создание модели логистической регрессии.
    """

    st.title('Создание модели')

    st.sidebar.header('Фичи:')

    features = np.array(['gender', 'ethnicity', 'lunch', 'test_preparation_course', 'parental_level_of_education'])
    selected_features = features[[st.sidebar.checkbox(f, f) for f in features]]

    X = df[features]
    y = df['have_no_problem']

    code = """
X = df[features]
y = df['have_no_problem']
"""
    st.code(code, language='python')

    st.subheader('Формирование тренировочной и тестовой выборки.')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)

    code = '''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)
    '''
    st.code(code, language='python')

    log_reg = glm(
        f'have_no_problem ~ {" + ".join(selected_features) if len(selected_features) else "1"}',
        data=pd.concat([X_train, y_train], axis=1),
        family=sm.families.Binomial()).fit()

    st.subheader('Создание и обучение модели.')

    code = '''
log_reg = glm(
    f'have_no_problem ~ {" + ".join(features)}',
    data=pd.concat([X_train, y_train], axis=1),
    family=sm.families.Binomial()).fit()

log_reg.summary()
    '''
    st.code(code)

    st.subheader('Сводная таблица.')

    st.code(log_reg.summary())

    st.subheader('Предсказания модели.')

    code = '''
predictions = log_reg.predict(X_test)
    '''
    st.code(code)

    predictions = log_reg.predict(X_test)
    st.code(predictions)

    st.subheader('Выявленные студенты, которым нужны помощь.')

    students_with_problems = X_test
    students_with_problems['predictions'] = predictions
    st.code(students_with_problems[students_with_problems.iloc[:,-1] < .5])

    st.subheader('Метрики качества модели.')
    st.code(classification_report(y_test, round(predictions)))

    st.subheader('ROC кривая.')

    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    st.pyplot(plt)


if __name__ == '__main__':
    main()

