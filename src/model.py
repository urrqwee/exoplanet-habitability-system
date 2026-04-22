import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def prepare_data(df):
    df = df.copy()

    # преобразование целевой переменной
    df['P_HABITABLE'] = df['P_HABITABLE'].isin([1, 2]).astype(int)

    # объединение значений
    df['P_RADIUS_COMBINED'] = df['P_RADIUS'].combine_first(df['P_RADIUS_EST'])

    # логарифмирование
    df['P_RADIUS_LOG'] = np.log1p(df['P_RADIUS_COMBINED'])
    df['P_DISTANCE_LOG'] = np.log1p(df['P_DISTANCE'])
    df['S_HZ_CON_MAX_LOG'] = np.log1p(df['S_HZ_CON_MAX'])

    # удаление пропусков
    df = df.dropna(subset=[
        'P_HABITABLE',
        'P_RADIUS_LOG',
        'P_DISTANCE_LOG',
        'P_TEMP_EQUIL',
        'S_HZ_CON_MAX_LOG'
    ])

    # признаки
    X = df[['P_RADIUS_LOG', 'P_DISTANCE_LOG', 'P_TEMP_EQUIL', 'S_HZ_CON_MAX_LOG']]
    y = df['P_HABITABLE']

    return X, y


def train_model(X, y):
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X, y)
    return model


def prepare_single_object(data_dict):
    df = pd.DataFrame([data_dict])

    df['P_RADIUS_LOG'] = np.log1p(df['P_RADIUS_EST'])
    df['P_DISTANCE_LOG'] = np.log1p(df['P_DISTANCE'])
    df['S_HZ_CON_MAX_LOG'] = np.log1p(df['S_HZ_CON_MAX'])

    X = df[['P_RADIUS_LOG', 'P_DISTANCE_LOG', 'P_TEMP_EQUIL', 'S_HZ_CON_MAX_LOG']]

    return X


def predict(model, X):
    return model.predict_proba(X)[:, 1]