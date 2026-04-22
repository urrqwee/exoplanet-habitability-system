import pandas as pd
import joblib
from src.model import prepare_data, train_model

# загрузка данных
df = pd.read_csv("data/phl_exoplanet_catalog_2019.csv")

# подготовка
X, y = prepare_data(df)

# обучение
model = train_model(X, y)

# сохранение
joblib.dump(model, "model/rf_model.pkl")

print("Модель обучена и сохранена")