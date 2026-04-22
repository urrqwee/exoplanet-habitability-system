import joblib
from src.model import prepare_single_object

# загрузка модели
model = joblib.load("model/rf_model.pkl")

print("Введите параметры планеты:")

radius = float(input("Радиус планеты (в радиусах Земли): "))
distance = float(input("Расстояние до звезды (в AU, Земля = 1): "))
hz = float(input("Внешняя граница обитаемой зоны (в AU): "))
temp = float(input("Температура планеты (в Кельвинах, Земля ≈ 288): "))

new_planet = {
    'P_RADIUS_EST': radius,
    'P_DISTANCE': distance,
    'S_HZ_CON_MAX': hz,
    'P_TEMP_EQUIL': temp
}

X_new = prepare_single_object(new_planet)

prob = model.predict_proba(X_new)[0][1]

print("\nРезультат:")
print("Вероятность обитаемости:", round(prob, 4))