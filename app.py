import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Definición de la función de eliminación de duplicados
def remover_duplicados(df):
    df_sin_duplicados = df.drop('ID', axis=1).drop_duplicates()
    return df_sin_duplicados

# Definición de la función de eliminación de outliers sobre la variable precio con el método de IQR
def remover_outliers(df):
    Q1 = np.percentile(df['Price'], 25)
    Q3 = np.percentile(df['Price'], 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = (df['Price'] < lower) | (df['Price'] > upper)
    df_sin_outliers = df[~mask]
    return df_sin_outliers

# Definición de la función de eliminación de registros sin datos de Levy
def imputar_nulos(df):
    df_sin_na = df.copy()
    df_sin_na = df_sin_na.dropna()
    return df_sin_na

# Definición de la función de eliminación de la variable Modelo por tener muy alta cardinalidad y la variable ID
def remover_var(df):
    df_remover_var = df.drop(['Model', 'ID'], axis=1, errors='ignore')
    return df_remover_var

# Definición de la función de separación x-y
def sep_x_y(df):
    x_inicial = df.drop('Price', axis=1)
    y = df.Price
    return x_inicial, y
# Cargar el modelo
model = joblib.load('pipeline_model.pkl')

# Título de la aplicación
st.title('Data Science - Modelo de Machine Learning para la predicción del precio de un auto en función de sus características')

# Formulario para la entrada de datos
st.header('Introduzca los datos del vehículo')

ID = st.number_input('ID', value=0)
Levy = st.number_input('Levy', min_value=0.0)
Manufacturer = st.selectbox('Manufacturer', ['ACURA', 'ALFA ROMEO', 'ASTON MARTIN', 'AUDI', 'BENTLEY', 'BMW', 'BUICK', 'CADILLAC', 'CHEVROLET', 'CHRYSLER', 'CITROEN', 'DAEWOO', 'DAIHATSU', 'DODGE', 'FERRARI', 'FIAT', 'FORD', 'GAZ', 'GMC', 'GREATWALL', 'HAVAL', 'HONDA', 'HUMMER', 'HYUNDAI', 'INFINITI', 'ISUZU', 'JAGUAR', 'JEEP', 'KIA', 'LAMBORGHINI', 'LANCIA', 'LAND ROVER', 'LEXUS', 'LINCOLN', 'MASERATI', 'MAZDA', 'MERCEDES-BENZ', 'MERCURY', 'MINI', 'MITSUBISHI', 'MOSKVICH', 'NISSAN', 'OPEL', 'PEUGEOT', 'PONTIAC', 'PORSCHE', 'RENAULT', 'ROLLS-ROYCE', 'ROVER', 'SAAB', 'SATURN', 'SCION', 'SEAT', 'SKODA', 'SSANGYONG', 'SUBARU', 'SUZUKI', 'TESLA', 'TOYOTA', 'UAZ', 'VAZ', 'VOLKSWAGEN', 'VOLVO', 'ZAZ', 'სხვა'])
Model = st.text_input('Model', value='0')
Prod_year = st.number_input('Prod. year', min_value=1939, max_value=2020, step=1)
Category = st.selectbox('Category', ['Cabriolet', 'Coupe', 'Goods wagon', 'Hatchback', 'Jeep', 'Limousine', 'Microbus', 'Minivan', 'Pickup', 'Sedan', 'Universal'])
Leather_interior = st.selectbox('Leather interior', ['Yes', 'No'])
Turbo = st.selectbox('Turbo', ['yes', 'no'])
Fuel_type = st.selectbox('Fuel type', ['CNG', 'Diesel', 'Hybrid', 'Hydrogen', 'LPG', 'Petrol', 'Plug-in Hybrid'])
Engine_volume = st.number_input('Engine volume', min_value=0.0, max_value=20.0)
Mileage_km = st.number_input('Mileage_km', min_value=0.0)
Cylinders = st.number_input('Cylinders', min_value=1, max_value=16, step=1)
Gear_box_type = st.selectbox('Gear box type', ['Automatic', 'Manual', 'Tiptronic', 'Variator'])
Drive_wheels = st.selectbox('Drive wheels', ['4x4', 'Front', 'Rear'])
Doors = st.selectbox('Doors', ['2', '4', '>5'])
Wheel = st.selectbox('Wheel', ['Left wheel', 'Right-hand drive'])
Color = st.selectbox('Color', ['Beige', 'Black', 'Blue', 'Brown', 'Carnelian red', 'Golden', 'Green', 'Grey', 'Orange', 'Pink', 'Purple', 'Red', 'Silver', 'Sky blue', 'White', 'Yellow'])
Airbags = st.number_input('Airbags', min_value=0, max_value=16, step=1)

# Valor de Price y variables a eliminar en preprocesamiento predeterminados
Price = 0
ID = 0
Model = 0

# Preprocesamiento de datos
input_data = pd.DataFrame({
    'ID': [0],  # ID es eliminada en preprocesamiento, así que puede ser 0
    'Levy': [Levy],
    'Manufacturer': [Manufacturer],
    'Model': [0],  # Model es eliminada en preprocesamiento, así que puede ser 0
    'Prod_year': [Prod_year],
    'Category': [Category],
    'Leather_interior': [Leather_interior],
    'Turbo': [Turbo],
    'Fuel_type': [Fuel_type],
    'Engine_volume': [Engine_volume],
    'Mileage': [Mileage_km],
    'Cylinders': [Cylinders],
    'Gear_box_type': [Gear_box_type],
    'Drive_wheels': [Drive_wheels],
    'Doors': [Doors],
    'Wheel': [Wheel],
    'Color': [Color],
    'Airbags': [Airbags],
    'Price': [Price]  # Incluimos Price como 0 para completar las columnas necesarias
})

# Realizar la predicción
if st.button('Predecir'):
    prediction = model.predict(input_data)
    st.write(f'La predicción del modelo es: {prediction}')
