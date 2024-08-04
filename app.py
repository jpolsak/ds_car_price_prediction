import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor as RF
import gzip
import requests
from io import BytesIO

#Extracción dataset
df = pd.read_csv('https://raw.githubusercontent.com/jpolsak/datasets/main/car_price_prediction_modif.csv')

# Definición de la función de eliminación de duplicados
def remover_duplicados(df):
  df_sin_duplicados = df.drop('ID',axis=1).drop_duplicates()
  return df_sin_duplicados

# Definición del transformador de eliminación de duplicados
transformador_remover_duplicados = FunctionTransformer(remover_duplicados)

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

# Definición del transformador de eliminación de outliers
transformador_remover_outliers = FunctionTransformer(remover_outliers)

# Definición de la función de eliminación de registros sin datos de Levy
def imputar_nulos(df):
    df_sin_na = df.copy()
    df_sin_na = df_sin_na.dropna()
    return df_sin_na

# Definición de la función de eliminación de la variable Modelo por tener muy alta cardinalidad y la variable ID
def remover_var(df):
  df_remover_var = df.drop(['Model','ID'],axis=1,errors='ignore')
  return df_remover_var

# Definición del transformador de eliminación de variables
transformador_remover_variables = FunctionTransformer(remover_var)

# Definición de la función de eliminación de registros sin datos de Levy
def imputar_nulos(df):
  df_sin_na = df.copy()
  df_sin_na = df_sin_na.dropna()
  return df_sin_na

# Definición del transformador de imputar nulos
transformador_imputación = FunctionTransformer(imputar_nulos)

# Definición de la función de separación x-y
def sep_x_y(df):
  x_inicial = df.drop('Price',axis=1)
  y = df.Price
  return x_inicial,y

# Definición del transformador para el encoding de variables categóricas y Standard Scaler de variables numéricas
transformador_enc_sc = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), make_column_selector(dtype_include=np.number)),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), make_column_selector(dtype_include=object))
    ]
)

# Definición del pipeline de preprocesamiento para encoding
pipeline_preprocesamiento_2 = Pipeline(steps=[
    ('encoding', transformador_enc_sc)
])

# Creación del pipeline
pipeline_preprocesamiento_1 = Pipeline(steps=[('Remover duplicados',transformador_remover_duplicados),
                                            ('Eliminación de outliers',transformador_remover_outliers),
                                            ('Eliminación de variables',transformador_remover_variables),
                                            ('Imputación de Levy',transformador_imputación),
                                            ('Separación x-y',transformador_sep_x_y)])

pipeline_preprocesamiento_2 = Pipeline(steps=[('encoding',transformador_encoding)])

# Preprocesamiento
x_inicial,y = pipeline_preprocesamiento_1.fit_transform(df)
x = pipeline_preprocesamiento_2.fit_transform(x_inicial)
x = x.reset_index(drop=True)
y = y.reset_index(drop=True)
x_train_f, x_test_f, y_train_f, y_test_f = train_test_split(x,y,random_state=42)

#Modelo
model = LGBMRegressor(
    n_estimators=100,      # Número de árboles
    learning_rate=0.1,     # Tasa de aprendizaje
    num_leaves=31,         # Número de hojas en cada árbol
    max_depth=-1,          # Profundidad máxima del árbol, -1 para sin límite
    random_state=42        # Semilla para reproducibilidad
)
model.fit(x_train_f, y_train_f)

# Título de la aplicación
st.title('Data Science - Modelo de Machine Learning para la predicción del precio de un auto en función de sus características 🚗')

# Formulario para la entrada de datos
st.header('Introduzca los datos del vehículo')

Levy = st.number_input('Levy', min_value=0.0)
Manufacturer = st.selectbox('Manufacturer', ['ACURA', 'ALFA ROMEO', 'ASTON MARTIN', 'AUDI', 'BENTLEY', 'BMW', 'BUICK', 'CADILLAC', 'CHEVROLET', 'CHRYSLER', 'CITROEN', 'DAEWOO', 'DAIHATSU', 'DODGE', 'FERRARI', 'FIAT', 'FORD', 'GAZ', 'GMC', 'GREATWALL', 'HAVAL', 'HONDA', 'HUMMER', 'HYUNDAI', 'INFINITI', 'ISUZU', 'JAGUAR', 'JEEP', 'KIA', 'LAMBORGHINI', 'LANCIA', 'LAND ROVER', 'LEXUS', 'LINCOLN', 'MASERATI', 'MAZDA', 'MERCEDES-BENZ', 'MERCURY', 'MINI', 'MITSUBISHI', 'MOSKVICH', 'NISSAN', 'OPEL', 'PEUGEOT', 'PONTIAC', 'PORSCHE', 'RENAULT', 'ROLLS-ROYCE', 'ROVER', 'SAAB', 'SATURN', 'SCION', 'SEAT', 'SKODA', 'SSANGYONG', 'SUBARU', 'SUZUKI', 'TESLA', 'TOYOTA', 'UAZ', 'VAZ', 'VOLKSWAGEN', 'VOLVO', 'ZAZ', 'სხვა'])
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
    # Preprocesar los datos de entrada usando el pipeline completo
    input_data = input_data.drop(columns=['Model', 'ID'], errors='ignore')
    x_input_preprocesado = pipeline_preprocesamiento_2.transform(input_data)
    
    # Realizar la predicción
    prediction = model.predict(x_input_preprocesado)
    
    # Obtener el valor de la predicción y formatearlo
    precio_predicho = prediction[0]
    st.write(f'El precio predicho es: ${precio_predicho:.2f}')
    # Mostrar los datos preprocesados (útil para depuración)
    st.write('Datos preprocesados:')
    st.write(pd.DataFrame(x_input_preprocesado, columns=pipeline_preprocesamiento_2.named_steps['encoding'].get_feature_names_out()))
