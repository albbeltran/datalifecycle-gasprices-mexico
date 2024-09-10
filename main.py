import pandas as pd
import numpy as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Etapa 1 & 2 - Recolección y Almacenamiento de Datos

# Usamos un conjunto de datos de ejemplo de scikit-learn
from sklearn.datasets import load_boston
boston = load_boston()

# Convertimos los datos a un DataFrame de pandas
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['MEDV'] = boston.target


# Etapa 3 - Procesamiento de datos

# En esta etapa se inspeccionan y limpian los datos
print('Primeras filas del conjunto de datos:')
print(data.head())

# Verificamos si hay valores nulos
print('\n¿Hay valores nulos?:')
print(data.isnull().sum())


# Etapa 4 - Análisis Exploratorio de Datos (EDA)

# Estadísticas descriptivas 
print('\nEstadísticas descriptivas:')
print(data.describe())

# Visualización de la distribución de la variable objetivo
plt.figure(figsize=(8,6))
sns.histplot(data['MEDV'], bins=30, kde=True)
plt.title('Distribución de los precios de las casas (MEDV)')
plt.xlabel('Precio (MEDV)')
plt.ylabel('Frecuencia')
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Mapa de calor de correlación')
plt.show()


# Etapa 4.1 - Preparación de los datos
X = data.drop('MEDV', axis=1)
y = data['MEDV']

# División de los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado de las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4.2. Modelado
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 4.3. Evaluación
# Predicciones en el conjunto de prueba
y_pred = model.predict(X_test_scaled)

# Métrica de evaluación: Error cuadrático medio (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'\nError cuadrático medio (MSE) en el conjunto de prueva:')

# 4.4 Visualización de resultados
# Comparación entre valores reales y predichos
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k',alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Valores reales')
plt.ylabel('Valores predichos')
plt.title('Comparación de valores reales vs predichos')
plt.show()