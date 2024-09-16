import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

# Etapa 1. Obtención de datos
tree_prices = ET.parse('prices.xml')
root_prices = tree_prices.getroot()

tree_places = ET.parse('places.xml')
root_places = tree_places.getroot()

precio_data = []
for place in root_prices.findall('place'):
    place_id = place.get('place_id')
    for gas_price in place.findall('gas_price'):
        tipo_combustible = gas_price.get('type')
        precio = gas_price.text
        precio_data.append({
            'place_id': place_id,
            'fuel_type': tipo_combustible,
            'gas_price': float(precio)
        })

gasolinera_data = []
for place in root_places.findall('place'):
    place_id = place.get('place_id')
    nombre = place.find('name').text
    cre_id = place.find('cre_id').text
    location = place.find('location')
    x = location.find('x').text
    y = location.find('y').text
    gasolinera_data.append({
        'place_id': place_id,
        'name': nombre,
        'cre_id': cre_id,
        'x': float(x),
        'y': float(y)
    })

df_precios = pd.DataFrame(precio_data)
print(df_precios.head())

df_gasolineras = pd.DataFrame(gasolinera_data)
print(df_gasolineras.head())

# Etapa 2. Preparación de los datos

df_precios = df_precios[df_precios['gas_price'] >= 10]

df_gasolineras = df_gasolineras[(df_gasolineras['x'] >= -119) & (df_gasolineras['x'] <= -86) &
                                 (df_gasolineras['y'] >= 14) & (df_gasolineras['y'] <= 33)]

df_merged = pd.merge(df_precios, df_gasolineras, on='place_id', how='inner')

shp_estados = gpd.read_file('dest2019gw/dest2019gw.shp')

gdf_gasolineras = gpd.GeoDataFrame(df_merged, geometry=gpd.points_from_xy(df_merged.x, df_merged.y))
gdf_gasolineras = gdf_gasolineras.set_crs("EPSG:4326")
gdf_gasolineras = gpd.sjoin(gdf_gasolineras, shp_estados, how='left', predicate='within')

df_gasolineras = pd.DataFrame(gdf_gasolineras)

# Etapa 3. Exploración y visualización de los datos

grouped = df_precios.groupby('fuel_type')['gas_price']

print("\nAnálisis con Estadísticas descriptivas por tipo de gasolina:")
print(grouped.describe())


plt.figure(figsize=(10, 6))
sns.histplot(data=df_merged, x='gas_price', hue='fuel_type', kde=True, bins=30)
plt.xlabel('Precio de Gasolina')
plt.ylabel('Frecuencia')
plt.title('Distribución de precios por tipo de gasolina')
plt.show()


plt.figure(figsize=(10, 8))
sc = plt.scatter(df_merged['x'], df_merged['y'], c=df_merged['gas_price'], cmap='viridis', s=100)
plt.colorbar(sc, label='Precio de gasolina (MXN)')
plt.xlabel('Longitud (x)')
plt.ylabel('Latitud (y)')
plt.title('Distribución de precios de gasolina por ubicación')
plt.show()

# Etapa 4. Evaluación 

for fuel_type, group in grouped:
    min_price = group.min()
    max_price = group.max()

    print(f"\nTipo de gasolina: {fuel_type}")
    print(f"Precio mínimo: {min_price}")
    print(f"Precio máximo: {max_price}")

    place_id_min = df_precios[(df_precios['fuel_type'] == fuel_type) & (df_precios['gas_price'] == min_price)]['place_id'].values[0]
    place_id_max = df_precios[(df_precios['fuel_type'] == fuel_type) & (df_precios['gas_price'] == max_price)]['place_id'].values[0]

    place_min = df_gasolineras[df_gasolineras['place_id'] == place_id_min]
    print(f"\nGasolinera con el precio mínimo de {fuel_type.upper()}:")
    print(place_min['name'])

    place_max = df_gasolineras[df_gasolineras['place_id'] == place_id_max]
    print(f"\nGasolinera con el precio máximo de {fuel_type.upper()}:")
    print(place_max['name'])


precio_promedio = df_merged.groupby('fuel_type')['gas_price'].mean()
print("\nPrecio promedio de gasolina:")
print(precio_promedio)

promedio_precio_estado = gdf_gasolineras.groupby(['NOM_ENT', 'fuel_type'])['gas_price'].mean().reset_index()
promedio_precio_estado = promedio_precio_estado.rename(columns={'gas_price': 'precio_promedio'})
print("\nEstados con sus precios promedio de gasolina:")
print(promedio_precio_estado)

top_precios = df_gasolineras.groupby('fuel_type').apply(lambda x: x.nlargest(100, 'gas_price'))
top_precios = top_precios[['gas_price', 'NOM_ENT']]

conteo_estados_altos = top_precios['NOM_ENT'].value_counts()
top_precios = top_precios.groupby(['NOM_ENT', 'fuel_type'])['gas_price'].mean().reset_index()
top_precios['frecuencia'] = top_precios['NOM_ENT'].map(conteo_estados_altos)

top_precios = pd.merge(top_precios, promedio_precio_estado, on=['NOM_ENT', 'fuel_type'], how='left')

top_precios['diferencia'] = top_precios['gas_price'] - top_precios['precio_promedio']
top_precios['variacion'] = top_precios['diferencia'] / top_precios['precio_promedio'] * 100

top_precios = top_precios.rename(columns={'NOM_ENT': 'estado', 
                                                    'fuel_type': 'tipo_combustible',
                                                    'gas_price': 'precio'})
top_precios = top_precios[['estado', 'frecuencia', 'tipo_combustible', 'precio', 'precio_promedio', 'diferencia', 'variacion']]
top_precios = top_precios.sort_values('frecuencia', ascending=False)

print("\nTop de estados con los precios más altos de gasolina:")
print(top_precios)