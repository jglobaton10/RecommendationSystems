##Este sistema de reomienda se crea utilizando pandas es content based
#Primera etapa
import pandas as pd


#datos
column_names = ['user_id', 'item_id', 'rating', 'timestamp']

df = pd.read_csv('file.tsv', sep='\t', names=column_names)
#print(df['item_id'].sort_values(ascending = True))
#1 - 1682
eventos = pd.read_csv('Movie_Id_Titles (1).csv')

#Se hace el merge de los datos
mergedData = pd.merge(df, eventos, on='item_id')

#Agrupa por el title y calcula el promedio de la media
ratings = pd.DataFrame(mergedData.groupby('title')['rating'].mean())

#Crea una nueva columna con la sumatoria del atributo rating por pelicula
ratings['num of ratings'] = pd.DataFrame(mergedData.groupby('title')['rating'].count())

matriz = mergedData.pivot_table(index ='user_id', columns ='title', values ='rating')


entrada = input("Ingrese un numero:\n 1. consulta\n 2. Salir")

while entrada == "1":
    nombre = input("Ingrese un nombre")
    userRatings = matriz[nombre]

    # Calcula la correlacion de las demas peliculas con starwats
    similarToEvento = matriz.corrwith(userRatings)
    # Agrega el header de la columna de correlaciones al data frame
    correlaciones = pd.DataFrame(similarToEvento, columns=['Correlation'])

    # Remueve las filas que tienen valores faltantes
    correlaciones.dropna(inplace=True)

    # Une los dos DataFrames ahora el data frame de correlacion tiene las cuentas respecto al numero de rating
    correlaciones = correlaciones.join(ratings['num of ratings'])

    # correlaciones.hist(bins = 90)
    # Se define un criterio de tolerancia
    tol = 0.6

    resTol = correlaciones[correlaciones['Correlation'] > tol].sort_values('Correlation',ascending=False)  # Restriccion tolerancia

    resNRating = resTol[resTol['num of ratings'] > 20]  # Restriccion del rating
    print(resNRating['Correlation'])
    #resNRating['Correlation'].plot().
    entrada = input("Ingrese un numero:\n 1. consulta\n 2. Salir")
# Grafica las correlaciones
print("Salio")

##

