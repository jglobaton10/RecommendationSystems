##
import numpy as np
import pandas as pd
import math
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds

#Data set de articulos compartidos
articulos = pd.read_csv("shared_articlesR2.csv")
#Filtrar por contenido compartido
articulos = articulos[articulos['eventType'] == 'CONTENT SHARED']
#print(articles_df.head())

#Data set  interacciones de usuarios
interacciones = pd.read_csv("users_interactionsR2.csv")
#print(interactions_df)

#Darle peso a las interacciones
valoresEventos = {'VIEW': 1.0, 'LIKE': 2.0, 'BOOKMARK': 2.5, 'FOLLOW': 3.0, 'COMMENT CREATED': 4.0}

#Se crea la columna que representa la fuerza que se le da a las interacciones a traves de una funcion lambda
interacciones['eventStrength'] = interacciones['eventType'].apply(lambda x: valoresEventos[x])

#Cuenta el numero de interacciones y lo agrupa por el id de las personas
cuentaInteraccionesUsuarios = interacciones.groupby(['personId', 'contentId']).size().groupby('personId').size()
#Filtra los usuarios con menos de 5 interacciones con el fin de evitar el problema de ser cold start
users_with_enough_interactions_df = cuentaInteraccionesUsuarios[cuentaInteraccionesUsuarios >= 5].reset_index()[['personId']]
#Combina los dos data set por el atributo personId. Hace un rigrht join
interactions_from_selected_users_df = interacciones.merge(users_with_enough_interactions_df, how ='right', left_on ='personId', right_on ='personId')

def smooth_user_preference(x):
    return math.log(1+x, 2)
dataSetInteracciones = interactions_from_selected_users_df.groupby(['personId', 'contentId'])['eventStrength'].sum().apply(smooth_user_preference).reset_index()

#Content base
#Genera la lista de stopwords en ingles y portugues
#b = nlt.download('stopwords') #Se descarga el paquete stopwords
stopwords_list = stopwords.words('english') + stopwords.words('portuguese')

#Genera un modelo de vectorizacion
ModeloVectorizacion = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.003, max_df=0.5, max_features=5000, stop_words=stopwords_list)
##
#Convierte en lista los elementos
item_ids = articulos['contentId'].tolist()
#Le hace fitting al modelo de vectorizacion con  las columnas de titulo y articulo de acuerdo a las palabras pone sus valores
#de aparicion en la matriz
tfidf_matrix = ModeloVectorizacion.fit_transform(articulos['title'] + "" + articulos['text'])
tfidf_feature_names = ModeloVectorizacion.get_feature_names()
#print(tfidf_feature_names)

##Evaluacion del modelo
#SE haca cross validation con 20%
interactions_train_df, interactions_test_df = train_test_split(dataSetInteracciones, stratify=dataSetInteracciones['personId'], test_size=0.20, random_state=42)
#print(interactions_train_df.head())

##Creacion de la matriz dispersa
matrizPivotesItemsDeUsuario = interactions_train_df.pivot(index='personId',
                                                          columns='contentId',
                                                          values='eventStrength').fillna(0)
#print(users_items_pivot_matrix_df.head(10))
#Convierte la matriz de pivotes en una matriz
MatrizPivotesItemsUsuarios = matrizPivotesItemsDeUsuario.values

#Saca una lista de los ids de los usuarios
ListaUsuarios = list(matrizPivotesItemsDeUsuario.index)
#print(users_ids)
#print(users_items_pivot_matrix)

#Comprime la matriz a una matriz donde solo hay valores distintos de 0
MatrizComprimida = csr_matrix(MatrizPivotesItemsUsuarios)
#print(MatrizComprimida)
##
#Valores a computar
NUMBER_OF_FACTORS_MF = 15
#Performs matrix factorization of the original user item matrix
#U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)
U, sigma, Vt = svds(MatrizComprimida, k = NUMBER_OF_FACTORS_MF)

sigma = np.diag(sigma)
#Hace el producto punto dos veces
RatingsPrediccion = np.dot(np.dot(U, sigma), Vt)
#Normaliza los valores
RatingsPrediccionesNormalizados = (RatingsPrediccion - RatingsPrediccion.min()) / (RatingsPrediccion.max() - RatingsPrediccion.min())
##
#Matriz a dataframe la matriz tiene informacion respecto  a cada usuario y su preferencia en items
matrizPreferencias = pd.DataFrame(RatingsPrediccionesNormalizados, columns = matrizPivotesItemsDeUsuario.columns, index=ListaUsuarios).transpose()
#print(matrizPreferencias.head(10))
##Recomendador Colaborativo
class FiltradoColaborativo:
    NOMBRE = 'Collaborative Filtering'


    #Constructor
    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df

    def get_model_name(self):
        return self.NOMBRE

    def ItemsRecomendados(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Organiza la lista de items en orden descendente
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
            .reset_index().rename(columns={user_id: 'recStrength'})

        # Recomidna los que tienen el recStrength y que no estan en el  la lista de items a ignorar
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['contentId'].isin(items_to_ignore)] \
            .sort_values('recStrength', ascending=False) \
            .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='contentId',
                                                          right_on='contentId')[
                ['recStrength', 'contentId', 'title', 'url', 'lang']]

        return recommendations_df


modelo = FiltradoColaborativo(matrizPreferencias, articulos)
print(modelo.ItemsRecomendados(-1479311724257856983))
##

