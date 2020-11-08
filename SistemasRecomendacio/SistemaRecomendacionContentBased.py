##
import numpy as np
import scipy
import pandas as pd
import math
import sklearn
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



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
print(tfidf_feature_names)

##Evaluacion del modelo
#SE haca cross validation con 20%
interactions_train_df, interactions_test_df = train_test_split(dataSetInteracciones, stratify=dataSetInteracciones['personId'], test_size=0.20, random_state=42)
#print(interactions_train_df.head())
##Diseño del perfil de usuario
#Obtiene el perfil de los items
def getPerfilObjeto(item_id):
    idx = item_ids.index(item_id)
    item_profile = tfidf_matrix[idx:idx + 1]
    return item_profile

#Lista de perfiles de los items
def getPerfilesObjetos(ids):
    item_profiles_list = [getPerfilObjeto(x) for x in ids]
    item_profiles = scipy.sparse.vstack(item_profiles_list)
    return item_profiles

#PErfil de un usuario
def getUserProfile(person_id, interactions_indexed_df):
    interactions_person_df = interactions_indexed_df.loc[person_id]
    user_item_profiles = getPerfilesObjetos(interactions_person_df['contentId'])
    user_item_strengths = np.array(interactions_person_df['eventStrength']).reshape(-1, 1)
    # Weighted average of item profiles by the interactions strength
    user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(
        user_item_strengths)
    user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
    return user_profile_norm

#Perfil de todos los usuarios
def getUsersProfiles():
    interactions_indexed_df = interactions_train_df[interactions_train_df['contentId'] \
        .isin(articulos['contentId'])].set_index('personId')
    user_profiles = {}
    for person_id in interactions_indexed_df.index.unique():
        user_profiles[person_id] = getUserProfile(person_id, interactions_indexed_df)
    return user_profiles

user_profiles = getUsersProfiles()
print(len(user_profiles))
##
class RecomendadorBasadoContenido:
    Modelo = 'Basado_Contenido'

    def __init__(self, items_df=None):
        self.item_ids = item_ids
        self.items_df = items_df

    def get_model_name(self):
        return self.Modelo

    def itemsParecidosParaUsuario(self, person_id, topn=1000):
        # Similitud entre el perfil de usuario y los demas perfiles de usuarios
        cosine_similarities = cosine_similarity(user_profiles[person_id], tfidf_matrix)
        # Obtiene los items mas similares
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        # Ordena los items de acuerdo al indice de similitud
        similar_items = sorted([(item_ids[i], cosine_similarities[0, i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items

    def ItemsRecomendados(self, idUsuario, ignorar=[], topn=10, verbose=False):
        similar_items = self.itemsParecidosParaUsuario(idUsuario)


        similar_items_filtered = list(filter(lambda x: x[0] not in ignorar, similar_items))

        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['contentId', 'recStrength']) \
            .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='contentId',
                                                          right_on='contentId')[
                ['recStrength', 'contentId', 'title', 'url', 'lang']]

        return recommendations_df


modelo = RecomendadorBasadoContenido(articulos)
print(modelo.ItemsRecomendados(-1479311724257856983))

##Prueba

"""
##Metricas de modelo
cb_global_metrics, cb_detailed_results_df = evaluate_model(content_based_recommender_model)
print(cb_global_metrics)
print(cb_detailed_results_df)
"""


