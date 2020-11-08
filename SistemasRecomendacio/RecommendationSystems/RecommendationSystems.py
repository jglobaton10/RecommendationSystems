##
import pandas as pd

def recomendar(id):
    reviews = pd.read_csv(
        'https://raw.githubusercontent.com/jglobaton10/Databases-repository/master/Data%20pipeline/ReviewsData.csv')
    activities = pd.read_csv(
        'https://media.githubusercontent.com/media/jglobaton10/Databases-repository/master/Data%20pipeline/Activities%20final.csv')

    data = pd.merge(reviews, activities, left_on='id_activity', right_on='id')
    data.drop('id', axis=1, inplace=True)

    ratings = pd.DataFrame(data.groupby(by='id_activity').mean()['rating '])
    ratings['num of ratings'] = pd.DataFrame(data.groupby('id_activity')['rating '].count())

    matriz = data.pivot_table(index='id_user', columns='id_activity', values='rating ')

    nombre = id  # int(input("Ingrese un nombre"))

    userRatings = matriz[nombre]
    similarToEvento = matriz.corrwith(userRatings)
    # Agrega el header de la columna de correlaciones al data frame
    correlaciones = pd.DataFrame(similarToEvento, columns=['Correlation'])
    correlaciones.dropna(inplace=True)

    # Une los dos DataFrames ahora el data frame de correlacion tiene las cuentas respecto al numero de rating
    correlaciones = correlaciones.join(ratings['num of ratings'])
    # correlaciones.hist(bins = 90)
    # Se define un criterio de tolerancia
    tol = 0.6
    resTol = correlaciones[correlaciones['Correlation'] > tol].sort_values('Correlation', ascending=False)

    resNRating = resTol[resTol['num of ratings'] > 20]  # Restriccion del rating
    # print(resNRating['Correlation'])

    final = pd.merge(resNRating, activities, left_index=True, right_on='id')
    final.drop(['Correlation', 'num of ratings'], axis=1, inplace=True)
    return final


#final.columns


from flask import Flask, request, jsonify
from flask_marshmallow import Marshmallow
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)
ma = Marshmallow(app)

class UserSchema(ma.Schema):
    class Meta:
        fields = ('id', 'Name', 'Description', 'Location', 'Address', 'Date', 'Hour','Image', 'Category', 'Cost', 'Organizer', 'Grupal')

user_schema = UserSchema()
users_schema = UserSchema(many=True)
import json
class UserManager(Resource):
    pass
    @staticmethod
    def get():
        respuesta = None
        try:
            id_activity = int(request.args['id'])

            final = recomendar(id_activity)
            result = final.to_json(orient="records")
            parsed = json.loads(result)
            json.dumps(parsed, indent=4)
            respuesta = parsed

        except Exception as _:
            respuesta  = None

        return   respuesta  #jsonify(user_schema.dump(final))

api.add_resource(UserManager, '/api/recommendations')
if __name__ == '__main__':
        app.run(debug=True)

