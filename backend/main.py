from flask import Flask, render_template, request, jsonify
import os
import importlib
import sys
import pandas as pd
import json
import csv
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from functools import reduce

engine = create_engine('sqlite:///../bdd/sql_database/ml_processed.db')

# engine.execute('alter table reviews add column topics_str varchar(30)')


app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///../bdd/sql_database/ml_processed.db"
db = SQLAlchemy(app)
db.Model.metadata.reflect(db.engine)


class Reviews(db.Model):
    __tablename__ = 'reviews'
    __table_args__ = {'extend_existing': True}


sys.path.append('../MachineLearning_processing')

# DATA_PROCESSING =  importlib.import_module('src.processing.process')
ELASTIC = importlib.import_module('elasticsearch_browser')
call_sql = importlib.import_module('process_sqldata')
dashboard = call_sql.dashboard_queries()
# import lstm_live_predict as KERAS_ML
# import transformers_live_prediction as TRANSFORMERS


# @app.route("/")
# def hello():
#     return render_template('chat.html')


@app.route("/")
def index():
    return render_template('index.html')


ROWS_PER_PAGE = 15


@app.route('/processed_reviews')
def processed_reviews():
    reviews = Reviews.query.all()
    targets = ['ALLERGENS', 'COMPETITION',
               'DELIVERY', 'PACKAGING', 'PRICE', 'TASTE']
    for review in reviews:
        if review.topic:
            review.topics_str = reduce(
                lambda x, y: x + ', ' + y, [targets[int(topic)-1] for topic in review.topic.split(',')])
        else:
            review.topics_str = "No topic related"
    db.session.flush()
    db.session.commit()
    page = request.args.get('page', 1, type=int)
    processed_reviews = Reviews.query.paginate(
        page=page, per_page=ROWS_PER_PAGE)
    return render_template('processed_reviews.html', processed_reviews=processed_reviews)


@app.route("/queries")
def queries():
    return render_template('queries.html')


@app.route("/prediction")
def prediction():
    return render_template('prediction.html')


@app.route('/queries/results_queries', methods=['GET', 'POST'])
def search_request():
    search_term = request.form["input"]
    elastic_request = ELASTIC.elastic_queries(10)
    res = elastic_request.basic_query(search_term)
    return render_template('results_queries.html', res=res)


@app.route('/prediction/results_prediction', methods=['GET', 'POST'])
def prediction_request():
    review = request.form["input"]
    res = {}
    if '[TFRS]' in review:
        review = review.replace(' [TFRS]', '')
        result = TRANSFORMERS.init_predict(review)
        res['ml_score'] = result['score']
        res['ml_topic'] = result['topics']
        res['review_body'] = result['text_clean']
    elif '[LTSM]' in review:
        review = review.replace(' [LSTM]', '')
        res = KERAS_ML.single_prediction(sequence=review).predict_reviews()
        res['ml_score'] = "POSITIF" if res['ml_score'] == 1 else 'NEGATIF'
    else:
        review = review.replace(' [LSTM]', '')
        res = KERAS_ML.single_prediction(sequence=review).predict_reviews()
        res['ml_score'] = "POSITIF" if res['ml_score'] == 1 else 'NEGATIF'
    return render_template('results_prediction.html', res=res)


@app.route('/dash')
def dash():
    return render_template('dash.html')


@app.route('/get_piechart_data')
def get_piechart_data():
    pieChartData = dashboard[1]
    return jsonify(pieChartData)


@app.route('/get_barchart_data')
def get_barchart_data():
    barChartData = dashboard[0]
    return jsonify(barChartData)


if __name__ == "__main__":
    app.run(debug=False, port=4000, host='0.0.0.0', use_reloader=False)
