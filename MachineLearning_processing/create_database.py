# coding: utf-8
import sqlite3
import re
import logging

"""

$sqlite3 ml_processed.db

"""
conn = sqlite3.connect('./ml_processed.db')
conn.row_factory = sqlite3.Row

reviews = conn.cursor()
try:
    reviews.execute('CREATE  TABLE  IF NOT EXISTS ml_topics \
        (ID  INTEGER PRIMARY KEY AUTOINCREMENT, \
        topic varchar(30))'
    );
except Exception as e:
    logging.warning(e)


try:
    reviews.execute('CREATE  TABLE  IF NOT EXISTS products \
        (ID  varchar(30) PRIMARY KEY, \
        name varchar(30))'
    );
except Exception as e:
    logging.warning(e)


try:
    reviews.execute('CREATE  TABLE  IF NOT EXISTS retailers \
        (ID  INTEGER PRIMARY KEY AUTOINCREMENT, \
        name varchar(30))'
    );
except Exception as e:
    logging.warning(e)

try:
    reviews.execute('CREATE  TABLE  IF NOT EXISTS reviews \
        ( ID INTEGER PRIMARY KEY AUTOINCREMENT, \
        review_body varchar(100), \
        review_title varchar(100),\
        id_produit varchar(30), \
        opinion varchar(30), \
        topic varchar(30), \
        review_date date,\
        review_rating INT,\
        score INT,\
        retailer INT,\
        FOREIGN KEY (topic) REFERENCES ml_topics(ID), \
        FOREIGN KEY (retailer) REFERENCES retailers(ID), \
        FOREIGN KEY (id_produit) REFERENCES products(ID))'
 );
except Exception as e:
    logging.warning(e)


""" création des indexes afin d'optimiser les requetes SQL"""
create_index_products="CREATE INDEX IF NOT EXISTS id_products ON products(ID)"
create_index_topics="CREATE INDEX IF NOT EXISTS id_topics ON ml_topics(ID)"
create_index_reviews="CREATE INDEX IF NOT EXISTS id_reviews ON reviews(ID)"

reviews.execute(create_index_products)
reviews.execute(create_index_topics)
reviews.execute(create_index_reviews)

conn.commit()
reviews.close()
