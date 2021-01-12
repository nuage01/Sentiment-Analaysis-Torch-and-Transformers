import sqlite3
import logging
import sys
import pandas as pnd
import re
from functools import reduce

conn = sqlite3.connect('./ml_processed.db')
conn.row_factory = sqlite3.Row
data = conn.cursor()
exists = False
global retailer_id

def regex_sub(x):
    return re.sub(r'\s+', '', x)


def etl(retailer):
    topics_defiend = {
        'allergens': 1,
        'competition': 2,
        'delivery': 3,
        'packaging': 4,
        'price': 5,
        'taste': 6 }
    try:
        data.execute('''SELECT ID
                            FROM retailers 
                            WHERE retailers.name = ? ''', (retailer,))
        retailer_id = data.fetchone()
        if retailer_id is None:
            data.execute('INSERT into retailers  (name) VALUES(?)', (retailer,))
            retailer_id =data.lastrowid
    except Exception as e:
        logging.warning(e)
    exists = False
    df = pnd.read_csv(retailer + '_final_ml_processed.csv')
    for k, v in topics_defiend.items():
        data.execute('INSERT into ml_topics  (ID, topic) VALUES(?,?)', (v, k))
    for row in df.iterrows():
        try:
            row = row[1].to_dict()
            data.execute('''SELECT ID
                                FROM products 
                                WHERE products.ID = ? ''', (row['asin'],))
            e = data.fetchone()
            if e is not None:
                exists = True
            else:
                data.execute('INSERT into products  (ID, name) VALUES(?,?)', (row['asin'], row['product_name']))
        except Exception as e:
            logging.warning(e)
        score = 1 if row['ml_score'] > 0 else 0
        topics = row['ml_topic']
        topics = topics.replace(']','').replace('[', '').replace('\'','').split(',')
        topics = list(filter(lambda x: x not in '', topics))
        topics = [regex_sub(topic) for topic in topics]
        key_topics = ''
        if len(topics)> 0:
            # for topic in topics:
            #     key_topics +=  str(topics_defiend[topic])
            key_topics = reduce(lambda x, y: x + ',' + y, [str(topics_defiend[topic]) for topic in topics])
        else:
            key_topics = None
        data.execute('INSERT into reviews  (review_body, review_title, id_produit, opinion, topic, review_date, review_rating, score, retailer) VALUES(?,?,?,?,?,?,?,?,?)',
                     (row['review_body'],
                     row['review_title'], 
                     row['asin'], 
                     row['opinion'], 
                     key_topics, 
                     row['review_date'], 
                     row['review_rating'], 
                     score,
                     retailer_id))

    conn.commit()
    conn.close()

if __name__ == "__main__":
    etl(sys.argv[1])
