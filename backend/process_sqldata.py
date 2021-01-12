import pandas as pd
import sqlite3


db = sqlite3.connect("../bdd/sql_database/ml_processed.db")
db.text_factory = str


def simple_call(retailer):
    query = """select r.review_body, r.id, r.score,r.review_date, m.topic from reviews as r  join ml_topics as m on m.id where  INSTR(r.topic, cast(m.id  as varchar(255))) > 0 and r.retailer={}""".format(
        retailer)
    df = pd.read_sql(query, db)
    return df.to_dict()


def dashboard_queries():
    query = """select r.review_body, r.id, r.score,r.id_produit, r.review_date, r.topics_str, p.name from reviews as r join products as p on  (r.id_produit=p.ID) """
    df = pd.read_sql(query, db)

    total_neg_query = 'select count(m.topic) as measure, m.topic as category from reviews as r \
    join ml_topics as m on m.id \
    where  INSTR(r.topic, cast(m.id  as varchar(255))) > 0 and r.score=0 group by m.topic;'
    df_total = pd.read_sql(total_neg_query, db)
    df_total = df_total[['category', 'measure']]
    df_total.to_csv('static/data/topics_negatives.csv', index=False)
    all_neg = df_total.to_dict('records')
    sum_values = sum([value['measure']for value in all_neg])
    for value in all_neg:
        value['group'] = 'All'
        value['measure'] = round(value['measure'] / sum_values, 1)

    negs = df.loc[df['score'] == 0].groupby(
        ['id_produit'])['score'].count().sort_values(ascending=False)
    negs = negs[:5].to_dict()
    pie_data = {k: round(v/sum(negs.values()), 1) for k, v in negs.items()}

    targets = ['ALLERGENS', 'COMPETITION',
               'DELIVERY', 'PACKAGING', 'PRICE', 'TASTE']
    targets_min = [topic.lower() for topic in targets]
    barChartData = []
    pieChartData = []
    dict_names = {}
    topics_query = 'select  r.id_produit, m.topic from reviews as r \
    join ml_topics as m on m.id \
    where  INSTR(r.topic, cast(m.id  as varchar(255))) > 0 and r.id_produit="{}" and r.score=0 ;'
    name_query = 'select name from products where (products.ID="{}");'

    for id_produit in negs.keys():
        name = pd.read_sql(name_query.format(id_produit), db).name.iloc[0]
        dict_names[id_produit] = name
        df_topics = pd.read_sql(topics_query.format(id_produit), db)
        df_topics = df_topics.groupby('topic').count().to_dict()['id_produit']
        df_topics = {k: round(v/sum(df_topics.values()), 1)
                     for k, v in df_topics.items()}
        for target in targets_min:
            mesure = df_topics[target] if target in df_topics.keys() else 0.0
            data = {'category': target, 'group': name[:20], 'measure': mesure}
            barChartData.append(data)
    barChartData.extend(all_neg)

    for key, value in pie_data.items():
        pieChartData.append(
            {'category': dict_names[key][:20], 'measure': value})

    return barChartData, pieChartData


if __name__ == "__main__":
    print('process module')
