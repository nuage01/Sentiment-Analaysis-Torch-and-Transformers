# -*- coding: utf-8 -*-
# script that contains queries and other operations on our Elasticsearch database

# runing on http://localhost:9200

from elasticsearch import Elasticsearch
import requests
import json


q1 = {"query": {
    "match_all": {}
},
    "aggs": {
    "mydata_agg": {
        "terms": {"field": "content"}
    }
}}


q2 = {"query": {"match": {"content": "good"}}}

q3 = {
    "aggs": {
        "keywords": {
            "significant_text": {
                "field": "content",
            }
        }
    }
}

q4 = {
    "query": {
        "multi_match": {
            "query": '{0}',
            "fields": [
                'content^3',
                'rating',
                'productName'
            ]
        }
    }
}

"""Comment faire des reqûetes sur note BDD Elasticsearch"""
# init es
# es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

# method 1 using Elasticsearch library
# es.search(index="scrapy_reviews", body=q4)

# method 2 using requests
# r = requests.get('http://localhost:9200/scrapy_reviews/_search?size=11')
# result = json.loads(r.text)

if __name__ == "__main__":
    pass
