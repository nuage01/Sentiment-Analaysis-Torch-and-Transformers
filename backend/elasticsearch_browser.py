import string
import re
import logging
from elasticsearch import Elasticsearch
import os
import importlib

# fuser -n tcp -k 5000 to restart app
es_queries = importlib.import_module('src.processing.elastic')


class elastic_queries():
    def __init__(self, size, port=9200, index="scrapy_reviews", request_type=None):
        self.index = index
        self.size = size
        self.request_type = request_type
        self.port = port
        self.es = Elasticsearch('localhost', port=self.port)

    def basic_query(self, term):
        query = es_queries.q4
        query['query']['multi_match']['query'] = term
        res = self.es.search(
            index=self.index,
            size=self.size,
            body=query
        )
        return res

if __name__ == '__main__':
    pass