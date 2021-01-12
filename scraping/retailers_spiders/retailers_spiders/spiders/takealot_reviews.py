from scrapy import Request
import scrapy   
import pandas as pnd
from retailers_spiders.items import ReviewItem
import json
from datetime import date as dt


class TakealotReviwSpider(scrapy.Spider):
    def __init__(self, shop_id=1, enseigne_id=1, name='takealot', country='UK', *a, **kw):
        super(TakealotReviwSpider, self).__init__(*a, **kw)

        self.shopInfo = {}
        self.shopInfo['enseigne_id']  = enseigne_id
        self.shopInfo['shop_id']  = shop_id
        self.shopInfo['country'] = country
        self.shopInfo['enseigne_name'] = name
        self._headers = {}
    name = 'takealot_reviews'
    url = "https://api.takealot.com/rest/v-1-9-0/product-details/{}/reviews?platform=desktop"

    def start_requests(self):
        
        df = pnd.read_csv(self.shopInfo['enseigne_name'] + '_products.csv')
        # df.isnull().sum(axis = 0)
        df =  df.loc[:, ['product_id', 'productName', 'productRating']]
        # reviews = df.productRating.fillna(0)
        df['productRating'] = pnd.to_numeric(df['productRating'],errors='coerce')
        ids = list(df['product_id'][df['productRating']>0 ])
        names = list(df['productName'][df['productRating']>0 ])
        for id, name in zip(ids, names):
            url = self.url.format(id)
            yield Request(url, self.parse_product_reviews, meta={'product_name': name})

    def parse_product_reviews(self, response):

        js = json.loads(response.text)
        reviews = js['items']
        product_rating_count = js['count']
        product_rating_average = f"{js['star_rating']}/5"

        for review in reviews:
            rating = f"{review['star_rating']}/5"
            content = review['review']
            date = review['date']
            product_id = review['id'].split('-')[1]
            item = ReviewItem()
            item['rating'] = rating
            item['content'] = content
            item['date'] = date
            item['product_id'] = product_id
            item['product_rating_count'] = product_rating_count
            item['product_rating_average'] = product_rating_average
            item['country'] = self.shopInfo['country']
            today = dt.today()
            item['crawl_date'] = today.strftime("%d/%m/%Y")
            item['enseigneID'] = self.shopInfo['enseigne_id']
            item['enseigneName'] = self.shopInfo['enseigne_name']
            item['productName'] = response.meta['product_name']
            yield item
