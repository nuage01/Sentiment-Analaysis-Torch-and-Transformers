# -*- coding: utf-8 -*-
import scrapy
from urllib.parse import urljoin

from scrapy import Request
from retailers_spiders.items import CrawlingItem
import json
from furl import furl
import re
import random
from retailers_spiders.tools.session import requests_retry_session
import requests

class TakealotSpider(scrapy.Spider):
    def __init__(self, shop_id=None, enseigne_id=1, country='ZA', *a, **kw):
        super(TakealotSpider, self).__init__(*a, **kw)

        self.shopInfo = {}
        self.shopInfo['enseigne_id']  = enseigne_id
        self.shopInfo['shop_id']  = shop_id
        self.shopInfo['country'] = country
        self._headers = {}

    ITEMS_PER_PAGE = 20
    name    = "takealot_product"
    domain = 'https://www.takealot.com'
    proxies_source = 'https://free-proxy-list.net/'
    proxy = None
    USER_AGENT_LIST = ['Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.369',
    'Mozilla/5.0 (Macintosh; U; PPC Mac OS X; de-de) AppleWebKit/125.2 (KHTML, like Gecko) Safari/125.7',
    'Mozilla/5.0 (Macintosh; U; PPC Mac OS X; en-us) AppleWebKit/312.8 (KHTML, like Gecko) Safari/312.6',
    'Mozilla/5.0 (Windows; U; Windows NT 5.1; cs-CZ) AppleWebKit/523.15 (KHTML, like Gecko) Version/3.0 Safari/523.15',
    'Mozilla/5.0 (Windows; U; Windows NT 6.0; en-US) AppleWebKit/528.16 (KHTML, like Gecko) Version/4.0 Safari/528.16',
    'Mozilla/5.0 (Macintosh; U; PPC Mac OS X 10_5_6; it-it) AppleWebKit/528.16 (KHTML, like Gecko) Version/4.0 Safari/528.16',
    'Mozilla/5.0 (Windows; U; Windows NT 6.1; zh-HK) AppleWebKit/533.18.1 (KHTML, like Gecko) Version/5.0.2 Safari/533.18.5',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2486.0 Safari/537.36 Edge/13.10547',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; Xbox; Xbox One) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2486.0 Safari/537.36 Edge/13.10586',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.82 Safari/537.36 Edge/14.14359']

    @property
    def headers(self):
        return self._headers

    @headers.setter
    def headers(self, rotating=False):
        if rotating:
            self._headers = {'User-Agent': f'{random.choice(self.USER_AGENT_LIST)}',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,' + \
            'image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9'}
        else:
            slef._headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'
            }

    api_url = "https://api.takealot.com/rest/v-1-8-0/productlines/" + \
        "search?sort=BestSelling%20Descending&rows=200&start={}&detail=" + \
        "mlisting&filter=Category:{}&filter=Available:true"

    def start_requests(self):
        yield Request(self.proxies_source, self.parse_proxies)

    def parse_proxies(self, response):
        # picking a random free proxy
        proxies = re.findall(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}.', response.text)
        for p in proxies:
            proxies = {
            "http": f"http://{p}",
            "https": f"http://{p}",
            }
            # s = requests_retry_session(proxies=proxies)
            self.headers = 1
            s = requests.Session()
            s.headers.update(self.headers)
            r = s.get(self.api_url.format(0, '25746'))
            if r.status_code !=200:
                print(proxies)
                r = s.get(self.api_url.format(0, '25746'), proxies=proxies, verify=False)
                if r.status_code == 200:
                    self.proxy = f'https://{p}'
                    yield Request(self.api_url.format(0, '25746'), self.parse_categories,meta={"proxy": self.proxy})
                    break
            else:
                yield Request(self.api_url.format(0, '25746'), self.parse_categories)
                break

    def parse_categories(self, response):
        categories = ['25184', '25746', '25750', '25749', '25748', '25747', '31737']
        for cat in categories:
            self.headers = 1
            url = self.api_url.format(0, cat)
            if response.meta.get('proxy', None):
                yield Request(url, self.parse_products, headers=self.headers, meta={"proxy": self.proxy})
            else:
                yield Request(url, self.parse_products, headers=self.headers)

    def parse_products(self, response):
        js = json.loads(response.body)
        total = js['results']['num_found']
        category = [
            entry['display_name']
            for entry in js['results']['breadcrumbs']['category']['entries']
        ]
        pages_num = total / self.ITEMS_PER_PAGE if total % self.ITEMS_PER_PAGE == 0 \
            else total // self.ITEMS_PER_PAGE + 1

        try:
            products = js['results']['productlines']
        except Exception:
            self.logger.exception('pas de données récupérées')
        if products:
            position = response.meta.get('position', 0)
            for p in products:
                product = CrawlingItem()
                title = p['title']
                rating = p['star_rating']
                p_id = p['id']
                detail_link = p['uri']
                product['currentURL'] = response.url
                product['productRating'] = rating
                product['productLinkDetail'] = detail_link
                product['product_id'] = str(p_id)
                product['productName'] = title.encode('utf-8')
                yield product
        print(pages_num)
        for page in range(2, pages_num + 1):
            f = furl(response.url)
            f.args['start'] = (page - 1) * self.ITEMS_PER_PAGE
            url = f.url
            yield Request(url, self.parse_products, meta=response.meta)
