
import scrapy


class CrawlingItem(scrapy.Item):
    product_id = scrapy.Field()                  # ID du produit
    productName = scrapy.Field()                # Nom du produit
    productDate = scrapy.Field()                # Date de prise du produit
    productRating = scrapy.Field()              # Note du produit
    productQuantityRating = scrapy.Field()      # Nombre de vote
    currentURL = scrapy.Field()
    enseigneID = scrapy.Field()                 # ID de l'enseigne
    enseigneName = scrapy.Field()               # ID Nom de l'enseigne
    shopID = scrapy.Field()                     # ID du magasin chez dataimpact
    country = scrapy.Field()                    # Pays de l'enseigne
    productLinkDetail = scrapy.Field()

class ReviewItem(scrapy.Item):
    rating = scrapy.Field() 
    content = scrapy.Field() 
    date = scrapy.Field() 
    product_id = scrapy.Field() 
    product_rating_count  = scrapy.Field() 
    product_rating_average  = scrapy.Field() 
    country = scrapy.Field()
    crawl_date =  scrapy.Field()
    enseigneID = scrapy.Field()
    enseigneName = scrapy.Field()
    productName = scrapy.Field()