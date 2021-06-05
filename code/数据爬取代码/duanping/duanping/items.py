# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class DuanpingItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    comment = scrapy.Field()
    comment_time = scrapy.Field()
    star = scrapy.Field()
    likesCount = scrapy.Field()
