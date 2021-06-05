# -*- coding: utf-8 -*-
import scrapy
from ..items import DuanpingItem


class SansanSpider(scrapy.Spider):
    name = 'sansan'
    start_urls = [
        'https://book.douban.com/subject/30409058/comments/?start=0&limit=20&status=P&sort=new_score']
    start = 0

    def parse(self, response):
        items = DuanpingItem()
        lists = response.xpath('//li[@class="comment-item"]')
        for i in lists:
            items['comment'] = i.xpath('./div/p/span/text()').get()
            items['comment_time'] = i.xpath('./div/h3/span/span/@title').get()
            items['star'] = i.xpath('./div/h3/span/span/@class="comment-time"/text()').get()
            items['likesCount'] = i.xpath('./div/h3/span/span/text()').get()

            yield items

        if self.start <= 500:
            self.start += 20
            url = 'https://book.douban.com/subject/30409058/comments/?start={}&limit=20&status=P&sort=new_score'.format(
                str(self.start))

            yield scrapy.Request(url=url, callback=self.parse)