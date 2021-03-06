# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter


class DuanpingPipeline:
    def process_item(self, item, spider):
        with open('应物兄短评.csv', 'a') as f:
            item['comment'] = item.get('comment')
            item['star'] = item.get('star')
            item['likesCount'] = item.get('likesCount')
            txt = str.format('{},{},{}\n', item['comment'], item['star'], item['likesCount'])
            f.write(txt)
        print(item)
        return item
