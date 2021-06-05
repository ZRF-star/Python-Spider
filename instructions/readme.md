**一、*****\*研究内容\****

对我这学期看了断断续续看了两个月的《应物兄》这本书的豆瓣书评进行文本分析。看一下大家对这本获得了茅盾文学奖的书的看法，看看大家的想法。

 

***\*补充\****：爬取豆瓣的《应物兄》书评是作为Plan B的，想爬的是微信读书的书评，不过微信读书的反爬虫机制真的非常好，它的内容是在canvas 上直接绘制，把文字画出来的，然后网页加载，在Google和360浏览器可以看到的书评数量是不一样的，爬取的时候，只能爬取到很小的一部分书评，在网页开始加载的后面一部分是爬取不到的，有的网页后面的都加载不了。关于爬取微信读书想法的文章几乎没有。了解和使用Android原生App爬虫，使用Python来操作手机上的app来爬取，还是有问题。时间原因，最终选择爬取豆瓣书评。虽然问题没有解决，不过还是了解和学习到很多。

 

 

 

**二、*****\*数据爬取过程\****

**1、*****\*使用scrapy框架进行数据的爬取\****

 

***\*遇到的问题：\****

（1）不会使用xpath来解析，就想着用正则,发现用正则更复杂，和css解析对比之后选择了学习xpath来解析，最初问题解决；

（2）每次在爬取了两百多天的时候出现403错误；解决方法：在setting.py文件中设置如下：

DOWNLOADER_MIDDLEWARES = {
  'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
}

COOKIES_ENABLED = False

第二个问题解决；

 

 

**2、*****\*数据分析与可视化\****

 

***\*遇到的问题：\****

（1）爬取的数据读取不了，出现编码错误；解决方法：在读取文件之前修改文件的编码格式为utf-8，问题解决；

（2）在进行中文分词时出现MerryError错误，解决方法：比较奇怪，把两句代码作为一句就没有问题了

comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]|/d+')
comment = comp.sub('', comment)

 

comm = re.sub(r'[^A-Z^a-z^0-9^\u4e00-\u9fa5]|/d+', '', comm)  # 用于移除英文和数字

（3）在对评论时间进行处理时，出现问题：TypeError: unsupported operand type(s) for |: 'str' and 'str'；  解决方法：给键入的数值定义；分成两个条件语句，问题解决；

（4）在显示日期的时候显示不完全，解决方法：将字符串日期转换为int类型，问题解决；

（5）可视化的时候词云图的蒙版显示不了；解决方法：png格式的图片转换为jpg格式；

（6）然后就是各种在做的时候出现的小问题。

 

 

 

**三、*****\*数据分析方法\****

基本的统计方法、朴素贝叶斯

 

 

 

**四、*****\*Python代码\****

**（1）*****\*爬取数据代码\****

 

***\*Setting.py文件：\****

BOT_NAME = 'duanping'
SPIDER_MODULES = ['duanping.spiders']
NEWSPIDER_MODULE = 'duanping.spiders'
USER_AGENT = 'Mozilla/5.0'
DOWNLOADER_MIDDLEWARES = {
'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
}
ROBOTSTXT_OBEY = False
DOWNLOAD_DELAY = 1
COOKIES_ENABLED = False
DEFAULT_REQUEST_HEADERS = {
 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=

0.8',
 'Accept-Language': 'en',
}
ITEM_PIPELINES = {
  'duanping.pipelines.DuanpingPipeline': 300,
}

 

***\*items.py文件：\****

import scrapy
class DuanpingItem(scrapy.Item):
  comment = scrapy.Field()
  comment_time = scrapy.Field()
  star = scrapy.Field()
  likesCount = scrapy.Field()

 

***\*Sansan.py文件：\****

\# -*- coding: utf-8 -*-
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

​      yield items

​    if self.start <= 500:
​      self.start += 20
​      url = 'https://book.douban.com/subject/30409058/comments/?start={}&limit=20&status=P&sort=new_score'.format(
​        str(self.start))
​      yield scrapy.Request(url=url, callback=self.parse)

 

 

 

***\*pipelines.py\****

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

 

***\*run.py文件：\****

from scrapy import cmdline
cmdline.execute('scrapy crawl sansan'.split())

 

 

**（2）*****\*数据分析与可视化数据代码\****

import array
import re
from tkinter import _flatten
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import metrics

from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB  
from sklearn.model_selection import train_test_split 
import jieba
import wordcloud
from wordcloud import ImageColorGenerator, STOPWORDS

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)

f = pd.read_csv("应物兄短评.csv")
f = pd.DataFrame(f)

comment = f['comment']
comment_time = f['comment_time']
star = f['star']
likesCount = f['likesCount']

\#对缺失值进行处理
comment = comment.fillna(" ", inplace=True)
star = star.fillna("较差", inplace=True)
likesCount = likesCount.fillna("0", inplace=True)

\#中文分词
comment = f['comment'].tolist()#将series类型转换为列表类型
comments = list()
for comm in comment:
  comm = re.sub(r'[^A-Z^a-z^0-9^\u4e00-\u9fa5]|/d+', '', comm)
  comments.append(comm)
print("中文分词:")
print(comments)

\#分词
stopwords = open("stopwords.txt", encoding="utf-8")
stopwords1 = list()
for stopword in stopwords.readlines():
  curLine = stopword.strip().split(" ")
  stopwords1.append(curLine)
stopwords1 = list(_flatten(stopwords1))

new_Series = pd.Series()

new_list = list()
for comment in comments:
  ls = list(jieba.cut_for_search(comment))
  ls = [w for w in ls if w not in stopwords1]
  txt = " ".join(ls)
  new_list.append(txt)

new_Series = pd.Series(new_list)
f['comment'] = new_Series

\#处理星级
star = f['star'].tolist()
stars = list()
for s in star:
  if s == "力荐":
    stars.append(5)
  elif s == "推荐":
    stars.append(4)
  elif s == "还行":
    stars.append(3)
  elif s == "较差":
    stars.append(2)
  elif s == "很差":
    stars.append(1)
print(stars)

new_star = pd.Series(stars)
f['star'] = new_star

\#星级计算和
star_num = list()
wuxing = f[(f.star == 5)]['likesCount']
star_num.append(wuxing.sum())
print("五星：", end="")#不换行输出
print(wuxing.sum())

sixing = f[(f.star == 4)]['likesCount']
star_num.append(sixing.sum())
print("四星：", end="")#不换行输出
print(sixing.sum())

sanxing = f[(f.star == 3)]['likesCount']
star_num.append(sanxing.sum())
print("三星：", end="")#不换行输出
print(sanxing.sum())

erxing = f[(f.star == 2)]['likesCount']
star_num.append(erxing.sum())
print("二星：", end="")#不换行输出
print(erxing.sum())

yixing = f[(f.star == 1)]['likesCount']
star_num.append(yixing.sum())
print("一星：", end="")#不换行输出
print(yixing.sum())


print("=======================================可视化==============================")
print("=======================评论时间===================")
matplotlib.rcParams['font.family'] = 'Kaiti'
comment_time = comment_time.tolist()
comments_time = list()
for time in comment_time:
  time = time.split('/')
  if time[1]=="10":
    time = time[0] + time[1]
    time = int(time)
    comments_time.append(time)
  elif time[1]=="11":
    time = time[0] + time[1]
    time = int(time)
    comments_time.append(time)
  elif time[1]=="12":
    time = time[0] + time[1]
    time = int(time)
    comments_time.append(time)
  else:
    time = time[0] + "0" + time[1]
    time = int(time)
    comments_time.append(time)
new_comments = pd.Series(comments_time)

new_comments = new_comments.value_counts()
new_comments = new_comments.sort_index()
new_comments = pd.DataFrame(new_comments).reset_index()
new_comments = new_comments.rename(columns={'index':'日期', 0:'数量'})
x = new_comments['日期'].tolist()
x = [str(i) for i in x]#为了显示完全，转换数据类型
y = new_comments['数量'].tolist()
plt.plot(x, y, linewidth=2, marker='.', color="blue")
plt.title("随着日期的评论的数量")
plt.xlabel("评论日期")
plt.ylabel("评论数量")
plt.xticks(rotation=-30)
plt.show()
print("=========================评分比例=========================")
labels = '5星', '4星', '3星', '2星', '1星'
sizes = star_num
explode = 0.05, 0.05, 0.05, 0.05, 0.05
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%'

, shadow=True)
plt.title("《应物兄》豆瓣评分比例")
plt.axis('equal')
plt.show()
print("=================认为好=========================")
content = f[(f.star=='4')|(f.star=='5')]['comment']
contents = list(content)
file = open("contents.txt", 'a')
for i in range(len(contents)):
  s = str(contents[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
  s = s.replace("'",'').replace(',','') +'\n'  #去除单引号，逗号，每行末尾追加换行符
  file.write(s)
alice_coloring = np.array(Image.open('好.jpg'))
f_w = open("contents.txt", "r")
t = f_w.read()
ls_w = jieba.lcut(t)
txt_w = " ".join(ls_w)
w = wordcloud.WordCloud(font_path="msyh.ttc", background_color="white",
            width=500,height=500,mask=alice_coloring,

 collocations=False,
            max_words=300 ,stopwords=['应物','小说','分子','文学'])
w.generate(t)
image_color = ImageColorGenerator(alice_coloring)
plt.imshow(w, interpolation='bilinear')
plt.title("认为《应物兄》不错")
plt.axis('off')
plt.show()
w.to_file("like.png")

print("=================认为不好=========================")
content = f[(f.star=='1')|(f.star=='2')]['comment']
contents = list(content)
file = open("contents2.txt", 'a')
for i in range(len(contents)):
  s = str(contents[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
  s = s.replace("'",'').replace(',','') +'\n'  #去除单引号，逗号，每行末尾追加换行符
  file.write(s)
alice_coloring = np.array(Image.open('不好.jpg'))
f_w = open("contents.txt", "r")
t = f_w.read()
ls_w = jieba.lcut(t)
txt_w = " ".join(ls_w)
w = wordcloud.WordCloud(font_path="msyh.ttc", background_color="white",
            width=500,height=500,mask=alice_coloring,

 collocations=False,
            max_words=300,
            stopwords=['应物','小说','分子','文学'])
w.generate(t)
image_color = ImageColorGenerator(alice_coloring)
plt.imshow(w, interpolation='bilinear')
plt.title("认为《应物兄》不好")
plt.axis('off')
plt.show()
w.to_file("dislike.png")

print("=============================朴素贝叶斯======================")
star2 = list()
for st in stars:
  if st > 3:
    st = 1
    star2.append(st)
  else:
    st = 0
    star2.append(st)
f['star'] = star2
print(f['star'])
print(f)

clf = MultinomialNB()
x = f['comment']
x = list(x)
y = f['star']
y = list(y)
n = len(x)//8
x_train, y_train = x[n:], y[n:]
x_train = pd.Series(x_train)
y_train = pd.Series(y_train)
x_test, y_test = x[:n], y[:n]
x_test = pd.Series(x_test)
y_test = pd.Series(y_test)


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
count_vec = CountVectorizer(max_df=0.8, min_df=3)
tfidf_vec = TfidfVectorizer()

def MNB_Classifier():
  return Pipeline([
    ('count_vec', CountVectorizer()),
    ('mnb', MultinomialNB())
  ])
mnbc_clf = MNB_Classifier()
\# 进行训练
print("Start training...")
mnbc_clf.fit(x_train, y_train)
print("training done!")
answer_b = mnbc_clf.predict(x_test)
print("0：差评和中评；1：好评")
print(answer_b)
print("Prediction done!")
\#准确率测试
accuracy = metrics.accuracy_score(y_test,answer_b)
print('准确率：'+str(accuracy))
print("The classification report for b:")
print(classification_report(y_test, answer_b))

 

 

 

 

 

 

 

 

 

 

**五、*****\*实验运行结果\****

 

![img](file:///C:\Users\zrf\AppData\Local\Temp\ksohtml15808\wps1.jpg) 

 

 

 

![img](file:///C:\Users\zrf\AppData\Local\Temp\ksohtml15808\wps2.jpg) 

 

 

 

**六、*****\*可视化展示\****

 

![img](file:///C:\Users\zrf\AppData\Local\Temp\ksohtml15808\wps3.jpg) 

 

![img](file:///C:\Users\zrf\AppData\Local\Temp\ksohtml15808\wps4.jpg) 

![img](file:///C:\Users\zrf\AppData\Local\Temp\ksohtml15808\wps5.jpg) 

 

![img](file:///C:\Users\zrf\AppData\Local\Temp\ksohtml15808\wps6.jpg) 

 

 

**七、*****\*结果分析\****

根据折线图可以得到大多评论在2019年8月到2019年9月，该书出版在2018年12月，所以故意恶评或者水军存在可能性小；所以数据是可信的；

从豆瓣评分比例图可以看到好评和差评比例是差不多的；

从词云图可以看到好评和差评都集中在故事、讽刺、掉书袋、儒学等等；

经过朴素贝叶斯进行分析，差评和中评占很大一部分；

***\*结论\****：经过分析，我觉得跟我预想还是有一部分差别的，没有想到差评占大部分，不过作为一本比较小众的书，可能大家都有自己的看法。我自己最开始看了一部分，然后弃了，这学期又开始看了，我个人觉得还是很不错的一本书，描述了一个社会现状，值得细细品味的。最后结论是，看豆瓣书评也可能不能正确帮助我们判断一本书的好与不好，还是自己看完书再来看书评，看看别人对书的想法。

 