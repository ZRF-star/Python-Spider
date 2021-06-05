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
from sklearn.naive_bayes import MultinomialNB  #导入朴素贝叶斯分类器
from sklearn.model_selection import train_test_split #导入自动生成训练集和测试集的模块train_test_split
import jieba
import wordcloud
from wordcloud import ImageColorGenerator, STOPWORDS

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)#禁止自动换行

f = pd.read_csv("应物兄短评.csv")
f = pd.DataFrame(f)

comment = f['comment']
comment_time = f['comment_time']
star = f['star']
likesCount = f['likesCount']

#对缺失值进行处理
comment = comment.fillna(" ", inplace=True)
star = star.fillna("较差", inplace=True)
likesCount = likesCount.fillna("0", inplace=True)


#数据预处理
#中文分词
comment = f['comment'].tolist()#将series类型转换为列表类型
comments = list()
for comm in comment:
    comm = re.sub(r'[^A-Z^a-z^0-9^\u4e00-\u9fa5]|/d+', '', comm)  # 用于移除英文和数字
    comments.append(comm)
print("中文分词:")
print(comments)

#分词
stopwords = open("stopwords.txt", encoding="utf-8")
stopwords1 = list()
for stopword in stopwords.readlines():
    curLine = stopword.strip().split(" ")
    stopwords1.append(curLine)
stopwords1 = list(_flatten(stopwords1))#二维转一维
# print("停用词：")
# print(stopwords1)

new_Series = pd.Series()
#处理停用词
new_list = list()
for comment in comments:
    ls = list(jieba.cut_for_search(comment))
    ls = [w for w in ls if w not in stopwords1]
    txt = " ".join(ls)
    new_list.append(txt)
print("去除停用词：")
print(new_list)

new_Series = pd.Series(new_list)
f['comment'] = new_Series

#处理星级
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

#星级计算和
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
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)
plt.title("《应物兄》豆瓣评分比例")
plt.axis('equal')
plt.show()
print("=================认为好=========================")
content = f[(f.star=='4')|(f.star=='5')]['comment']
contents = list(content)
file = open("contents.txt", 'a')
for i in range(len(contents)):
    s = str(contents[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
    s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
    file.write(s)
alice_coloring = np.array(Image.open('好.jpg'))
f_w = open("contents.txt", "r")
t = f_w.read()
ls_w = jieba.lcut(t)
txt_w = " ".join(ls_w)
w = wordcloud.WordCloud(font_path="msyh.ttc", background_color="white",
                        width=500,height=500,mask=alice_coloring, collocations=False,
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
    s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
    file.write(s)
alice_coloring = np.array(Image.open('不好.jpg'))
f_w = open("contents.txt", "r")
t = f_w.read()
ls_w = jieba.lcut(t)
txt_w = " ".join(ls_w)
w = wordcloud.WordCloud(font_path="msyh.ttc", background_color="white",
                        width=500,height=500,mask=alice_coloring, collocations=False,
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
# 进行训练
print("Start training...")
mnbc_clf.fit(x_train, y_train)
print("training done!")
answer_b = mnbc_clf.predict(x_test)
print("0：差评和中评；1：好评")
print(answer_b)
print("Prediction done!")
#准确率测试
accuracy = metrics.accuracy_score(y_test,answer_b)
print('准确率：'+str(accuracy))
print("The classification report for b:")
print(classification_report(y_test, answer_b))

