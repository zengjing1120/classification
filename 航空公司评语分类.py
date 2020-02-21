import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

mpl.rcParams['font.sans-serif'] = ['SimHei']    # 指定默认字体
mpl.rcParams['font.serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False     # 解决保存图像是负号'-'显示为方块的问题

# 先确定字体，以免无法识别汉字
my_font = font_manager.FontProperties(fname=
                                      "C:/Windows/Fonts/msyh.ttc")

data = pd.read_csv('Tweets.csv')
data = data[['airline_sentiment','text']]
print(len(data))
print(data.head())

data_p = data[data.airline_sentiment=='positive']#3000
data_n = data[data.airline_sentiment=='negative']#9000
data_n = data_n.iloc[:len(data_p)]#3000

data = pd.concat([data_n, data_p])
print(data.shape)

data = data.sample(len(data))#打乱

data['review'] = (data.airline_sentiment == 'positive').astype('int')#0-1
del data['airline_sentiment']

token = re.compile('[A-Za-z]+|[!?,.()]')
def reg_text(text):
    new_text = token.findall(text)
    # print(new_text)
    new_text = [word.lower() for word in new_text]
    return new_text
data['text'] = data.text.apply(reg_text)

word_set = set()
for text in data.text:
    for word in text:
        word_set.add(word)
max_word = len(word_set)+1#7101
print(max_word)
word_list = list(word_set)
word_index = dict((word, word_list.index(word)+1) for word in word_list)

data_ok = data.text.apply(lambda x:[word_index.get(word, 0) for word in x])
print(data_ok)

maxlen = max(len(x) for x in data_ok)
data_ok = keras.preprocessing.sequence.pad_sequences(data_ok.values, maxlen=maxlen)#填充（10000，40）

model = keras.Sequential()
model.add(layers.Embedding(max_word, 50, input_length=maxlen))#（7100，40，50）
model.add(layers.LSTM(64))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())

# LSTM:decay=0.035  GRU:decay=0.023
adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.035)
model.compile(optimizer=adam,
             loss='binary_crossentropy',
             metrics=['acc'])

history=model.fit(data_ok, data.review.values,epochs=10,batch_size=128,validation_split=0.25)


#第一行第一列图形
ax1 = plt.subplot(1,2,1)
#第一行第二列图形
ax2 = plt.subplot(1,2,2)

#选择ax1
plt.sca(ax1)
#设置标题
plt.title('Loss')

plt.plot(history.epoch, history.history.get('loss'), c='r',label=u'training')
plt.plot(history.epoch, history.history.get('val_loss'), c='b',label=u'test')
plt.legend(prop=my_font)
#设置坐标轴刻度
my_x_ticks = np.arange(0, 10, 1)
plt.xticks(my_x_ticks)


#选择ax2
plt.sca(ax2)
#设置标题
plt.title('Accuracy')

plt.plot(history.epoch, history.history.get('acc'), c='r',label=u'training')
plt.plot(history.epoch, history.history.get('val_acc'), c='b',label=u'test')
plt.legend(prop=my_font)

#设置坐标轴刻度
my_x_ticks = np.arange(0, 10, 1)
plt.xticks(my_x_ticks)
plt.show()
model.save('my_model.h5')



