#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install konlpy
# !pip install koco


# In[9]:


import pandas as pd
import time
import json
# import koco


# In[10]:


# df1 = pd.read_excel("출발.xlsx",sheet_name='Stop', engine='openpyxl')
# df2 = pd.read_excel("출발.xlsx",sheet_name='Go', engine='openpyxl')
# df3 = pd.read_excel("멈춤.xlsx", engine='openpyxl')
# len(df1), len(df2), len(df3)


# In[11]:


# df3.rename(columns = {'내려주세요' : '문장', 0:'label'}, inplace = True)
# df3


# In[12]:


# df = pd.concat([df1,df3])
# df


# In[13]:


# df.drop_duplicates(inplace=True)
# df2.drop_duplicates(inplace=True)
# len(df), len(df2)


# In[14]:


# df = pd.concat([df,df2])


# In[15]:


# df


# In[11]:


# pip install tensorflow


# In[16]:


import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


# In[17]:


# df['문장'] = df['문장'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 한글만 남기기


# In[18]:


# df['문장'] = df['문장'].str.replace('^ +', "") # 공백 제거
# df['문장'].replace('', np.nan, inplace=True)
# df.isnull().sum() # 한글만 남긴 뒤 null값


# In[19]:


# 한글만 남겼을 때 null 값 31개 drop
# df = df.dropna(how = 'any')


# In[20]:


# df.shape


# In[21]:


stopwords = ['의','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다'] # 한글에서 자주 사용하는 불용어


# In[22]:


# train, test= train_test_split(df, test_size=0.2, random_state=42)


# In[23]:


okt = Okt()


# In[24]:


# X_train = []
# for sentence in train['문장']:
#     temp_X = okt.morphs(sentence, stem=True) # 토큰화
#     temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
#     X_train.append(temp_X)


# In[25]:


# X_test = []
# for sentence in test['문장']:
#     temp_X = okt.morphs(sentence, stem=True) # 토큰화
#     temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
#     X_test.append(temp_X)


# In[26]:


# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(X_train)


# In[27]:


# len(tokenizer.word_index)


# In[28]:


# vocab_size = len(tokenizer.word_index)+1


# In[29]:


# tokenizer = Tokenizer(vocab_size) 
# tokenizer.fit_on_texts(X_train)
# X_train = tokenizer.texts_to_sequences(X_train)
# X_test = tokenizer.texts_to_sequences(X_test)


# In[30]:


# y_train = np.array(train['label'])
# y_test = np.array(test['label'])


# In[31]:


# max_len = 5


# In[32]:


# X_train = pad_sequences(X_train, maxlen = max_len)
# X_test = pad_sequences(X_test, maxlen = max_len)


# In[ ]:


###########################################################################################################
# LSTM 모델
#https://wikidocs.net/44249


# In[33]:


from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop


# In[34]:


loaded_model = load_model('best_model.h5')
# print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

import pickle5 as pickle

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# In[39]:


def stop_or_go(new_sentence):
    
    st = time.time()
    new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
    encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen = 5) # 패딩
    score = float(loaded_model.predict(pad_new)) # 예측
    ed = time.time()

    print(f"{ed-st}")
    if(score > 0.5):
        return "출발"
    else:
        return "정지"


# In[40]:


# bad_or_not('이제 출발해주세요')


# In[ ]:




