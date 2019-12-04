#!/usr/bin/env python
# coding: utf-8

# #   
# 
# # 作業目標:
# 
#     1. 藉由固定的 dataset, 來驗證不同loss function
#     2. Dataset 的特性跟我們選用的loss function 對accrancy 的影響
#     
#     
# # 作業重點: 
#     請分別選用 "MSE", "binary _crossentropy"
#     查看Train/test accurancy and loss rate
#     

# # 導入必要的函數

# In[2]:


from keras.datasets import cifar10
import numpy as np
np.random.seed(10)


# # 資料準備

# In[3]:


#取得Keras Dataset
(x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()


# In[4]:


#確認 CIFAR10 Dataset 資料維度
print("train data:",'images:',x_img_train.shape,
      " labels:",y_label_train.shape) 
print("test  data:",'images:',x_img_test.shape ,
      " labels:",y_label_test.shape) 


# In[5]:


#資料正規化
x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0


# In[6]:


#針對Label 做 ONE HOT ENCODE
from keras.utils import np_utils
y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)
y_label_test_OneHot.shape


# # 建立模型

# In[7]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D


# In[8]:


model = Sequential()


# In[9]:


#卷積層1


# In[10]:


model.add(Conv2D(filters=32,kernel_size=(3,3),
                 input_shape=(32, 32,3), 
                 activation='relu', 
                 padding='same'))


# In[11]:


model.add(Dropout(rate=0.25))


# In[12]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[13]:


#卷積層2與池化層2


# In[14]:


model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                 activation='relu', padding='same'))


# In[15]:


model.add(Dropout(0.25))


# In[16]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[17]:


#建立神經網路(平坦層、隱藏層、輸出層)


# In[18]:


model.add(Flatten())
model.add(Dropout(rate=0.25))


# In[19]:


model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.25))


# In[20]:


model.add(Dense(10, activation='softmax'))


# In[21]:


#檢查model 的STACK
print(model.summary())


# # 載入之前訓練的模型

# In[22]:


try:
    model.load_weights("SaveModel/cifarCnnModel.h5")
    print("載入模型成功!繼續訓練模型")
except :    
    print("載入模型失敗!開始訓練一個新模型")


# # 訓練模型

# In[29]:


def create_model():
    model = Sequential()
    
    #卷積層1
    model.add(Conv2D(filters=32,kernel_size=(3,3),
                 input_shape=(32, 32,3), 
                 activation='relu', 
                 padding='same'))
    model.add(Dropout(rate=0.25))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #卷積層2與池化層2
    model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                 activation='relu', padding='same'))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #建立神經網路(平坦層、隱藏層、輸出層)
    model.add(Flatten())
    model.add(Dropout(rate=0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(10, activation='softmax'))

    return model


# In[30]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train_acc', 'val_acc'], loc='upper left')
    plt.show()


# ## categorical_crossentropy

# In[32]:


model = create_model()
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

train_history=model.fit(x_img_train_normalize, y_label_train_OneHot,
                        validation_split=0.2,
                        epochs=5, batch_size=6, verbose=1) 

show_train_history('acc','val_acc')

show_train_history('loss','val_loss')

scores = model.evaluate(x_img_test_normalize, y_label_test_OneHot)
print()
print('accuracy=',scores[1])

score_0 = scores[1]


# ## MSE

# In[ ]:


model = create_model()
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
'''因使用筆電,這裏epochs只設定5次'''
train_history=model.fit(x_img_train_normalize, y_label_train_OneHot,
                        validation_split=0.25,
                        epochs=5, batch_size=6, verbose=1) 

show_train_history('acc','val_acc')

show_train_history('loss','val_loss')

scores = model.evaluate(x_img_test_normalize, y_label_test_OneHot)
print()
print('accuracy=',scores[1])

score_1 = scores[1]


# ## binary_crossentropy

# In[ ]:


model = create_model()
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
'''因使用筆電,這裏epochs只設定5次'''
train_history=model.fit(x_img_train_normalize, y_label_train_OneHot,
                        validation_split=0.25,
                        epochs=5, batch_size=6, verbose=1) 

show_train_history('acc','val_acc')

show_train_history('loss','val_loss')

scores = model.evaluate(x_img_test_normalize, y_label_test_OneHot)
print()
print('accuracy=',scores[1])

score_2 = scores[1]


# In[ ]:


print({'categorical_crossentropy:',score_0)
print({'mean_squared_error:',score_1)
print({'binary_crossentropy:',score_2)

