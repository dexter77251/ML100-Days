#!/usr/bin/env python
# coding: utf-8

# # 作業重點
# 
# 使用function y=(x+5)² 來練習學習率( leanrning rate ) 調整所造成的影響

# # 作業目標:
#     請嘗試使用不同的組合驗證 learning rate 對所需 iteration 的影響
#     lr = [0.1, 0.0001]
#     主要驗證 Lr 對於grandient 收斂的速度
#     

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


cur_x = 3 # The algorithm starts at x=3
precision = 0.000001 #This tells us when to stop the algorithm
max_iters = 10000 # maximum number of iterations


# In[4]:


'''
需要執行初始值設定, 下列三個
# Learning rate
#iteration counter
#Step size  
'''
'''
while previous_step_size > precision and iters < max_iters:

  算法迭代更新

print("the learning rate is",lr, "\nIteration",iters,"\nX value is",cur_x) #Print iterations
'''


# In[5]:


def df(x):
    return 2*(x+5)

def Fn(x):
    return (x+5)**2


# In[6]:


x = np.linspace(-15, 5, 100)
y = (x + 5) ** 2
plt.plot(x, y)
plt.grid()

lr = 0.1
iters = 0
cur_x = 3
previous_step_size = 1
while previous_step_size > precision and iters < max_iters:
    '''
    算法迭代更新
    '''
    iters += 1
    previous_step_size = df(cur_x) * lr
    next_x = cur_x - previous_step_size
    plt.scatter(cur_x, Fn(cur_x), s=50)
    plt.plot([cur_x, next_x], [Fn(cur_x), Fn(next_x)], 'k')
    cur_x = next_x


# In[7]:


x = np.linspace(-15, 5, 100)
y = (x + 5) ** 2
plt.plot(x, y)
plt.grid()

lr = 0.0001
iters = 0
cur_x = 3
previous_step_size = 1
while previous_step_size > precision and iters < max_iters:
    '''
    算法迭代更新
    '''
    iters += 1
    previous_step_size = df(cur_x) * lr
    next_x = cur_x - previous_step_size
    plt.scatter(cur_x, Fn(cur_x), s=50)
    plt.plot([cur_x, next_x], [Fn(cur_x), Fn(next_x)], 'k')
    cur_x = next_x


# In[ ]:




