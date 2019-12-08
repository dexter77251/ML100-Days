#!/usr/bin/env python
# coding: utf-8

# # 作業目標:
#     寫出 ReLU & dReLU 一階導數
#     並列印
# 

# # 作業重點
# 
# # Rectified Linear Unit- Relu 
# 
# f(x)=max(0,x)
# 

# In[1]:


import numpy as np
from numpy import *
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')

'''
作業:
    寫出 ReLU & dReLU 一階導數
    並列印
'''


# In[2]:


X=np.arange(-10,10,0.01)

def ReLu(x):
    return x if x>0 else 0
def dReLu(x):
    return 1 if x>0 else 0


# In[3]:


Y = [ReLu(x) for x in X]
plt.plot(X, Y)
plt.grid(color='b' , linewidth='0.1' ,linestyle='--')


# In[4]:


dY = [dReLu(x) for x in X]
plt.plot(X, dY)
plt.grid(color='b' , linewidth='0.1' ,linestyle='--')

