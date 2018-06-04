
# coding: utf-8

# #### Set seed for Visibilty

# In[1]:


import numpy as np
np.random.seed(42)


# #### Load dependencies

# In[2]:


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


# #### Load Data

# In[3]:


(X_train, y_train) ,(X_test, y_test) = mnist.load_data()


# In[4]:


X_train.shape


# In[5]:


y_train.shape


# In[6]:


y_train[0:100]


# In[7]:


X_train[0]

#Below is the pixel information for number 5


# In[8]:


y_test.shape


# In[9]:


X_test.shape


# #### Preprocess the Data

# In[10]:


X_train = X_train.reshape(60000,784).astype('float32')
X_test = X_test.reshape(10000,784).astype('float32')


# In[11]:


#convert the highest values from 255 to 1

X_train /= 255
X_test /= 255


# In[12]:


#Number 5 from our training set
X_train[0]


# In[13]:


#One hot encoding
n_classes = 10
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)


# In[14]:


y_train[0]


# #### Design neural network architecture

# In[15]:


#We use sequential models
model = Sequential()

#This is the hidden layer (784*64)+64
model.add(Dense((64), activation = 'sigmoid', input_shape = (784,)))

#This is the output layer (64*10)+10
model.add(Dense((10), activation = 'softmax'))


# In[16]:


model.summary()


# #### Configure Model

# In[17]:


#lr is Learning rate and we use Accuracy Metric
model.compile(loss ='mean_squared_error', optimizer = SGD(lr = 0.01), metrics = ['accuracy'])


# #### Train

# In[18]:


#model.fit(X_train, y_train, batch_size = 128, epochs = 1, validation_data=(X_test, y_test))
#Only 9% accuarcy.So we need to increase epochs ie number of times model goe through data


# In[19]:


model.fit(X_train, y_train, batch_size = 128, epochs = 200, validation_data=(X_test, y_test))

