#!/usr/bin/env python
# coding: utf-8

# In[14]:


from keras.datasets import cifar10
import matplotlib.pyplot as plt
 
(train_X,train_Y),(test_X,test_Y)=cifar10.load_data()


# In[15]:


n=6
plt.figure(figsize=(20,10))
for i in range(n):
    plt.subplot(330+1+i)
    plt.imshow(train_X[i])
    plt.show()


# In[16]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils


# In[17]:


train_x=train_X.astype('float32')
test_X=test_X.astype('float32')
 
train_X=train_X/255.0
test_X=test_X/255.0


# In[18]:


train_Y=np_utils.to_categorical(train_Y)
test_Y=np_utils.to_categorical(test_Y)
 
num_classes=test_Y.shape[1]


# In[19]:


model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(32,32,3),
    padding='same',activation='relu',
    kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3),activation='relu',padding='same',kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512,activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


# In[20]:


sgd=SGD(learning_rate=0.01,momentum=0.9,decay=(0.01/25),nesterov=False)
 
model.compile(loss='categorical_crossentropy',
  optimizer=sgd,
  metrics=['accuracy'])


# In[21]:


model.summary()


# In[22]:


model.fit(train_X,train_Y,
    validation_data=(test_X,test_Y),
    epochs=30,batch_size=32)


# In[23]:


_,acc=model.evaluate(test_X,test_Y)
print(acc*100)


# In[24]:


model.save("model1_cifar_10epoch.h5")


# In[36]:


results={
   0:'aeroplane',
   1:'automobile',
   2:'bird',
   3:'cat',
   4:'deer',
   5:'dog',
   6:'frog',
   7:'horse',
   8:'ship',
   9:'truck'
}
from PIL import Image
import numpy as np
im1=Image.open(r"C:\Users\sbabu5\Documents\ship.jpg")
im2=Image.open(r"C:\Users\sbabu5\Documents\horse.jpg")
# the input image is required to be in the shape of dataset, i.e (32,32,3)
 
im1=im1.resize((32,32))
im1=np.expand_dims(im1,axis=0)
im1=np.array(im1)
pred=model.predict([im1])
pred=np.argmax(pred)
print(pred,results[pred])

im2=im2.resize((32,32))
im2=np.expand_dims(im2,axis=0)
im2=np.array(im2)
pred=model.predict([im2])
pred=np.argmax(pred)
print(pred,results[pred])


# In[ ]:





# In[ ]:




