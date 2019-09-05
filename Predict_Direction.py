# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 19:30:25 2019

@author: MaxeDemon
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.enable_eager_execution()


f = open('E:\ML\mycode\ZQETF.csv')
df = pd.read_csv(f)

data = np.array(df[['Close','Open','High','Low','Volume']])


time_step = 5
rnn_unit = 10
batch_size = 50
input_size = 5
output_size = 1
lr = 0.0005
train_length = round(data.shape[0]*0.7)
train_data = data[0:train_length,:]
mean = np.mean(train_data,axis = 0)
std = np.std(train_data, axis = 0)
normalized_train_data = (train_data-mean)/std

whole_size = data.shape[0]
train_size = train_data.shape[0]
def get_batch(normalized_train_data, batch_size, time_step, whole_size):
    train_x, train_y = [],[]
    for i in range(batch_size):
        index = np.random.randint(0,whole_size - time_step-1)
        train_x.append(normalized_train_data[index:index+time_step,:])
        if  (normalized_train_data[index+time_step+1,0] > normalized_train_data[index+time_step+1,1]) |  (normalized_train_data[index+time_step+1,0] >  normalized_train_data[index+time_step,0]):
            train_y.append([1,0])
        else:
            train_y.append([0,1])
    return np.array(train_x), np.array(train_y)

test_data = data[train_length+1:whole_size,:]
normalized_test_data = (test_data-mean)/std
test_size = test_data.shape[0]

def get_test_data(normalized_test_data, time_step,whole_size):
    test_x = []
    test_y = []
    for i in range(whole_size - time_step - 1):
        x = normalized_test_data [i:i+time_step,:]
        test_x.append(x.tolist())
        if  (normalized_train_data[i+time_step+1,0] > normalized_train_data[i+time_step+1,1]) |  (normalized_train_data[i+time_step+1,0] >  normalized_train_data[i+time_step,0]):
            test_y.append([1,0])
        else:
            test_y.append([0,1])
    return np.array(test_x), np.array(test_y)




class rNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = 50)
        self.dense = tf.keras.layers.Dense(units = 2, activation='sigmoid')
        
    def call(self, inputs):
        batch_size, seq_length,features= tf.shape(inputs)
        state = self.cell.zero_state(batch_size = batch_size, dtype = tf.float64)
        for t in range(seq_length.numpy()):
            output, state = self.cell(inputs[:,t,:], state)
        output = self.dense(output)
        return output
    
    def predict(self,inputs):
        y = self(inputs)
        return y


model = rNN()
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
num_epoch = 5000
for i in range(num_epoch):
    X,y = get_batch(normalized_train_data, batch_size, time_step,train_size)
    with tf.GradientTape() as tape:
        
        y_hat = model(X)
        '''
        flag1 = tf.reshape(y_hat,[-1])-tf.reshape(X[:,time_step-1,0],[-1])
        flag2 = tf.reshape(y,[-1])-tf.reshape(X[:,time_step-1,0],[-1])
        flag = flag1*flag2
        weight = np.zeros(batch_size)
        for j in range(batch_size):
            if flag[j]>0:
                weight[j] = 0.5
            else:
                weight[j] = 5
        loss = tf.reduce_sum(weight*tf.abs(tf.reshape(y_hat,[-1])-tf.reshape(y,[-1])))
        '''
        loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.argmax(y, axis=1), logits=y_hat)
        if i%100 == 0:
            print("batch %d:loss %f" % (i, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars = zip(grads, model.variables))

test_x,test_y = get_test_data(normalized_test_data, time_step, test_size)

y_hat = model.predict(test_x)

true = tf.argmax(test_y, axis=1)
predict = tf.argmax(y_hat, axis = 1)

test_y = test_y * std[0]+mean[0]
y_hat = y_hat * std[0]+mean[0]



plt.figure()
plt.plot(list(range(len(test_y))), test_y, color='b')
plt.plot(list(range(len(y_hat))), y_hat,  color='r')
plt.show()


new_x = normalized_test_data[np.newaxis,test_size-time_step:test_size,:]
new_y = model.predict(new_x)
direction = new_y - new_x[0,time_step-1,0]