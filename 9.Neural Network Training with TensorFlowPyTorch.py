import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

df = load_breast_cancer()
xtrain,xtest,ytrain,ytest = train_test_split(df.data,df.target,test_size=0.2,random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(xtrain)
x_test = sc.transform(xtest)
model = tf.keras.models.Sequential([tf.keras.layers.Dense(1,activation='sigmoid',input_shape=(x_train.shape[1],))])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train,ytrain,epochs=5)

ypred = model.predict(x_test)
test_loss,test_accuracy = model.evaluate(x_test,ytest)
print('accuracy is = ',test_accuracy)