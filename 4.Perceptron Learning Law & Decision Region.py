import numpy as np
import matplotlib.pyplot as plt 

def perceptron(x,w,b):
    return np.sign(np.dot(x,w)+b)

def perceptron_learning(x,y,eta,epochs):
    w = np.zeros(2)
    b = 0

    for epoch in range(epochs):
        for i in range(x.shape[0]):
            y_pred = perceptron(x[i],w,b)

            if y_pred != y[i]:
                w += eta * y[i] * x[i]
                b += eta * y[i]
    
    return w,b

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([-1,-1,-1,1])

w,b = perceptron_learning(x,y,eta=1,epochs=10)

x_min, x_max = x[:,0].min() - 1,x[:,0].max() + 1
y_min, y_max = x[:,1].min() - 1,x[:,1].max() + 1

xx,yy = np.meshgrid(np.arange(x_min,x_max,0.01),np.arange(y_min,y_max,0.01))
z = np.array([perceptron(np.array([x,y]),w,b) for x,y in np.c_[xx.ravel(),yy.ravel()]])
z = z.reshape(xx.shape)

# colors = ['red' if label==-1 else 'green' for label in y]

plt.contourf(xx,yy,z,cmap=plt.cm.coolwarm,alpha=0.8)
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.coolwarm)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title('Perceptron Decision regions')
plt.show()