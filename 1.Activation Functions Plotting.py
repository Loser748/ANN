import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5,5,100)
plt.plot(x,1/(1+np.exp(-x)),label='sigmoid')
plt.plot(x,np.tanh(x),label='tanh')
plt.plot(x,np.maximum(0,x),label = 'ReLu')
plt.plot(x,x,label = 'identity')
plt.plot(x,np.exp(x)/np.sum(np.exp(x)),label = 'softmax')

plt.xlabel('input')
plt.ylabel('activation')
plt.title('activation function')
plt.legend()
plt.grid(True)
plt.show()