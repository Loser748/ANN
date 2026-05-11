import numpy as np
x = np.array([[1,1,1,-1],[-1,-1,1,1]])
y = np.array([[1,-1],[-1,1]])
w = np.dot(y.T,x)

def bam(x):
    y = np.sign(np.dot(w,x))
    return y

x_test = np.array([1,-1,-1,-1])
y_test = bam(x_test)

print("Inputs = ",x_test)
print("output = ",y_test)