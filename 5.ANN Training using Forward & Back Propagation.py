import numpy as np 
class neuralNetwork:
    def __init__(self):
        self.weights = np.random.rand(2,1)
        self.bias = np.random.rand(1)
    
    def train(self,x,y,epochs):
        for i in range(epochs):
            output = self.predict(x)
            error = y - output

            delta = error * output * (1-output)
            self.weights += np.dot(x.T,delta)
            self.bias += np.sum(delta)
        
    def predict(self,x):
        return 1/(1+np.exp(-(np.dot(x,self.weights)+self.bias)))
    
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[0],[0],[1]])

nn = neuralNetwork()
nn.train(x,y,epochs=1000)

test_data = np.array([[0,0],[0,1],[1,0],[1,1]])
predictions = nn.predict(test_data)
for x,prediction in zip(test_data,predictions):
    print(f"input:{x}, prediction:{prediction}")