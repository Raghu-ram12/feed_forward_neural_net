import numpy as np 

class layer:

    def __init__(self,input_dims,output_dims,activation="relu",alpha=0.01):


        initialization_factor=0.01

        if activation in ["relu","leaky-relu"]:
            initialization_factor=np.sqrt(2/input_dims) 

        elif activation in ["sigmoid","tanh"]:
            initialization_factor=np.sqrt(1/input_dims)


        self.Weight=np.random.randn((input_dims,output_dims))*initialization_factor  

        self.bias=np.zeros((1,output_dims))  
        self.alpha=alpha 
        self.activation=activation 
        self.dw=None 
        self.db=None 
        self.A_prev=None 
        self.z=None 

    def tanh(self,z):

        numerator=np.exp(z)-np.exp(-z) 
        denominator=np.exp(z)+np.exp(-z) 
        return numerator/denominator 

    def tanh_prime(self,z):

        return 1-self.tanh(z)**2

    def relu(self,z):
        return np.maximum(0,z) 

    def relu_prime(self,z): 

        return np.where(z>=0,1,0) 
        
    def leaky_relu(self,z): 

        return np.maximum(self.alpha*z,z) 

    def leaky_relu_prime(self,z):

        return np.where(z>=0,1,self.alpha)

    def sigmoid(self,z):

        return 1/(1+np.exp(-z))

    def sigmoid_prime(self,z):

        return self.sigmoid(z)*(1-self.sigmoid(z)) 

    
    def forward_pass(self,A_prev): 

        self.A_prev=A_prev 

        self.z=np.dot(A_prev,self.Weight)+self.bias  

        if self.activation=='relu':

            return self.relu(self.z) 

        elif self.activation=="leaky-relu":

            return self.leaky_relu(self.z) 

        elif self.activation=='tanh':

            return self.tanh(self.z) 

        elif self.activation=='sigmoid':
            
            return self.sigmoid(self.z) 

        else:

            return self.z

    def get_derivative(self):

        if self.activation=='relu':

            return self.relu_prime(self.z) 

        elif self.activation=="leaky-relu":

            return self.leaky_relu_prime(self.z) 

        elif self.activation=='tanh':

            return self.tanh_prime(self.z) 

        elif self.activation=='sigmoid':
            
            return self.sigmoid_prime(self.z) 

        else:

            return np.ones_like(self.z)
        
    def backward_pass(self,delta_prev): 

        delta=delta_prev*self.get_derivative() 

        m=self.A_prev.shape[0] 

        self.dw=np.dot(self.A_prev.T,delta) /m

        self.db=np.sum(delta,axis=0,keepdims=True)/m  

        return np.dot(delta,self.Weight.T) 



class NeuralNetwork:

    def __init__(self):

        self.layers=[]  
        self.output=None 

    
    def add_layer(self,input_dims,output_dims,activation="relu"): 

        self.layers.append(layer(input_dims,output_dims,activation=activation))  

    def forward_propagation(self,x):

        A_prev=x

        for l in self.layers:

            A_prev=l.forward_pass(A_prev) 
        
        self.output=A_prev  

        return self.output
    
    def calculate_error(self,actual_data):
        
        return self.output-actual_data 


    def backward_propagation(self,actual_data,learning_rate): 

        prev_delta=self.calculate_error(actual_data)  

        for lyr in self.layers[::-1]:

            prev_delta=lyr.backward_pass(prev_delta) 

            layer.dw=layer.dw-learning_rate*lyr.dw 
            layer.db=db-learning_rate*lyr.db 
    
    
    def train(self,X_train,Y_train,epochs=100,learning_rate=0.01):


        for _ in range(epochs):

            self.forward_propagation(X_train) 
            self.backward_propagation(Y_train,learning_rate) 


    def predict(self,x_test): 


        output=self.forward_propagation(x_test) 

        return output 
