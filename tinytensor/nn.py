from tinytensor.engine import Tensor
from typing import Tuple
import numpy as np
from functools import reduce
np.random.seed(1337)

class Layer:
    def __init__(self, W:Tensor, B:Tensor = None):
        self.W = W
        self.B = B
    

class Dense(Layer):
    def __init__(self,inputs: int,outputs: int,activation: str,W: Tensor = None, B:Tensor = None):
        
        W_input = Tensor(np.random.randn(outputs,inputs) / (outputs * inputs)) if W is None else W
        B_input = Tensor(np.zeros((outputs,1))) if B is None else B

        super().__init__(W_input,B_input)
        
        self.activation = activation
        self.inputs = inputs
        self.outputs = outputs

    def __call__(self,x) -> Tensor:

        def _dense(x:Tensor, W:Tensor, B:Tensor,activation: str) -> Tensor:

            Z = W.dot(x) + B

            if activation == "relu":
                A = Z.relu()
            elif activation == "sigmoid":
                A = Z.sigmoid()
            elif activation == "linear":
                A = Z

            return A
        
        return _dense(x,self.W,self.B,self.activation)
    
class Conv2d:
    def __init__(self,kernel_shape: Tuple[int,int,int,int],image_shape: Tuple[int,int,int,int],W: Tensor = None):
        
        if W is not None:
            assert W.shape == kernel_shape
        self.W = Tensor(np.random.standard_normal(kernel_shape) / (kernel_shape[1] ** 2)) if W is None else W
        self.kernel_shape = kernel_shape
        self.image_shape = image_shape

    def __call__(self,image: Tensor):

        def get_image_key(key,kernel_weights = self.W):
            i,j = key[1],key[2]
            return slice(None,None,None),slice(i,i + kernel_weights.shape[1],None),slice(j,j + kernel_weights.shape[2],None),slice(None,None,None)
        
        output_keys = [(0,j,k,0) for j in range(self.output_shape()[1]) for k in range(self.output_shape()[2])]
        image_slices = (image.get_slice(get_image_key(key)) for i,key in enumerate(output_keys))
        
        def _convolve(packed_input: tuple,W: Tensor = self.W) -> Tensor:
            key, image_slice = packed_input[0],packed_input[1] #unpack input
            return (image_slice.tile(W.shape) * W).sum(axis = (1,2,3),keepdims = True).pad(self.output_shape(),key) #the convolution

        out = reduce(lambda acc,i: acc + _convolve(i),zip(output_keys,image_slices),Tensor(np.zeros(self.output_shape())))
        return out.reshape((out.shape[0],out.shape[1],out.shape[2]))

    def output_shape(self) -> Tuple:
        return (self.kernel_shape[0],self.image_shape[1] - self.W.shape[1] + 1,self.image_shape[2] - self.W.shape[2] + 1,1)
    
    def grads(self):
        return {"W":self.W.grad,"B":None}
    
class MaxPool:
    def __init__(self,kernel_shape: Tuple[int,int,int,int],input_shape: Tuple[int,int,int,int]):
        self.kernel_shape = kernel_shape
        self.input_shape = input_shape

    def __call__(self,input: Tensor) -> Tensor:
        
        def get_input_keys(key,kernel_shape = self.kernel_shape):
            i,j = key[1],key[2]
            return slice(None,None,None),slice(i*kernel_shape[1],i*kernel_shape[1] + kernel_shape[1],None),slice(j*kernel_shape[2],j*kernel_shape[2] + kernel_shape[2],None)
        
        output_keys = [(0,j,k) for j in range(self.output_shape()[1]) for k in range(self.output_shape()[2])]
        input_slices = (input.get_slice(get_input_keys(key)) for i,key in enumerate(output_keys))

        def _pool(packed_input: Tuple[Tuple,Tensor]) -> Tensor:
            key,input_slice = packed_input[0],packed_input[1] #unpack input
            return input_slice.max(axis = (1,2),keepdims = True).pad(self.output_shape(),key) #max pool

        return reduce(lambda acc,i: acc + _pool(i),zip(output_keys,input_slices),Tensor(np.zeros(self.output_shape())))

    def output_shape(self) -> tuple:
        return (self.kernel_shape[0],self.input_shape[1] // self.kernel_shape[1],self.input_shape[2] // self.kernel_shape[2])
    
    def grads(self):
        return {"W":None,"B":None}

class Softmax:
    def __init__(self,input_shape: Tuple,outputs: int,W: Tensor = None,B: Tensor = None):
        self.input_shape = input_shape
        self.outputs = outputs
        self.input_size = reduce(lambda acc,s: acc * s,input_shape,1)
        self.W = Tensor(np.random.randn(self.input_size,outputs) / self.input_size) if W is None else W
        self.B = Tensor(np.zeros((1,outputs))) if B is None else B
        self.output_shape = (1,outputs)

    def __call__(self,input: Tensor):
        input_flattened = input.reshape((1,self.input_size))
        dot = input_flattened.dot(self.W)
        Z = dot + self.B
        A = Z.exp()
        D = A.sum(keepdims = True)

        def get_input_keys(key):
            i,j = key[0],key[1]
            return slice(None,None,None),slice(j,j+1,None)

        output_keys = [(0,i) for i in range(A.shape[1])]
        A_slices = (A.get_slice(get_input_keys(key)) for i,key in enumerate(output_keys))
        
        def _softmax(packed_input: Tuple[int,Tensor],D:Tensor = D,output_shape: Tuple = A.shape) -> Tensor:
            key,input_slice = packed_input[0],packed_input[1] #unpack input
            return (input_slice / D).pad(self.output_shape,key) #doing the softmax division (and padding the result)

        return reduce(lambda acc,i: acc + _softmax(i),zip(output_keys,A_slices),Tensor(np.zeros(A.shape)))
    
    def grads(self):
        return {"W":self.W.grad,"B":self.B.grad}
    
class Model:
    def __init__(self,layers = []):
        self.layers = layers

    def __call__(self,x: Tensor) -> Tensor:

        def _forward(x: Tensor,layers: list) -> Tensor:
            return reduce(lambda acc, layer: layer(acc),layers,x)
        
        return _forward(x,self.layers)
    
def array_to_Tensor_list(array: np.ndarray) -> list:
        return [Tensor(r.reshape(-1,1)) for r in array]

def mse(yhats: list, y: list) -> Tensor:

    #calculate losses
    losses = [yi - yhat for yhat,yi in zip(yhats,y)]
    squared_losses = [loss ** 2 for loss in losses]
    sum_squared_losses = reduce(lambda acc,ys: acc + ys,squared_losses,Tensor(np.array([[0.0]])))
    data_loss = sum_squared_losses * Tensor(np.array([[1 / len(losses)]]))

    return data_loss

def svm_max_margin(yhats: list, y: list) -> Tensor:
    one = Tensor(np.ones_like(yhats[0].data))
    losses = [one + -yi*yhat for yhat, yi in zip(yhats, y)]
    relu_losses = [loss.relu() for loss in losses]
    sum_relu_losses = reduce(lambda acc,ys: acc + ys,relu_losses,Tensor(np.array([[0.0]])))
    data_loss = sum_relu_losses * Tensor(np.array([[1 / len(losses)]]))
    return data_loss

def categorical_cross_entropy(y: Tensor,yhats: Tensor) -> Tensor:
    loss = y * yhats.log()
    return -loss.sum(keepdims = True)

def l2_reg(model: Model,alpha: int = 1e-4) -> Tensor:

    def square(layer: Dense) -> Tensor:
        squared_weights = layer.W ** 2
        summed_weights = squared_weights.sum(keepdims = True)
        x_alpha = summed_weights * Tensor(np.array([[alpha]]))
        return x_alpha
    
    def apply_square(acc: Tensor,layer: Dense) -> Tensor:
        return acc + square(layer)
    
    return reduce(lambda acc,layer: apply_square(acc,layer),model.layers,Tensor(np.array([[0]])))

def sgd(layer: Layer,grad: Tensor,learning_rate:float = 1e-2) -> Layer:
    updated_W = Tensor(layer.W.data - learning_rate * grad["W"]) if grad["W"] is not None else None
    updated_B = Tensor(layer.B.data - learning_rate * grad["B"]) if grad["B"] is not None else None
    if isinstance(layer,Dense):
        new_layer = Dense(layer.inputs,layer.outputs,layer.activation,updated_W,updated_B)
    if isinstance(layer,Conv2d):
        new_layer = Conv2d(layer.kernel_shape,layer.image_shape,updated_W)
    if isinstance(layer,MaxPool):
        new_layer = MaxPool(layer.kernel_shape,layer.input_shape)
    if isinstance(layer,Softmax):
        new_layer = Softmax(layer.input_shape,layer.outputs,updated_W,updated_B)

    return new_layer
    
if __name__ == "__main__":
    
    x_a = np.array([
                    [0,0],
                    [1,0],
                    [0,1],
                    [1,1]
                    ])

    y_a = np.array([
                    [0],
                    [1],
                    [1],
                    [0]
                    ])
    
    print(x_a.shape,y_a.shape)


    X = array_to_Tensor_list(x_a)
    y = array_to_Tensor_list(y_a)

    print(X[0].data)

    initial_model = Model()
    initial_model.layers.append(Dense(2,4,"relu"))
    initial_model.layers.append(Dense(4,1,"sigmoid"))

    def loss(model: Model,x: list,y: list) -> Tuple[Tensor,float]:
        
        #forward pass
        yhats = list(map(model,x))

        #loss functions
        data_loss = mse(yhats,y)
        total_loss = data_loss

        #calculate accuracy
        accuracies = [(yhat.data > 0.5) == (yi.data > 0.5) for yhat,yi in zip(yhats,y)]
        accuracy = sum(accuracies) / len(accuracies)

        return total_loss,accuracy
    
    def train(model: Model,X: list,y: list) -> list:
    
        total_loss,accuracy = loss(model,X,y)
        print(f"total_loss: {total_loss.data}; accuracy:{accuracy}")

        #backward
        total_loss.backward()

        #get grads
        grads = [layer.grads() for layer in model.layers]
        
        return grads

    def train_loop(model: Model,X: list,y: list,num_iterations: int) -> Model:
        if num_iterations == 0:
            return model
        else:
            #TODO: return grads here, then implement a standalone optimization function
            grads = train(model,X,y)
            new_layers = [sgd(layer,grad) for layer,grad in zip(model.layers,grads)]
            new_model = Model(layers = new_layers)

            return train_loop(new_model,X,y,num_iterations - 1)
    
    final_model = train_loop(initial_model,X,y,num_iterations = 10)

    #convolution layer test
    image = Tensor(np.ones((1,5,5,1)))
    weights = Tensor(np.ones((1,2,2,1)))

    conv = Conv2d((1,2,2,1),image.shape,W = weights,B = None)
    out = conv(image)
    out.backward()
    test = np.array([[[[1.],
         [2.],
         [2.],
         [2.],
         [1.]],

        [[2.],
         [4.],
         [4.],
         [4.],
         [2.]],

        [[2.],
         [4.],
         [4.],
         [4.],
         [2.]],

        [[2.],
         [4.],
         [4.],
         [4.],
         [2.]],

        [[1.],
         [2.],
         [2.],
         [2.],
         [1.]]]])
    
    assert np.array_equal(np.round(test,3),np.round(image.grad),3)

    #max pool layer test
    a = Tensor(np.ones((1,5,5,1)))
    a.data[0,2,3,0] = 2

    m = MaxPool((1,2,2,1),(1,5,5,1))
    out = m(a)
    out.backward()
    test = np.array([[[[1.],
         [1.]],

        [[1.],
         [2.]]]])
    assert out.shape == (1,2,2,1)
    assert np.array_equal(np.round(test,3),np.round(out.data,3))
        

