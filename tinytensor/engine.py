import numpy as np
import time

class Tensor:
    
    def __init__(self,data: np.ndarray,_children = (),_op = "",parent_key = None,parent_shape = None):

        self.data = data
        self.grad = np.zeros_like(data,dtype = float)
        self.shape = self.data.shape

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.parent_key = parent_key
        self.parent_shape = parent_shape

    def __add__(self,other):

        assert isinstance(other,Tensor),"Both operands must be a Tensor."
        assert self.shape == other.shape,"Both operands must be the same shape"
        out = Tensor(np.add(self.data,other.data,dtype = float),(self,other),"+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
    
        out._backward = _backward

        return out
    
    def __mul__(self,other):
        assert isinstance(other,Tensor),"Both operands must be a tensor"
        assert self.shape == other.shape,"Both operands must be the same shape"
        out = Tensor(np.multiply(self.data,other.data),(self,other),"*")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
            _ = 1
        out._backward = _backward

        return out
    
    def divide(self,other):
        assert isinstance(other,Tensor),"Both perands must be a Tensor"
        assert self.shape == other.shape,"Both operands must be the same shape"
        out = Tensor(np.divide(self.data,other.data),(self,other),"divide")

        def _backward():
            self.grad += out.grad * -1 / (other.data ** 2)
            other.grad += out.grad * -1 / (self.data ** 2)

        out._backward = _backward

        return out

    def dot(self,other):
        assert isinstance(other,Tensor),"Both operands must be a tensor"
        assert self.shape[1] == other.shape[0],"Incompatible shapes for dot product"
        out = Tensor(np.dot(self.data,other.data),(self,other),"dot")

        def _backward():
            self.grad += out.grad.dot(other.data.T)
            other.grad += self.data.T.dot(out.grad)

        out._backward = _backward

        return out
    
    def __pow__(self,other):
        assert isinstance(other,int) or isinstance(other,float)
        out = Tensor(self.data ** other, (self,),"**")

        def _backward(other = other):
            self.grad += out.grad * (other * np.power(self.data,(other - 1)))

        out._backward = _backward

        return out
    
    def get_slice(self,key):

        out = Tensor(self.data[key],(self,),"get_slice",parent_key=key)

        def _backward(key = key):
            self.grad[key] += out.grad

        out._backward =_backward

        return out
    
    def sum(self,**kwargs):

        out = Tensor(np.sum(self.data,**kwargs),(self,),"sum")

        def _backward():
            self.grad = self.grad + np.broadcast_to(out.grad,self.grad.shape)
            _ = 1
        out._backward = _backward

        return out
    
    def max(self,**kwargs):
        epsilon = np.random.standard_normal(self.data.shape) * 1e-10
        out = Tensor(np.max(self.data + epsilon,**kwargs),(self,),"max")
    
        def _backward(epsilon = epsilon):
            #TODO - I don't know how robust this is for uses outside of MaxPool layers
            max_values = out.data
            max_indices = np.argwhere(self.data + epsilon == max_values)
            self_index = tuple(max_indices[:,i] for i in range(len(max_values.shape)))
            self.grad[self_index] += out.grad.reshape(-1,)
            _ = 1
        out._backward = _backward

        return out
    
    def pad(self,out_shape: tuple,index_location: tuple):
        
        assert len(out_shape) == len(index_location)
        assert len(out_shape) == len(self.shape)
        for i,j in zip(self.shape,out_shape):
            assert i <= j,"all dimensions of array to be padded must be less than or equal to those of the desired output shape"

        pads = tuple((index_location[i],out_shape[i] - self.shape[i] - index_location[i]) for i in range(len(out_shape)))
        key = tuple(slice(index_location[i],index_location[i] + self.shape[i],None) for i in range(len(self.shape)))

        out = Tensor(np.pad(self.data,pads),(self,),"pad",parent_shape=self.data.shape)
        
        def _backward(key = key):
            self.grad += out.grad[key]
            _ = 1
        out._backward = _backward

        return out
    
    
    def tile(self,new_shape: tuple):
        
        output_tuple = tuple(n - s + 1 for s,n in zip(self.data.shape,new_shape))
        out = Tensor(np.tile(self.data,output_tuple),(self,),"resize",parent_shape=self.data.shape)

        def _backward(output_tuple = output_tuple):
            axis_tuple = tuple(i for i,t in enumerate(output_tuple) if t > 1)
            self.grad += np.sum(out.grad,axis = axis_tuple,keepdims=True)
            _ = 1
        out._backward = _backward

        return out
    
    def transpose(self):

        out = Tensor(self.data.T,(self,),"transpose",parent_shape=self.shape)

        def _backward():
            self.grad = np.resize(out.grad,out.parent_shape)
        
        out._backward = _backward

        return out
    
    def reshape(self,new_shape: tuple):
        assert isinstance(new_shape,tuple)
        out = Tensor(np.reshape(self.data,new_shape),(self,),"reshape")

        def _backward(parent_shape = self.shape):
            self.grad = np.reshape(out.grad,parent_shape)

        out._backward = _backward

        return out
    
    def flatten(self):

        out = Tensor(self.data.flatten(),(self,),"flatten")

        def _backward(parent_shape = self.shape):
            self.grad = np.reshape(out.grad,parent_shape)

        out._backward = _backward

        return out
    
    def clip(self,epsilon: float):

        out = Tensor(np.clip(self.data,epsilon,1 - epsilon),(self,),"clip")

        def _backward():
            self.grad += out.grad

        out._backward = _backward

        return out
     
    def exp(self):

        out = Tensor(np.exp(np.clip(self.data,-30,30)),(self,),"exp")

        def _backward():
            self.grad += out.grad * np.exp(np.clip(self.data,-30,30))
            _ = 1
        out._backward = _backward

        return out
    
    def log(self):
        epsilon = 1e-10
        out = Tensor(np.log(np.clip(self.data,epsilon,1-epsilon)),(self,),"log")

        def _backward():
            self.grad += out.grad * 1 / np.clip(self.data,epsilon,1-epsilon)

        out._backward = _backward

        return out
    
    def sigmoid(self):

        def _sigmoid(z):
           return 1/(1+np.exp(-z))
        
        out = Tensor(_sigmoid(np.clip(self.data,-30,30)),(self,),"sigmoid")

        def _backward():
            self.grad += out.grad * _sigmoid(np.clip(self.data,-30,30)) * (1 - _sigmoid(np.clip(self.data,-30,30)))
        
        out._backward = _backward

        return out
    
    def relu(self):

        out = Tensor(np.where(self.data > 0,self.data,0),(self,),"relu")

        def _backward():
            self.grad += out.grad * np.where(out.data > 0,1,0)

        out._backward = _backward

        return out
    
    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad += np.ones_like(self.grad,dtype = float)
        for v in reversed(topo):
            v._backward()
    
    def __radd__(self, other): # other + self
        return self + other
    
    def __neg__(self):
        return self * Tensor(np.ones_like(self.data) * -1)
    
    def __sub__(self,other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

if __name__ == "__main__":

    #addition
    A = Tensor(np.random.randn(64,128))
    B = Tensor(np.random.randn(64,128))

    C = A + B
    D = C + A
    D.backward()
    assert np.allclose(2 * np.ones_like(A.grad),A.grad,atol = 1e-4)

    #addition
    A = Tensor(np.array([[1,1],[1,1]]))
    B = Tensor(np.array([[1,2],[3,4]]))

    C = A + B
    D = C + A
    D.backward()
    print(A.grad) #prints numpy array of [[2,2],[2,2]]

    #multiplication
    A = Tensor(2 * np.ones((4,4)))
    B = Tensor(np.ones((4,4)) * 3)
    C = A * B
    D = C + A
    E = D - B
    E.backward()
    assert np.array_equal(np.round(np.ones_like(B.grad),3),np.round(B.grad,3))

    #power
    A = Tensor(2 * np.ones((4,4)))
    B = A ** 2
    B.backward()
    assert np.array_equal(np.round(A.grad,3),np.round(4 * np.ones((4,4)),3))

    A = Tensor(np.random.randn(4,1))
    B = Tensor(np.random.randn(4,1))
    assert np.array_equal(np.round(((A - B) ** 2).data,4),np.round(((B - A) ** 2).data,4))

    #sum
    A = Tensor(np.ones((3,3)))
    B = A.sum(keepdims = True)
    B.backward()
    assert np.array_equal(A.grad,np.ones((3,3)))

    #get slice
    A = Tensor(np.ones((3,3)))
    key = (slice(2,3,None),slice(None,None,None))
    B = A.get_slice(key)
    B.backward()
    assert np.array_equal(A.grad,np.array([[0,0,0],[0,0,0],[1,1,1]]))
    
    #pad
    A = Tensor(np.ones((1,2)))
    out_shape = (3,3)
    index_location = (1,0)
    B = A.pad(out_shape,index_location)
    B.backward()

    assert np.array_equal(np.round([[0,0,0],[1,1,0],[0,0,0]],3),np.round(B.data,3))
    assert np.array_equal(np.round([[1,1]],3),np.round(A.grad,3))

    #tile
    A = Tensor(np.ones((4,1)))
    new_shape = (4,4)
    B = A.tile(new_shape)
    B.backward()
    
    assert np.array_equal(np.array([[4],[4],[4],[4]]),A.grad)

    #max
    A = Tensor(np.ones((1,3,3,2)))
    A.data[0,1,2,1] = 2

    B = A.max(keepdims = True)
    B.backward()
    test = np.array([[[[0., 0.],
         [0., 0.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.],
         [0., 1.]],

        [[0., 0.],
         [0., 0.],
         [0., 0.]]]])
    
    assert np.array_equal(np.round(A.grad,3),np.round(test,3))
    assert np.array_equal(np.round(A.grad[0,1,2,1],3),np.round(test[0,1,2,1],3))
    _ = 1

    C = Tensor(np.ones((2,3,3,1)))
    C.data[0,1,1,0] = 2
    C.data[1,2,2,0] = 3

    D = C.max(axis = (1,2),keepdims = True)
    D.backward()

    test = np.array([[
        [0,0,0],
        [0,1,0],
        [0,0,0]
    ],[[0,0,0],
        [0,0,0],
        [0,0,1]]]).reshape(2,3,3,1)
    
    assert np.array_equal(test,C.grad)

    #reshape
    A = Tensor(np.random.randn(1,3,3,2))
    B = A.flatten()
    B.backward()

    assert B.shape == (18,)

    A = Tensor(np.random.randn(1,3,3,2))
    B = A.reshape((3,3,2)) 
    B.backward()

    assert B.shape == (3,3,2)
    assert np.array_equal(A.grad,np.ones_like(A.data))

