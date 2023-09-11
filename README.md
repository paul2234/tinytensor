# tinytensor

![tinytensor - the little tensor + autograd engine that could.](tinytensor_engine.png)

tinytensor is the tiny tensor + autograd engine that could. 

It's [micrograd](https://github.com/karpathy/micrograd) but with tensors (of any rank).

micrograd beautifully illustrates how autograd works. But, it only works on scalar values. This is OK for simple MLPs. But it's difficult to use for more complex networks.

tinytensor attempts to keep the simplicity of micrograd while operating on tensors of higher rank as needed for deep learning. Right now, it works for MLPs & CNNs. The project's goal is to eventually demonstrate RNNs & Transformers while staying extremely simple. 

## Who/What tinytensor is for
tinytensor is for education. 

With tinytensor, you can understand how each step of a serious neural network works without drowning in code or getting lost in a massive repo.

It works on the same principles as "real" libraries like [pytorch](https://github.com/pytorch/pytorch) and [tensorflow](https://github.com/tensorflow/tensorflow). But it's super simple.

As of tinytensor's alpha release, pytorch & tensorflow are ~200k - 3M lines of code, respectively. In contrast, the magic of tinytensor happens in about 400 lines of code (plus some basic unit tests & demos).

If you're OK at python programming, are new to neural networks/deep learning, and want to get a gist of how the "real" libraries work under the hood, this library is for you.

## Who/What tinytensor is NOT for
tinytensor is NOT for production-grade deep learning tasks.

Right now, tinytensor is **slowwwwww**. Especially for models like CNNs that require striding/pooling over inputs.

It lacks many advanced features of the real libraries. 

There are plenty of sharp edges that the authors have not yet found and/or documented yet (although this will get better over time).

As of the alpha release, there are still asserts everywhere which is not safe for production use.

In short, if you're a serious neural network researcher, data scientist, or engineer working on a commercial project, tinytensor will not be not useful for you.

While it will eventually be able to build & run almost any neural network, it is not intended to compete with the real libraries.

## Installation
tinytensor is not yet packaged for easy distribution. If the community wants it, we'll do it.

For now:
```bash
git clone https://github.com/paul2234/tinytensor.git
```

Modify & copy files into your project directly as needed.

## Usage
tinytensor tries to follow & encourage functional programming principles as much as possible, though we haven't followed this for backpropagation yet.

### Tensor operations & autograd
```python
#adding two Tensors
A = Tensor(np.array([[1,1],[1,1]]))
B = Tensor(np.array([[1,2],[3,4]]))

C = A + B
D = C + A
D.backward() #backpropagation of derivatives with respect to D
print(A.grad) #dD/dA; prints numpy array of  [[2,2],[2,2]]
```
### Combining tensor operations into neural network layers
```python
# from nn.py
def _dense(x:Tensor, W:Tensor, B:Tensor,activation: str) -> Tensor:

    Z = W.dot(x) + B

    if activation == "relu":
        A = Z.relu()
    elif activation == "sigmoid":
        A = Z.sigmoid()
    elif activation == "linear":
        A = Z

    return A
```

### Combining multiple layers
Below is an excerpt for a model training on MNIST. There are other packages to import & variables to define before this.
```python
#...
conv = Conv2d(kernel_shape,image_shape)
pool = MaxPool(pooling_kernel_shape,conv.output_shape())
softmax = Softmax(pool.output_shape(),num_classes)
layers = [conv,pool,softmax]

for i in range(number_samples):
    #forward pass through each layer
    conv_out = layers[0](X[i])
    pool_out = layers[1](conv_out)
    softmax_out = layers[2](pool_out)

    #calculating loss function
    data_loss = categorical_cross_entropy(y[i],softmax_out)

    #backpropatagion 
    data_loss.backward()

    #applying updated gradients
    grads = [layer.grads() for layer in layers]
    new_layers = [sgd(layer,grad,learning_rate) for layer,grad in zip(layers,grads)]
    layers = new_layers
    #...
#...
```
The notebook `mlp_moons.ipynb` demonstrates using an MLP model to solve moons categorization. This demo is almost identical to the demo in micrograd. 

The notebook `cnn_mnist.ipynb` demonstrates using a CNN to solve MNIST.

More demonstrations will be added as they are completed.

## What's next?
* Keep expanding the library until it supports RNN's and Transformers.
* Add additional loss functions, regularization functions, normalization functions, and optimizers as appropriate & useful.
* Refactoring to improve performance. (to a point; the priority will be keeping things simple)
* (Maybe) Package the repo for distribution.

If you want an alternative to PyTorch and TensorFlow that is:
* is full-featured
* is able to run performantly across platforms
* is still quite simple
* has the potential to democratize production-grade AI

...[tinygrad](https://github.com/tinygrad/tinygrad) is the future.

## License
MIT