# tinytensor

![tinytensor - the little tensor + autograd engine that could.](tinytensor_engine.png)

tinytensor is the tiny tensor + autograd engine that could. 

It's [micrograd](https://github.com/karpathy/micrograd) but with tensors (of any rank).

micrograd beautifully illustrates how autograd works. But, it only works on scalar values. This is OK for simple MLPs. But it's difficult to use for more complex networks.

tinytensor attempts to keep the simplicity of micrograd while operating on tensors of higher rank as needed for deep learning. Right now, it works for CNNs. The project's goal is to eventually demonstrate RNNs & Transformers while staying extremely simple. 

## Who/What tinytensor is for
tinytensor is for education. 

It is for people who want to develop an inutitive understanding of how neural networks work.

It works on the same principles as "real" libraries like [pytorch](https://github.com/pytorch/pytorch), [tensorflow](https://github.com/tensorflow/tensorflow), and [tinygrad](https://github.com/tinygrad/tinygrad). But it's super simple.

As of tinytensor's alpha release, pytorch & tensorflow are ~200k - 3M lines of code, respectively. In contrast, the magic of tinytensor happens in about 400 lines of code (plus some basic unit tests & demos).

If you're OK at python programming, are new to neural networks/deep learning, and want to get a gist of how the "real" libraries work under the hood, this library is for you.

## Who/What tinytensor is NOT for
tinytensor is NOT for production-grade deep learning tasks.

Right now, tinytensor is slowww. Like the little engine that could, it slowly climbs the mountain (or rather, descends the gradient) of your neural network model. If you listen very closely to the fans spinning in your computer while tinytensor is training, you'll hear a whisper of, "I think I can, I think I can..." (Yet, like that little engine, it will eventually succeed. Without any tuning other than defining basic kernel sizes,  a tinytensor CNN can comfortably achieve 95-99% accuracy on [MNIST](https://en.wikipedia.org/wiki/MNIST_database)).

It lacks many advanced features of the real libraries. 

There are plenty of sharp edges that the authors have not yet found and/or documented yet (although we'll keep documenting and/or fixing these over time).

As of the alpha release, there are still asserts everywhere which is not safe for production use.

In short, if you're a serious neural network researcher, data scientist, or engineer working on a commercial project, tinytensor is probably not useful for you.

While it will eventually be able to build & run almost any neural network, it is not intended to compete with the real libraries. At least not now.

## Installation
tinytensor is not yet packaged for distribution. If the community wants it, we'll do it.

For now:
```bash
git clone https://github.com/username/repository-name.git
```

Modify & copy files into your project directly as needed.

## Usage
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
The bare bones of a model to solve MNIST. There are other packages to import & variables to define before this.
```python
#...
conv = Conv2d(kernel_shape,image_shape)
pool = MaxPool(pooling_kernel_shape,conv.output_shape())
softmax = Softmax(pool.output_shape(),num_classes)
layers = [conv,pool,softmax]

for i in range(number_samples):
    conv_out = layers[0](X[i])
    pool_out = layers[1](conv_out)
    softmax_out = layers[2](pool_out)
    data_loss = categorical_cross_entropy(y[i],softmax_out)
    data_loss.backward()
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
* Add additional loss functions, optimizers, regularization, normalization functions as appropriate & useful.
* Try to improve performance with some refactoring, parallelization, jit compilation, etc. as appropriate & useful while keeping things simple.
* (Maybe) Package the repo for distribution.

Unless there are serious innovations that radically improve performance, seamlessly enable execution across various hardware platforms, etc. while keeping things super simple, that will be it. As of now, the author does not see a need for another production-grade deep learning library.

If you want a library that:
* is full-featured
* is able to run performantly across platforms
* is still quite simple
* has the potential to democratize production-grade AI

...[tinygrad](https://github.com/tinygrad/tinygrad) is the future.

## License
MIT