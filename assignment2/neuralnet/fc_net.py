import numpy as np
from neuralnet.layers import affine_forward, softmax_loss, affine_backward
from neuralnet.layer_utils import affine_relu_forward, affine_relu_backward


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - relu} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 reg=0.0, weight_scale=1e-2, dtype=np.float32):
        """

        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
          the network should not use dropout at all.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be
          performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        """
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        #######################################################################
        # TODO: Initialize the parameters of the network, storing all values i
        # the self.params dictionary. Store weights and biases for the first
        # layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should
        # be #
        # initialized from a normal distribution with standard deviation equal
        # to  #
        # weight_scale and biases should be initialized to zero.
        #
        #######################################################################
        if type(hidden_dims) != list:
            raise ValueError('hidden_dim has to be a list')

        self.L = len(hidden_dims) + 1
        self.N = input_dim
        self.C = num_classes
        dims = [self.N] + hidden_dims + [self.C]
        Ws = {
                'W' + str(i + 1):
                weight_scale * np.random.randn(
                    dims[i], dims[i + 1]
                ) for i in range(
                    len(dims) - 1
                )
        }
        b = {
            'b' + str(i + 1):
            np.zeros(dims[i + 1]) for i in range(len(dims) - 1)
        }

        self.params.update(b)
        self.params.update(Ws)

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        #######################################################################
        # TODO: Implement the forward pass for the fully-connected net,
        # computing  #
        # the class scores for X and storing them in the scores variable.
        #######################################################################

        # We are gonna store everythin in a dictionnary hidden
        hidden = {}
        hidden['h0'] = X.reshape(X.shape[0], np.prod(X.shape[1:]))

        for i in range(self.L):
            idx = i + 1
            # Naming of the variable
            w = self.params['W' + str(idx)]
            b = self.params['b' + str(idx)]
            h = hidden['h' + str(idx - 1)]

            # Computing of the forward pass.
            # Special case of the last layer (output)
            if idx == self.L:
                h, cache_h = affine_forward(h, w, b)
                hidden['h' + str(idx)] = h
                hidden['cache_h' + str(idx)] = cache_h
            # For all other layers
            else:
                h, cache_h = affine_relu_forward(h, w, b)
                hidden['h' + str(idx)] = h
                hidden['cache_h' + str(idx)] = cache_h

        scores = hidden['h' + str(self.L)]

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}

        #######################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store
        # the #
        # loss in the loss variable and gradients in the grads dictionary.
        # Compute #
        # data loss using softmax, and make sure that grads[k] holds the
        # gradients #
        # for self.params[k]. Don't forget to add L2 regularization!
        #######################################################################

        # Computing of the loss
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0
        for w in [self.params[f] for f in self.params.keys() if f[0] == 'W']:
            reg_loss += 0.5 * self.reg * np.sum(w * w)

        loss = data_loss + reg_loss

        # Backward pass

        hidden['dh' + str(self.L)] = dscores
        for i in range(self.L)[::-1]:
            idx = i + 1
            dh = hidden['dh' + str(idx)]
            h_cache = hidden['cache_h' + str(idx)]
            if idx == self.L:
                dh, dw, db = affine_backward(dh, h_cache)
                hidden['dh' + str(idx - 1)] = dh
                hidden['dW' + str(idx)] = dw
                hidden['db' + str(idx)] = db

            else:
                dh, dw, db = affine_relu_backward(dh, h_cache)
                hidden['dh' + str(idx - 1)] = dh
                hidden['dW' + str(idx)] = dw
                hidden['db' + str(idx)] = db

        # w gradients where we add the regulariation term
        list_dw = {
            key[1:]: val + self.reg * self.params[key[1:]]
            for key, val in hidden.items() if key[:2] == 'dW'
        }
        # Paramerters b
        list_db = {
            key[1:]: val
            for key, val in hidden.items() if key[:2] == 'db'
        }

        grads = {}
        grads.update(list_dw)
        grads.update(list_db)
        return loss, grads
