from builtins import range
from builtins import object
import numpy as np

from neuralnet.layers import *
from neuralnet.layer_utils import *




class AutoEncoder(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a l2 norm loss function. For a network with L layers,
    the architecture will be

    {affine - relu} x (L - 1) - affine

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=28*28, output_dim=28*28, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - output_dim: An integer giving the size of the output.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.input_dim = input_dim
        self.output_dim = output_dim

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.params['W1'] = np.random.normal(0,weight_scale,(input_dim,hidden_dims[0]))
        self.params['b1'] = np.zeros((hidden_dims[0],))
        for i in range(self.num_layers-2):
            self.params['W'+str(i+2)] = np.random.normal(0,weight_scale,(hidden_dims[i],hidden_dims[i+1]))
            self.params['b'+str(i+2)] = np.zeros((hidden_dims[i+1],))
        self.params['W'+str(self.num_layers)] = np.random.normal(0,weight_scale,(hidden_dims[-1],output_dim))
        self.params['b'+str(self.num_layers)] = np.zeros((output_dim,))

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

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        hidden_num = self.num_layers - 1
        scores = X
        cache_history = []
        L2reg = 0
        for i in range(hidden_num):
            scores, cache = affine_relu_forward(scores, self.params['W%d' % (i + 1)],self.params['b%d' % (i + 1)])
            cache_history.append(cache)
            L2reg += np.sum(self.params['W%d' % (i + 1)] ** 2)
        i += 1
        scores, cache = affine_forward(scores, self.params['W%d' % (i + 1)],self.params['b%d' % (i + 1)])
        cache_history.append(cache)
        L2reg += np.sum(self.params['W%d' % (i + 1)] ** 2)
        L2reg *= 0.5 * self.reg

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using l2_norm_loss, and make sure that grads[k] holds the      #
        # gradients for self.params[k]. Don't forget to add L2 regularization!     #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        N = y.shape[0]
        loss, dout = l2_norm_loss(scores.reshape((N,self.output_dim)), y.reshape((N,self.output_dim)))
        loss += L2reg
        dout, grads['W%d' % (i + 1)], grads['b%d' % (i + 1)] = affine_backward(dout, cache_history.pop())
        grads['W%d' % (i + 1)] += self.reg * self.params['W%d' % (i + 1)]
        i -= 1
        while i >= 0:
            dout, grads['W%d' % (i + 1)], grads['b%d' % (i + 1)] = affine_relu_backward(dout, cache_history.pop())
            grads['W%d' % (i + 1)] += self.reg * self.params['W%d' % (i + 1)]
            i -= 1
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

