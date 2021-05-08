from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    
    N, D = X.shape
    C = W.shape[1]
    
    # Initialize the loss and gradient to zero.
    f = np.zeros((N, C))
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # X*W + b = prediction
    # y: answer value
        
    batch_size = X.shape[0]
    
    # forward
    for i in range(N):
        for j in range(C):
            for k in range(D):
                f[i, j] += X[i, k] * W[k, j]
        f[i, :] = np.exp(f[i, :])
        f[i, :] /= np.sum(f[i, :])
        
    loss = -np.sum(np.log(f[np.arange(batch_size), y])) / batch_size
    # cross entropy loss
    loss += 0.5 * reg * np.sum(W**2)
    
    # backward
    f[np.arange(batch_size), y] -= 1
    
    for i in range(N):
        for j in range(D):
            for k in range(C):
                dW[j, k] += X[i, j] * f[i, k] 

    dW /= batch_size
    dW += reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    N, D = X.shape
    C = W.shape[1]
    
    batch_size = X.shape[0]
    f = np.zeros((N, C))
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    score = np.dot(X, W)
    f = np.exp(score)
    f /= np.sum(f, axis=1, keepdims=True)
    loss -= np.sum(np.log(f[np.arange(batch_size), y]))
    loss /= batch_size
    loss += 0.5 * reg * np.sum(W**2)
    
    bw = np.zeros((N, C))
    bw[np.arange(batch_size), y] = f[np.arange(batch_size), y] - 1
    dW = np.dot(X.T, bw) / batch_size
    # cross entropy loss
    dW += reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
