from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1] # This will give the number of rows is 1-d array 
    num_train = X.shape[0] # This will give the number of rows is 0-d array 
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                # loss의 gradient를 dW에 저장
                dW[:,j] += X[i].T
                dW[:,y[i]] += -X[i].T 


    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # loss function의 gradient를 dW에 저장한다.
    # loss 값을 갱신했던 for loop의 마지막 부분으로 가서 gradient를 dW에 저장하도록 한다.
    dW /= num_train
    dW += reg * W
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    
    # 앞의 for i in range(num_train) 부분을 vectorization 이용하여 줄여 보는 과정
    # 순서대로 한 줄씩 짜 보자
    
    # scores = X[i].dot(W)를 모든 i에 대하여 반복함
    # i는 num_train을 지칭하였고, num_train = X.shape[0] 
    # scores = X[num_train].dot(W)
    # f(x_i, W) 이런 느낌
    scores = np.dot(X, W)
    
    # correct_class_score = scores[y[i]]를 모든 i에 대하여 반복함
    correct_class_score = scores[np.arange(num_train), y]
    
    # for j in range(num_classes):
    #   if j == y[i]: 
    #       continue 
    #   margin = scores[j] - correct_class_score + 1
    #   if margin > 0: 
    #       loss += margin
    margin = np.maximum(scores - correct_class_score.reshape(num_train, 1) + 1, 0)
    # margin 값이 0보다 작다면 그냥 0으로 해 줌
    margin[np.arange(num_train), y] = 0
    # 정답 값은 그냥 0으로 해 줌
    
    # margin > 0이라면 loss 값에 margin 값을 더해 줘야 함
    loss = np.sum(margin) / num_train
    loss += 0.5 * reg * np.sum(W * W)
    
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # dW에다가 result를 저장해 줌
    # dW = np.zeros(W.shape)
    # dW /= num_train
    # dW += reg * W
    
    # 이 부분은 완벽하게 이해가 잘 가지 않음 ㅠ 복습하기
    
    tmp = np.zeros((num_train, num_classes))
    tmp[margin > 0] = 1
    tmp[range(num_train), list(y)] = 0
    tmp[range(num_train), list(y)] = -np.sum(tmp, axis=1)

    dW = (X.T).dot(tmp)
    dW = dW/num_train + reg*W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW