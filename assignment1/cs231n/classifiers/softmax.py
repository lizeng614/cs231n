import numpy as np
from random import shuffle

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_train):
      score = X[i].dot(W)
      # Normalization trick to avoid numerical instability, per http://cs231n.github.io/linear-classify/#softmax
      score -= np.max(score)
      score_exp = np.exp(score)
      loss += -np.log(score_exp[y[i]]/np.sum(score_exp))  
  
  # to compute the derivitave for softmax got to http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/
      for j in range(num_class):
          dW[:,j] += (score_exp[j]/np.sum(score_exp) - (j == y[i]))*X[i]
          
  loss = loss/num_train + 0.5*reg*np.sum(W*W)
  dW = dW/num_train + reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  scores = X.dot(W)
  # this normalization has some problem try another way
  scores -= np.max(scores,axis=1,keepdims = True)
  #scores -= np.max(scores)
  scores_exp = np.exp(scores)
  
  Probs = scores_exp/np.sum(scores_exp,axis=1,keepdims =True)
  # the following two lines of code encouter a invilid problem:
  # loss = np.sum(-np.log(Probs[np.arange(num_train),y]))
  # loss = loss/num_train + 0.5*reg*np.sum(W*W)
  # to solve the above problem try use np.mean instead of loss/train
  loss = np.mean(-np.log(Probs[np.arange(num_train),y]))  
  loss += 0.5*reg*np.sum(W*W)
  
  mat_Yi = np.zeros_like(Probs)
  mat_Yi[np.arange(num_train),y] = 1  # where Yi = 1
  
  dW = X.T.dot(Probs-mat_Yi)
  dW = dW/num_train + reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

