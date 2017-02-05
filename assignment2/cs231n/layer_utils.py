from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


pass


def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db

# define a new layer with affine_relu and BN
def affine_BN_relu_forward(x, W, b, gamma, beta, bn_param):
    '''
    a affine - relu - BN layer combined with fully-connected layer with ReLu and 
    Batch normalization
    
    The architecure is affine - relu - BN.
    
    compute the forward pass for affine - relu - BN layer  
    
    Inputs:
    - x: A numpy array containing input data, of shape 
    (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M), D = d_1*d_2...*d_k
    - b: A numpy array of biases, of shape (M,)
    - gamma: Scale parameter of shape (M,)
    - beta: Shift paremeter of shape (M,)
    - bn_param: Dictionary with the following keys:
        - mode: 'train' or 'test'; required
        - eps: Constant for numeric stability
        - momentum: Constant for running mean / variance.
        - running_mean: Array of shape (M,) giving running mean of features
        - running_var Array of shape (M,) giving running variance of features
        
    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: cache of affine layer,batch normalization layer and relu layer
    '''
    out,cache_affine = affine_forward(x,W,b)
    out,cache_BN = batchnorm_forward(out, gamma, beta, bn_param)
    out,cache_relu = relu_forward(out)
    
    cache = (cache_affine,cache_BN,cache_relu)
    return out,cache
    
def affine_BN_relu_backward(dout,cache):
    '''
    compute the backward pass for affine - relu - BN layer.
    
    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
        - cache_affine: cache for affine layer
        - cache_BN: cache for batch normalization layer
        - cache_relu: cache for relu layer

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    - dgamma: Gradient with respect to gamma, of shape (M,)
    - dbeta: Gradient with respect to beta, of shape (M,)
    '''
    cache_affine,cache_BN,cache_relu = cache    
    dout = relu_backward(dout,cache_relu)
    dout,dgamma,dbeta = batchnorm_backward_alt(dout, cache_BN)
    dx,dw,db = affine_backward(dout,cache_affine)
    
    return dx,dw,db,dgamma,dbeta

def conv_BN_relu_pool_forward(x, W, b, conv_param, gamma, beta, bn_param, pool_param):
    """
    Convenience layer that performs a convolution, batch normalization,a ReLU, 
    and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - gamma, beta, bn_param : Scale parameters for BN layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    scores, conv_cache = conv_forward_fast(x, W, b, conv_param)
    scores,BN_cache = spatial_batchnorm_forward(scores, gamma, beta, bn_param)
    scores, relu_cache = relu_forward(scores)
    out, pool_cache = max_pool_forward_fast(scores, pool_param)
    cache = (conv_cache, BN_cache, relu_cache, pool_cache)
    
    return out,cache

def conv_BN_relu_pool_backward(dout,cache):
    """
    Backward pass for the conv-BN-relu-pool convenience layer
    """
    conv_cache, BN_cache, relu_cache, pool_cache = cache
    dscores = max_pool_backward_fast(dout, pool_cache)
    dscores = relu_backward(dscores, relu_cache)
    dscores,dgamma,dbeta = spatial_batchnorm_backward(dscores,BN_cache)
    dx, dw, db = conv_backward_fast(dscores, conv_cache)
    
    return dx, dw, db, dgamma, dbeta

def conv_BN_relu_forward(x, W, b, conv_param, gamma, beta, bn_param):
    """
    Convenience layer that performs a convolution, batch normalization and a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - gamma, beta, bn_param : Scale parameters for BN layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    scores, conv_cache = conv_forward_fast(x, W, b, conv_param)
    scores,BN_cache = spatial_batchnorm_forward(scores, gamma, beta, bn_param)
    out, relu_cache = relu_forward(scores)
    cache = (conv_cache, BN_cache, relu_cache)
    
    return out,cache

def conv_BN_relu_backward(dout,cache):
    """
    Backward pass for the conv-BN-relu convenience layer
    """
    conv_cache, BN_cache, relu_cache = cache
    dscores = relu_backward(dout, relu_cache)
    dscores,dgamma,dbeta = spatial_batchnorm_backward(dscores,BN_cache)
    dx, dw, db = conv_backward_fast(dscores, conv_cache)
    
    return dx, dw, db, dgamma, dbeta