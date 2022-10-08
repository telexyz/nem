import struct

import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filesname) as f:
        # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
        pixels = np.frombuffer(f.read(), 'B', offset=16) # 1-pixel = 1-byte
        images = pixels.reshape(-1, 28*28).astype('float32') / 255 # normalize to 0-1

    with gzip.open(label_filename) as f:
        # First 8 bytes are magic_number, n_labels
        labels = np.frombuffer(f.read(), 'B', offset=8)
    
    return (images, labels)
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    # print(">>> ", Z, Z.shape) # (60000, 10)
    # print(">>> ", y_one_hot, y_one_hot.shape) # (60000, 10)
    x = ndl.exp(Z)
    x = ndl.summation(x, axes=1)
    x = ndl.log(x)
    # print(">>> ", x, x.shape) # (60000, )
    y = ndl.multiply(Z, y_one_hot)
    # print(">>> ", y, y.shape) # (60000,)
    y = ndl.summation(y, axes=1)

    r = ndl.add(x, ndl.negate(y))
    # print(">>> ", r, r.shape) # (60000,)
    r = ndl.summation(r, axes=0)
    # print(">>> ", r, r.shape) # (60000,)
    return r / Z.shape[0]
    '''
    total = 0
    batch_size = Z.shape[0]
    for i in range(batch_size):
        Zi = Z[i]
        exp_Zi = np.exp(Zi)
        total += math.log(sum(exp_Zi)) - Zi[y[i]]
    return total / batch_size
    '''
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()

    m = X.shape[0]      # số lượng mẫu huấn luyện (60k với toàn bộ ảnh mnist)
    n = X.shape[1]      # số chiều của vector đầu vào (28x28=784 với ảnh mnist)
    k = W2.shape[1]  # số lớp đầu ra (0..9 với bài toán phân loại ảnh mnist)
    assert n == W1.shape[0] # Đảm bảo phép nhân ma trận là hợp lệ
    assert W1.shape[1] == W2.shape[0] # Đảm bảo phép nhân ma trận là hợp lệ

    print("nn_epoch cho đầu vào:", m, n, k, W1.shape[1], batch)
    batch_begin = 0
    count = 0
    
    while (batch_begin < m):
        batch_end = batch_begin + batch
        if batch_end > m:
            batch_end = m # normalize batch_end
            batch = batch_end - batch_begin

        batch_range = range(batch_begin, batch_end)
        batch_begin += batch

        # Init
        Xb = X[batch_range] # print(Xb.shape) => (50, 5)
        yb = np.array(y[batch_range])

        # https://www.delftstack.com/howto/numpy/one-hot-encoding-numpy
        yb_one_hot = np.zeros((batch, k))
        yb_one_hot[np.arange(yb.size), yb] = 1 # print(yb_one_hot)

        """Forward"""
        Xb_W1 = ndl.matmul(ndl.Tensor(Xb), W1)
        Zb = ndl.matmul(ndl.relu( Xb_W1 ), W2)

        """Loss function"""
        loss = softmax_loss(Zb, ndl.Tensor(yb_one_hot)) # chính là out_grad

        """Backward"""
        loss.backward()

        grad_W1 = W1.grad.numpy()
        grad_W2 = W2.grad.numpy()
        # print(grad_W1, "\n\n", grad_W2) # OK!

        W1 = ndl.Tensor(W1.numpy() - lr * grad_W1)
        W2 = ndl.Tensor(W2.numpy() - lr * grad_W2)

    return (W1, W2)
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
