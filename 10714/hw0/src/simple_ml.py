import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


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

            y (numpy.ndarray[dypte=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE

    with gzip.open(image_filesname) as f:
        # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
        pixels = np.frombuffer(f.read(), 'B', offset=16) # 1-pixel = 1-byte
        images = pixels.reshape(-1, 28*28).astype('float32') / 255 # normalize to 0-1

    with gzip.open(label_filename) as f:
        # First 8 bytes are magic_number, n_labels
        labels = np.frombuffer(f.read(), 'B', offset=8)
    
    return (images, labels)
    ### END YOUR CODE

import math
def softmax_loss(Z, y):
    """ Return softmax loss. Bài tập này chưa cần tính `log-sum-max` tối ưu.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape (batch_size, num_classes), 
            containing the logit predictions for each class.

        y (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE

    total = 0
    batch_size = Z.shape[0]
    # print(Z.shape, len(y), batch_size) # (60000, 10) 10 60000
    for i in range(batch_size):
        Zi = Z[i]
        exp_Zi = np.exp(Zi)
        total += math.log(sum(exp_Zi)) - Zi[y[i]]
    return total / batch_size

    ### END YOUR CODE


def softmax(Z):
    ''' https://machinelearningcoban.com/2017/02/17/softmax/#-softmax-function-trong-python
    Đầu vào là một ma trận với mỗi cột là một vector z, đầu ra cũng là một ma trận mà 
    mỗi cột có giá trị là a = softmax(z). Các giá trị của z còn được gọi là scores.
    '''
    e_Z = np.exp(Z)
    return e_Z / e_Z.sum(axis = 0) # np.sum([[0, 1], [0, 5]], axis=0) => array([0, 6])

def softmax_stable(Z):
    """
    Compute softmax values for each sets of scores in Z.
    each column of Z is a set of score.    
    """
    e_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True))
    return e_Z / e_Z.sum(axis = 0)
    # trong đó axis = 0 nghĩa là lấy max theo cột (axis = 1 sẽ lấy max theo hàng), 
    # keepdims = True để đảm bảo phép trừ giữa ma trận Z và vector thực hiện được.


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size. This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameter, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """

    ### BEGIN YOUR CODE
    m = X.shape[0]      # số lượng mẫu huấn luyện (60k với toàn bộ ảnh mnist)
    n = X.shape[1]      # số chiều của vector đầu vào (28x28=784 với ảnh mnist)
    k = theta.shape[1]  # số lớp đầu ra (0..9 với bài toán phân loại ảnh mnist)
    assert n == theta.shape[0] # Đảm bảo phép nhân ma trận là hợp lệ

    print("Tính hồi quy softmax cho đầu vào:", m, n, k, batch)
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
        Xb = X[batch_range]
        yb = np.array(y[batch_range])
        XT = np.transpose(Xb)
        Xtheta = np.matmul(Xb, theta)
        
        # Tính G2
        G2 = np.zeros((batch, k))
        for j in range(0, batch):
            h = Xtheta[j]
            exp_h = np.exp(h)            

            z_ey = exp_h / sum(exp_h)
            z_ey[yb[j]] -= 1 # z + ey với ey = -1 { i = y }
            G2[j] = z_ey

        # Adjust theta
        mm = np.matmul(XT, G2) # (n,k),(k,m)->(n,m):
        theta -= (lr / batch) * mm

    ### END YOUR CODE
    # https://machinelearningcoban.com/2017/02/17/softmax/#44-h%C3%A0m-ch%C3%ADnh-cho-training-softmax-regression


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarrray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarrray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    m = X.shape[0]      # số lượng mẫu huấn luyện (60k với toàn bộ ảnh mnist)
    n = X.shape[1]      # số chiều của vector đầu vào (28x28=784 với ảnh mnist)
    k = W2.shape[1]  # số lớp đầu ra (0..9 với bài toán phân loại ảnh mnist)
    assert n == W1.shape[0] # Đảm bảo phép nhân ma trận là hợp lệ
    assert W1.shape[1] == W2.shape[0] # Đảm bảo phép nhân ma trận là hợp lệ

    print("Tính nn_epoch cho đầu vào:", m, n, k, W1.shape[1], batch)
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
        Xb = X[batch_range]
        yb = np.array(y[batch_range])
        XT = np.transpose(Xb)
        XW1 = np.matmul(Xb, W1)
        relu_XW1 = np.maximum(0, XW1) # relu(x) = if (x >= 0) x else 0
        relu_XW1_mm_W2 = np.matmul(relu_XW1, W2)

        # Tính G2
        G2 = np.zeros((batch, k))
        for j in range(0, batch):
            h = relu_XW1_mm_W2[j]
            exp_h = np.exp(h)
            z_ey = exp_h / sum(exp_h)
            z_ey[yb[j]] -= 1 # z + ey với ey = -1 { i = y }
            G2[j] = z_ey

        W2T = np.transpose(W2)

        # Adjust W1
        G2_mm_W2T = np.matmul(G2, W2T)
        derivate_relu_XW1 = np.where(XW1 <= 0, 0, 1)

        G1 = np.multiply(derivate_relu_XW1, G2_mm_W2T)
        W1 -= (lr / batch) * np.matmul(XT, G1)

        # Adjust W2
        Z1T = np.transpose(relu_XW1)
        W2 -= (lr / batch) * np.matmul(Z1T, G2)
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
