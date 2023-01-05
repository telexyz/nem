"""Operator implementations."""

from numbers import Number
from typing import Optional, List
from .autograd import NDArray # NDArray = numpy.ndarray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy as array_api

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        # đạo_hàm(x^n) = n*x^(n-1)
        # => x^(n-1) = out_grad / n
        # => out_grad * n * pow(x, n-1)
        return out_grad * self.scalar * power_scalar(node.inputs[0], self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        ''' Chain rule: gradient(a) = out_grad * gradient_a(f(a,b))
        b cố định đạo_hàm_a(a / b) = 1 / b
        https://www.cuemath.com/questions/find-the-derivative-of-1-x
        a cố định đạo_hàm_b(a / b):
        vì đạo_hàm(1/x) = đạo_hàm(x^-1)
        mà đạo_hàm(x^n) =  n * x^(n - 1)
        => đạo_hàm(1/x) = -1 * b^(-2) = -array_api.power(rhs, -2)
        '''
        a, b = node.inputs
        return out_grad / b, out_grad * (-power_scalar(b, -2) * a)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad / self.scalar,)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # Transpose: reverses the order of two axes (axis1, axis2), defaults to the last two axes (1 input, axes - tuple)
        # => !! Thao tác cần làm là hoán đổi thứ tự 2 trục axis1, axis2 !!
        # https://forum.dlsyscourse.org/t/q1-transpose-forward-got-error/1997/2
        '''I think you’re misunderstanding the self.axes parameter. It contains two axes which are to be swapped. While in the array_api.transpose function, axes constains a permutation of [0,1,…,N-1] where N is the number of axes of a. So consider using array_api.swapaxis or create a permutation to use array_api.transpose.'''

        if self.axes:
            return array_api.swapaxes(a, self.axes[0], self.axes[1])
        else:
            return array_api.swapaxes(a, len(a.shape)-1, len(a.shape)-2)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        result = transpose(out_grad, self.axes)
        return (result, )
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        reversed_shape = node.inputs[0].shape
        result = reshape(out_grad, reversed_shape)
        return (result, )
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    '''
    https://forum.dlsyscourse.org/t/intuiton-and-math-behind-the-backward-summation-and-broadcast-operations/2068/4

    broadcasting operation matches the dimensions of 2 tensors, a and b, if the dimension_i for both match or one of them is one i.e. say, A has a shape (1, 2) and B has a shape (4, 2) → A + B is possible since A can be broadcasted to (4, 2). However, say A has a shape (1, 2) and B has a shape (3, 3) then A + B is not possible since A can’t be broadcasted.

    Now, coming to the homework task, consider the following broadcasting operation
    Input I with shape (x, 1, z) --broadcast_to--> output O with shape (x, y, z)

    Thus, the output gradient too will be of the shape (x, y, z). Since there is no explicit operation happening that’s affecting the elements, broadcast_to won’t have an additional gradient component but rather take the out_grad and get it back to the input shape while aggregating the gradients wherever necessary. (this is something similar to what you might have done or thought of for reshape op)

    So, your task then is to identify the correct input shape and get the out_grad back to that.
    '''
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        # print(">>> broadcast_to", a.shape, self.shape, "=>", out_grad.shape)
        # >>> broadcast_to (3, 1) (3, 3) => (3, 3)
        # >>> broadcast_to (1, 3) (3, 3) => (3, 3)
        # >>> broadcast_to (1,) (3, 3, 3) => (3, 3, 3)
        # >>> broadcast_to () (3, 3, 3) => (3, 3, 3)
        # >>> broadcast_to (5, 4, 1) (5, 4, 3) => (5, 4, 3)
        
        axes = ()
        n = len(a.shape)
        for i in range(len(self.shape)):
            if i >= n or self.shape[i] != a.shape[i]:
                axes += (i,)
        r = summation(out_grad, axes=axes)
        # print(axes, r.shape)
        return (reshape(r, a.shape), )
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    # https://numpy.org/doc/stable/reference/generated/numpy.sum.html#numpy.sum
    # np.sum([[0, 1], [0, 5]], axis=0) => array([0, 6])
    # np.sum([[0, 1], [0, 5]], axis=1) => array([1, 5])
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION
    
    '''
    axis: None or int or tuple of ints, optional
    Axis or axes along which a sum is performed. The default, axis=None, will sum all of the elements of the input array. If axis is negative it counts from the last to the first axis.
    
    If axis is a tuple of ints, a sum is performed on all of the axes specified in the tuple instead of a single axis or all the axes as before.
    '''
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # (1,) => tupple một phần tử
        a = node.inputs[0]
        # print(">>> summation", a.shape, self.axes, "=>", out_grad.shape)
        # >>> summation (5, 4) (1,) => (5,)
        # >>> summation (5, 4) (0,) => (4,)
        # >>> summation (5, 4) (0, 1) => ()
        # >>> summation (5, 4, 1) (0, 1) => (1,)
        # >>> summation (5, 4, 2, 1, 3) (1, 3) => (5, 2, 3)
        s = out_grad.shape
        axes = self.axes
        if axes is None: axes = ()
        # Trường hợp axes không phải là tuple thì convert về tuple
        if not isinstance(axes, tuple): axes = (axes,)
        for i in range(len(axes)):
            idx = axes[i]
            s = s[0:idx] + (1,) + s[idx:]
        # print(s)
        r = reshape(out_grad, s)
        return (broadcast_to(r, a.shape), )
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ''' gradient of matrix product 𝐴𝐵 w.r.t. 𝐴 equal 𝐵^𝑇
        https://math.stackexchange.com/questions/1846339/why-does-the-gradient-of-matrix-product-ab-w-r-t-a-equal-bt

        https://forum.dlsyscourse.org/t/q2-backward-batch-matrix-mul/2013/10

        Actually, (6, 6, 5, 4) @ (4, 3) has a implicit broadcast for (4, 3) matrix, that is what implies in backward of BroadcastTo operator.

        If you read the docs for numpy.matmul, you can find this line
        > If either argument is N-D, N > 2, it is treated as a stack of
        > matrices residing in the last two indexes and broadcast accordingly.

        That implies that if A is (6, 6, 5, 4), B is (4, 3), A is treated as
        a stack of matrix and for the multiplication to work, B is broadcast
        to (6, 6, 4, 3). This means B’s contributions to A @ B span over the
        two extra axes, and you need to sum over these axes to get the final
        derivative. Just like in the chain rule, if y = x(s,t) + z(s,t) then
        dy/dt = dy/dx * dx/dt + dy / dz * dz / dt. In doing so, you are summing
        over all the derivatives of where t appears, in that case that is x and z
        '''

        '''https://forum.dlsyscourse.org/t/q2-backward-batch-matrix-mul/2013/19

        Sorry for asking this again, but I am not able to understand the difference. Lets say that A is (5,4) shaped tensor and B is a (5,1) shaped tensor. In this case, when I perform A/B, B is broadcasted implicitly to (5,4) and the resulting output is of shape (5,4). But the adjoint w.r.t B should have a shape of (5,1) and not (5,4). This means we should also take into account this implicit broadcast to (5,4) ( like in the batch matrix mul case where one tensor had shape (6,6,5,4) and other (4,3) ) and calculate the gradient of B accordingly. Is the reasoning correct?


        Consider 2 matrices, say A with dims (m, n) and B with dims (n, p). Then, the matrix multiplication (A @ B) operation produces a resulting matrix C of shape (m, p). This part does not involve any broadcasting or any other thing. Again, this is not an element wise multiplication but rather a matrix level operation that works as long as the #columns in the left matrix is the same as the #rows in the right matrix.

        Consider a sub case of the test case in question - A has a shape (5, 4) and B has a shape (4, 3). Then the matrix multiplication, A @ B = C with a shape (5, 3). Again, no broadcasting involved at all.
        
        Consider another possible sub case of the test case - A is a tensor with shape (6, 6, 5, 4) and B is a tensor with shape (6, 6, 4, 3). Look at the notes section of the numpy matmul doc. So, the operation now is essentially the same as the previous one. However, it’s happening for each of the 36 sub matrices of A and B i.e. A[0, 0] @ B[0, 0] = C[0, 0], A[0, 1] @ B[0, 1] = C[0, 1] and so on. This results in a tensor C with shape (6, 6, 5, 3).

        Considering the test case tensor A has a shape (6, 6, 5, 4) and matrix B has a shape (4, 3), now, there is a broadcasting happening here but it’s not from (4, 3) to (5, 4) which is a) not possible to broadcast as per the broadcasting rules and, b) not necessary. B is broadcasted to (6, 6, 4, 3). The original (4, 3) matrix is duplicated to get a (6, 6, 4, 3) tensor. Now this would be similar to the previously mentioned batch matmul case. Also, note that this broadcasting occurs during the compute call and not grad call. So, it’s not a needle broadcast op but rather a numpy/array_api one.

        As you correctly mentioned, the gradient for B would need to be of the shape (4, 3) and the out_grad that you have would have a shape of (6, 6, 5, 3). So, once you get the derivative w.r.t B (not sure how to explain this part without giving the answer away), it would be of the shape (6, 6, 4, 3) which needs to be aggregated to get the required (4, 3) grad.

        Hopefully, that clarifies matrix multiplication.
        '''
        a, b = node.inputs 

        bt = Transpose()(b)
        l = MatMul()(out_grad, bt)

        n = len(l.shape)-len(a.shape)
        if n > 0:
            axes = ()
            for i in range(n): axes += (i,)
            l = summation(l, axes=axes)


        at = Transpose()(a)
        r = MatMul()(at, out_grad)

        n = len(r.shape)-len(b.shape)
        if n > 0:
            axes = ()
            for i in range(n): axes += (i,)
            r = summation(r, axes=axes)

        return (l, r)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (-out_grad, )
        ### END YOUR SOLUTION

def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # đạo hàm ln(x) = 1 / x
        return (out_grad / node.inputs[0], )
        ### END YOUR SOLUTION

def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # đạo hàm e^x là e^x
        return (out_grad * exp(node.inputs[0]), )
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        assert len(node.inputs) == 1 # chỉ có 1 đầu vào
        a = node.inputs[0]
        assert out_grad.shape == a.shape
        relu_derivate = array_api.where(a.numpy() <= 0, 0, 1)
        return (out_grad * relu_derivate, )
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

