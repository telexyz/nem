#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;
using namespace std;

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code. This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (foat *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    const size_t iters = m / batch; // xác định số lượng mẻ
    const float delta = lr / batch;
    float* z = new float[batch * k];

    for (size_t i = 0; i < iters; i++) {
        // Xác định begin và end cho mẻ hiện tại
        size_t batch_begin_row = i * batch;
        size_t batch_end_row = batch_begin_row + batch;
        if (batch_end_row > m) { batch_end_row = m; }
        
        // Với từng vector dữ liệu trong mẻ
        size_t offset = 0;
        for (size_t b = batch_begin_row; b < batch_end_row; b++) {
            float total_exp = 0;
            for (size_t l = 0; l < k; l++) { // với mỗi nhãn l trong k nhãn
                z[offset + l] = 0;
                for (size_t j = 0; j < n; j++) { 
                // với mỗi phần tử của vector dữ liệu thứ b được nhân
                // với phần tử hàng tương ứng của cột theta thứ l
                    z[offset + l] += X[b * n + j] * theta[j * k + l];
                }
                z[offset + l] = exp(z[offset + l]);
                total_exp += z[offset + l];
            }
            for (size_t l = 0; l < k; l++) {
                z[offset + l] /= total_exp;
            }
            z[offset + y[b]] -= 1; // tương đưong với z_ey của vector dữ liệu thứ b
            offset += k;
        } // vectors of curr batch iter

        // XT = np.transpose(X[batch_range])
        // mm = np.matmul(XT, batch_z_ey) # (n,m),(m,k)->(n,k):
        // theta -=  delta * mm
        for (size_t row = 0; row < n; row++) {
            for (size_t col = 0; col < k; col++) {
                float mm = 0;
                for (size_t iter = 0; iter < batch; iter++) {
                    const size_t X_idx = (batch_begin_row + iter) * n + row; // batch * n
                    const size_t z_idx = iter * k + col; // batch * k
                    mm += X[X_idx] * z[z_idx];
                }
                theta[row * k + col] -= delta * mm; // m hàng, n cột
            }
        }

    } // batch iter

    delete[] z;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
