https://yao-lab.github.io/course/msbd5013/2022/slides/Lecture01_introduction.pdf

## Course overview
![](files/lec01-00.png)

## History of AI
![](files/lec01-01.png)

![](files/lec01-02.png)

- 1795: First machine learning method: Least Squares
- 1912: Fisher's Maximum Likelihood Principle. The least squares method is maximum likelihood estimate when the noise is Gaussian.
- 1957: First Neural Network: Perceptron. `l(w) = -sum_{i in M_w}y_i<w, x_i>, M_w = {i: y_i<x_i,w> < 0, y_i in {-1,1}` The Perceptron algorithm is a Stochastic Gradient Descent method.
- 1986: MLP and Back-Propagration Algorithms. BP as stochastic gradient descent algorithms.
- 1989: Convolutional Neural Network: shift variances and locality
![](files/lec01-03.png)

## Time series
- Linear Dynamical Systems (1940s-): Kalman Filter - A linearly transformed Gaussian is a Gaussian. So the distribution over the hidden state given the data so far is Gaussian.
- Hidden Markov Models (1970s-): HMMs have efficient algorithms (Baum-Welch or EM Algorithm) for inference and learning
- RNN (1986-): RNNs are very powerful, because they combine two properties: Distributed hidden state that allows them to store a lot of information about the past efficiently + Non-linear dynamics that allows them to update their hidden state in complicated ways.
- Long-Short-Term-Memory (1997-):

- - 

- 2000-2010: The Era of SVM, Boosting, … as nights of Neural Networks
- Decision Trees and Boosting
- 2012: Return of NN as Deep Learning
![](files/lec01-04.png)
![](files/lec01-05.png)

## DL is not perfect
![](files/lec01-06.png)
![](files/lec01-07.png)
![](files/lec01-08.png)
![](files/lec01-09.png)

- - -

![](files/lec01-10.png)
![](files/lec01-11.png)
![](files/lec01-12.png)
![](files/lec01-13.png)


# Overview of Supervised Learning
https://yao-lab.github.io/course/msbd5013/2022/slides/Lecture01_supervised.pdf

## Probability vs. Statistical Machine Learning
![](files/lec01-14.png)

![](files/lec01-15.png)
_Figure_: Larry Wasserman’s classification of statistical learning vs. machine learning in Computer Science

## No free lunch in statistics
![](files/lec01-16.png)

## The Bias-Variance Trade-Off of Prediction Error
![](files/lec01-17.png)

![](files/lec01-18.png)

![](files/lec01-19.png)

## Bias-variance tradeoff
![](files/lec01-20.png)
![](files/lec01-20.png)
![](files/lec01-21.png)
![](files/lec01-22.png)
![](files/lec01-12.png)
