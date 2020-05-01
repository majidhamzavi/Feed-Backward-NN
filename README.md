# Feed-Backward-NN
What is feed backward neural network?

In this repository, I will explain what is feed backward neural network (NN) and how it works.


Suppose you don't have any hidden layers, x and Y are given, and you will initialze randome weights (w) and biases (b).
Therefore, in this model, the output becomes P = x . w + b and the loss function would be:
<img src="https://render.githubusercontent.com/render/math?math=\textit{LOSS}= \frac{1}{2} [P - Y]^{2} = \frac{1}{2} [x . w %2B b - Y]^{2}">.

We want to minimize loss of a randomly assigned weights and biases backward, i.e. <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial \textit{LOSS}}{\partial w} = x ^{t} . (P - Y)"> and <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial \textit{LOSS}}{\partial b} = P - Y">.

Then, we can update weights and biases as <img src="https://render.githubusercontent.com/render/math?math=w = w - \lambda w ^{t} . (P - Y)">, <img src="https://render.githubusercontent.com/render/math?math=b = b - \lambda (P-Y)">, where <img src="https://render.githubusercontent.com/render/math?math=\lambda"> is a scaling coefficient to ensure that the value would not dominate the equation.
