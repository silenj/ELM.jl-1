using ELM
using Random
using DataFrames

Random.seed!(1234)

elm = ExtremeLearningMachine(100)

# Testing XOR
x = [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0]
y = [0.0, 1.0, 0.0, 1.0]
 
fit!(elm, x, y)

td=[1. 1.; 0. 1.; 1. 1.; 1. 0.]

y_pred = predict(elm, td)

@assert y_pred[1] < 0.2
@assert y_pred[2] > 0.8
@assert y_pred[3] < 0.2

# Testing as data frame:

xf=DataFrame(x)
 
fit!(elm, xf, y)

y_pred = predict(elm, td)

@assert y_pred[1] < 0.2
@assert y_pred[2] > 0.8
@assert y_pred[3] < 0.2
