##

# The MNIST MLP example
# from https://github.com/FluxML/model-zoo/blob/master/vision/mnist/mlp.jl
# see https://github.com/FluxML/Flux.jl/blob/master/docs/src/training/training.md


##
using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated
# using CuArrays

# Classify MNIST digits with a simple multi-layer-perceptron
##
imgs = MNIST.images() # training images given by default
# Stack images into one large batch
X = hcat(float.(reshape.(imgs, :))...)


labels = MNIST.labels()
# One-hot-encode the labels
Y = onehotbatch(labels, 0:9)
##
loss(x, y, m) = crossentropy(m(x), y)
accuracy(x, y, m) = mean(onecold(m(x)) .== onecold(y))
dataset = repeated((X, Y), 200)
##

m_mlp = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 10),
  softmax)


loss_mlp(x, y) = loss(x,y, m_mlp)

accuracy_mlp(x,y) = accuracy(x,y,m_mlp)

evalcb_mlp = () -> @show(loss_mlp(X, Y))
opt = ADAM()


##

Flux.train!(loss_mlp, params(m_mlp), dataset, opt, cb = throttle(evalcb_mlp, 10))

##

accuracy_mlp(X, Y)

##
# Test set accuracy
tX = hcat(float.(reshape.(MNIST.images(:test), :))...)
tY = onehotbatch(MNIST.labels(:test), 0:9)

accuracy_mlp(tX, tY)

##

m_logres = Chain(
    Dense(28^2, 10),
    softmax
)

loss_logres(x, y) = loss(x, y, m_logres)
accuracy_logres(x, y) = accuracy(x, y, m_logres)
evalcb_logres = () -> @show(loss_logres(X, Y))

opt_logres = Descent(0.5)

Flux.train!((x, y) -> loss(x,y,m_logres),
            params(m_logres),
            dataset,
            opt_logres,
            cb = throttle(evalcb_logres, 10))

##
# Test set accuracy
tX_true = hcat(float.(reshape.(MNIST.images(:test), :))...)
tY_true = onehotbatch(MNIST.labels(:test), 0:9)
accuracy_logres(tX, tY)

##
# TODO: convolutional? jos ehtii

##
# TODO: crossvalidation

##
# TODO: visualisations
# see  https://github.com/JuliaML/MLPlots.jl/blob/master/src/optional/onlineai.jl
# http://docs.juliaplots.org/latest/recipes/#recipes-1
#
