# see https://github.com/FluxML/model-zoo/blob/master/other/iris/iris.jl

using Flux, Flux.Data.MNIST, Statistics
using Flux: crossentropy, normalise, onecold, onehotbatch
using Statistics: mean

##

imgs = MNIST.images()

X = hcat(float.(reshape.(imgs, :))...)
# atm I have no idea what ... does

labels = MNIST.labels()
Y = onehotbatch(labels, 0:9)

# TODO: refactor m -> m_logres
m_logres = Chain(
    Dense(28^2, 10),
    softmax
)

loss(x, y) = crossentropy(m(x), y)
##
accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))

opt = Descent(0.5)

# 200 epochs (compare work.jl / fully connectd mlp)
data_iterator = Iterators.repeated((X, Y), 200)

evalcb = () -> @show(loss(X, Y))
##

Flux.train!(loss, params(m), dataset, opt, cb = throttle(evalcb, 10))
##

accuracy(X, Y)

##
# Test set accuracy
tX = hcat(float.(reshape.(MNIST.images(:test), :))...)
tY = onehotbatch(MNIST.labels(:test), 0:9)

accuracy(tX, tY)
