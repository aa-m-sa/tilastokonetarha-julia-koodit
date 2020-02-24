##

# The MNIST MLP example
# from https://github.com/FluxML/model-zoo/blob/master/vision/mnist/mlp.jl
# see https://github.com/FluxML/Flux.jl/blob/master/docs/src/training/training.md


##
using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated
using Base.Iterators: partition
using Random
using MLDataUtils
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

##

## fivefold CV
rperm = randperm(length(imgs))
n_c = trunc(Int,length(imgs)/5)

folds = kfolds(collect(1:length(imgs)), k=5)

mlp_tr_res = []
mlp_ts_res = []
logr_tr_res = []
logr_ts_res = []

##
for i in 1:5
  Xtmp = X[:, getobs(folds,i)[1]]
  Ytmp = Y[:, getobs(folds,i)[1]]
  tX = X[:, getobs(folds,i)[2]]
  tY = Y[:, getobs(folds,i)[2]]
  dataset = repeated((Xtmp, Ytmp), 80)

  # eläin:
  m_mlp = Chain(
    Dense(28^2, 32, relu),
    Dense(32, 10),
    softmax
  )
  # eläin päättyy

  loss_mlp(x, y) = crossentropy(m_mlp(x),y)

  accuracy_mlp(x,y) = mean(onecold(m_mlp(x)) .== onecold(y))

  evalcb_mlp = () -> @show(loss_mlp(Xtmp, Ytmp))
  opt = ADAM()

  Flux.train!(loss_mlp, params(m_mlp), dataset, opt, cb = throttle(evalcb_mlp, 10))
  accuracy_mlp(tX, tY)

  push!(mlp_tr_res, accuracy_mlp(Xtmp, Ytmp))
  push!(mlp_ts_res, accuracy_mlp(tX, tY))

end

mlp_ts_res

for i in 1:5
  Xtmp = X[:, folds[i][1]]
  Ytmp = Y[:, folds[i][1]]
  tX = X[:, folds[i][2]]
  tY = Y[:, folds[i][2]]
  dataset = repeated((Xtmp, Ytmp), 80)

  # logistinen regressio-eläin
  m_logres = Chain(
    Dense(28^2, 10),
    softmax
  )

  loss_logres(x, y) = crossentropy(m_logres(x),y)

  accuracy_logres(x,y) = mean(onecold(m_logres(x)) .== onecold(y))

  evalcb_logres = () -> @show(loss_logres(Xtmp, Ytmp))

  opt_logres = Descent(0.5)

  Flux.train!((x, y) -> loss(x,y,m_logres),
            params(m_logres),
            dataset,
            opt_logres,
            cb = throttle(evalcb_logres, 10)
  )

  accuracy_logres(tX, tY)

  push!(logr_tr_res, accuracy_logres(Xtmp, Ytmp))
  push!(logr_ts_res, accuracy_logres(tX, tY))
end
#
logr_ts_res
#
# TODO: visualisations
# see  https://github.com/JuliaML/MLPlots.jl/blob/master/src/optional/onlineai.jl
# http://docs.juliaplots.org/latest/recipes/#recipes-1
#
