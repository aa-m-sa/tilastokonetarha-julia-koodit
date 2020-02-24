# work vis
using Plots

xpt = 1:5

ypt = hcat(mlp_ts_res,
    logr_ts_res)

pyplot()

plot(xpt,ypt,
    line=(:line), markershape= :circle, markercolor = :match,
    label=["MLP" "log_res"],
    xlabel="opetusjoukon jae",
    ylabel="keskimääräinen virhe testijoukossa")
savefig("~/mlp_logres.png")
