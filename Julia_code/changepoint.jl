#using Pkg
#Pkg.add("Plots")

using QuadGK
using Random, Distributions, DataFrames, SpecialFunctions, Revise, StatsPlots, StatsBase, Plots
include("all_functions.jl")

@time begin

theta2_true = 0.5
#–– MCMC settings and data prep ––
Random.seed!(1234)                # for reproducibility
N0      = 1000                    # number of individuals
Tmax    = 30                      # days
theta   = 0.25                     # initial θ for dust sim
xi      = log(2)/7 
breaks  = 2
t_star  = 15                   

# run the dust‐obs sim and pull out observed counts
result0 = simulate_dust_observation(
    N      = N0,
    Tmax   = 30,
    theta1 = theta,
    xi     = xi,
    breaks = breaks, 
    theta2 = theta2_true,
    t_star = t_star
)

X_obs = result0.observations.Dust

infected_counts = [count(==(1), result0.states[:, t]) for t in 1:Tmax]
flag = any(infected_counts[1:end-1] .== 0) ? 1 : 0

posterior = changepoint_mcmc(
    X_obs;
    N = N0,
    Tmax = Tmax,
    xi = xi,
    breaks = breaks,
    n_iter = 10000,
    burn_in = 1000,
    thin = 1,
    prop_sd = (theta1=0.05, theta2=0.05, tau=3)
)

end

println("theta2 true = ", theta2_true)
posterior_df = DataFrame(posterior.samples, [:theta1, :theta2, :tau])


#freq_df = combine(groupby(posterior_df, :tau), nrow => :count)
#println(freq_df)

# 1) Extract the tau‐column
taus = posterior_df[:, 3]  

# 2) Find the mode 
println("The median θ1_cur = ", median(posterior_df[:, 1]))
println("The median θ2_cur = ", median(posterior_df[:, 2]))

median_tau = Int(median(taus))
println("The median τ = $median_tau")
println("Accepted Rate = ", posterior.accepted_rate)

change = Int(abs(median_tau - t_star) ≤ 2)

@show minimum(taus), maximum(taus)
@show length(taus)
τ_map = mode(taus)
println("MAP (posterior mode) τ = ", τ_map)

Tmax = 60 
counts = countmap(taus)  # Dict: τ => count

p_tau = [get(counts, t, 0) / length(taus) for t in 1:(Tmax-1)]

# show the top 10 most likely change points
df_tau = DataFrame(tau = 1:(Tmax-1), prob = p_tau)
first(sort(df_tau, :prob, rev=true), 10)

ci = quantile(taus, [0.025, 0.5, 0.975])
println("τ quantiles (2.5%, 50%, 97.5%) = ", ci)


p11 = plot(posterior_df.theta1, title="Traceplot for θ₁", xlabel="Iteration", ylabel="Value", legend=false)
p12 = plot(posterior_df.theta2, title="Traceplot for θ₂", xlabel="Iteration", ylabel="Value", legend=false)
p13 = plot(posterior_df.tau, title="Traceplot for τ", xlabel="Iteration", ylabel="Value", legend=false)

display(p11)

#l = @layout [a b c]

# Display the plots in a single layout
#plot(p1, p2, p3, layout=l)


p1 = @df posterior_df histogram(
    :theta1, bins=30, alpha=0.5, color=:blue,
    xlabel="θ₁", ylabel="Density",
    title="θ₁",
    legend=false
)

p2 = @df posterior_df histogram(
    :theta2, bins=30, alpha=0.5, color=:red,
    xlabel="θ₂", ylabel="Density",
    title="θ₂",
    legend=false
)

p3 = @df posterior_df histogram(
    :tau, bins=30, alpha=0.5, color=:green,
    xlabel="τ", ylabel="Density",
    title="τ",
    legend=false
)




