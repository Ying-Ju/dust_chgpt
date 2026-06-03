

using QuadGK
using Random, Distributions, DataFrames, SpecialFunctions, Revise, StatsPlots, StatsBase, Plots
include("all_functions.jl")

theta2_true = 0.25


#–– MCMC settings and data prep ––
Random.seed!(2025)                # for reproducibility
N0      = 1000                    # number of individuals
Tmax    = 60                      # days
theta   = 0.15                     # initial θ for dust sim
xi      = log(2)/7
breaks  = 2
t_star  = 45                      

# run the dust‐obs sim and pull out observed counts
result0 = simulate_dust_observation(
    N      = N0,
    Tmax   = Tmax,
    theta1 = theta,
    xi     = xi,
    breaks = breaks,
    theta2 = theta2_true,
    t_star = t_star
)

X_obs = result0.observations.Dust

Tmax = size(result0.states, 2) - 1

infected_counts = [count(==(1), result0.states[:, t]) for t in 1:(Tmax+1)]

epi = DataFrame(time = 0:Tmax, infected = infected_counts)

p2 = @df epi scatter(
    :time, :infected;
    markercolor = :firebrick,
    xlabel      = "Time (t)",
    ylabel      = "Number Infected",
    title       = "Number of Infected Individuals Over Time",
    legend      = false,
    framestyle  = :box,
    titlefont   = font(12, :bold),
    guidefont   = font(11)
)
display(p2)

