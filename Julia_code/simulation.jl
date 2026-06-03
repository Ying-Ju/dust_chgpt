using Dates
using Random
using Statistics
using DataFrames
using Distributions
using QuadGK
using SpecialFunctions
using StatsBase
using CSV

include("all_functions.jl")


results = simulation_study(
    n_reps = 20,
    theta1 = 0.15,
    theta2 = 0.25,
    true_t_star = 30,
    analysis_every = 10,
    min_analysis_day = 20,
    n_iter = 10000,
    burn_in = 1000,
    thin = 1,
    prop_sd = (theta1=0.05, theta2=0.05, tau=3),
)

save_results_csv(results)