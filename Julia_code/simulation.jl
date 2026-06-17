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


total_elapsed_seconds = @elapsed begin
    results = simulation_study(
        n_reps = 5,
        theta1 = 0.15,
        theta2 = 0.25,
        true_t_star = 30,
        analysis_every = 10,
        min_analysis_day = 20,
        n_iter = 10000,
        burn_in = 1000,
        thin = 1,
        prop_sd = (theta1=0.05, theta2=0.05, tau=3),
        p_effect_threshold = 0.70,
        guard_days = 4,
        stop_after_detection = true,
        track_timing = false,
        verbose = false,
    )

    save_results_csv(results)
end

println("Total simulation.jl elapsed time: ", round(total_elapsed_seconds; digits=2), " seconds")
println("Total simulation.jl elapsed time: ", round(total_elapsed_seconds / 60; digits=2), " minutes")
