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
        n_reps = 20,
        theta1 = 0.15,
        theta2 = 0.25,
        true_t_star = 30,
        analysis_every = 10,
        min_analysis_day = 20,
        n_iter = 3000,
        burn_in = 500,
        thin = 1,
        prop_sd = (theta1=0.05, theta2=0.05, tau=3),
        p_effect_threshold = 0.70,
        guard_days = 6,
        stop_after_detection = true,
        track_timing = false,
        tolerance = 3,
        verbose = false,
    )

    save_results_csv(results; prefix = "pilot_p070_guard6")
end

println("Total simulation.jl elapsed time: ", round(total_elapsed_seconds; digits=2), " seconds")
println("Total simulation.jl elapsed time: ", round(total_elapsed_seconds / 60; digits=2), " minutes")


results.performance[:, [
    :n_reps,
    :true_theta1,
    :true_theta2,
    :true_t_star,
    :detection_rate,
    :false_alarm_rate,
    :median_detection_delay,
    :median_abs_tau_error,
    :tau_within_tolerance_rate,
]]