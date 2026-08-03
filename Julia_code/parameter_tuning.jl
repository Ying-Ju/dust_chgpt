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
        n_reps = 50,
        theta1 = 0.15,
        theta2 = 0.25,
        true_t_star = 30,
        analysis_every = 10,
        min_analysis_day = 20,
        n_iter = 5000,
        burn_in = 1000,
        thin = 1,
        prop_sd = (theta1=0.05, theta2=0.05, tau=3),
        p_effect_threshold = 0.70,
        guard_days = 5,
        stop_after_detection = true,
        track_timing = false,
        tolerance = 3,
        verbose = false,
    )

    save_results_csv(results; prefix = "full_p070_guard5")
end


grid = tuning_grid_from_interims(
    results.interims;
    p_effect_thresholds = collect(0.60:0.05:0.90),
    guard_days_values = [4, 5, 6],
    tolerance = 3,
)

grid.performance

using StatsPlots

@df grid.performance plot(
    :p_effect_threshold,
    :detection_rate,
    group = :guard_days,
    marker = :circle,
    xlabel = "p_effect_threshold",
    ylabel = "Detection rate",
)

@df grid.performance plot(
    :p_effect_threshold,
    :false_alarm_rate,
    group = :guard_days,
    marker = :circle,
    xlabel = "p_effect_threshold",
    ylabel = "False alarm rate",
)

@df grid.performance plot(
    :p_effect_threshold,
    :tau_within_tolerance_rate,
    group = :guard_days,
    marker = :circle,
    xlabel = "p_effect_threshold",
    ylabel = "Tau within tolerance rate",
)


unstack(
    select(grid.performance, :p_effect_threshold, :guard_days, :tau_within_tolerance_rate),
    :guard_days,
    :tau_within_tolerance_rate
)

CSV.write("../results/tuning_grid_performance.csv", grid.performance)