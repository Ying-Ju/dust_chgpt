using Distributed

const N_PARALLEL_WORKERS = max(Sys.CPU_THREADS - 1, 1)

if nprocs() == 1
    addprocs(N_PARALLEL_WORKERS)
end

@everywhere begin
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
end

function format_number_for_prefix(x::Real)
    return lpad(string(round(Int, 100 * x)), 3, "0")
end

function scenario_prefix(;
    theta1::Float64,
    theta2::Float64,
    true_t_star::Int,
    n_reps::Int,
    n_iter::Int,
    burn_in::Int,
    p_effect_threshold::Float64,
    guard_days::Int,
    tolerance::Int,
)
    return join([
        "theta1_$(format_number_for_prefix(theta1))",
        "theta2_$(format_number_for_prefix(theta2))",
        "tau$(true_t_star)",
        "n$(n_reps)",
        "iter$(n_iter)",
        "burn$(burn_in)",
        "p$(format_number_for_prefix(p_effect_threshold))",
        "guard$(guard_days)",
        "tol$(tolerance)",
    ], "_")
end

function parallel_simulation_study(;
    n_reps::Int=200,
    seed::Int=2026,
    N::Int=1000,
    Tmax::Int=60,
    theta1::Float64=0.10,
    theta2::Float64=0.15,
    true_t_star::Int=15,
    xi::Float64=log(2) / 7,
    breaks::Int=2,
    analysis_every::Int=10,
    min_analysis_day::Int=20,
    n_iter::Int=5000,
    burn_in::Int=1000,
    thin::Int=1,
    prop_sd::NamedTuple=(theta1=0.05, theta2=0.05, tau=3),
    p_effect_threshold::Float64=0.70,
    guard_days::Int=6,
    stop_after_detection::Bool=true,
    track_timing::Bool=false,
    tolerance::Int=3,
)
    started_at = now()
    start_seconds = time()

    replicate_results = pmap(1:n_reps) do rep
        run_one_replicate(
            rep;
            seed = seed,
            N = N,
            Tmax = Tmax,
            theta1 = theta1,
            theta2 = theta2,
            true_t_star = true_t_star,
            xi = xi,
            breaks = breaks,
            analysis_every = analysis_every,
            min_analysis_day = min_analysis_day,
            n_iter = n_iter,
            burn_in = burn_in,
            thin = thin,
            prop_sd = prop_sd,
            p_effect_threshold = p_effect_threshold,
            guard_days = guard_days,
            stop_after_detection = stop_after_detection,
            track_timing = track_timing,
            tolerance = tolerance,
            verbose = false,
        )
    end

    summaries = DataFrame()
    interims = DataFrame()

    for result in replicate_results
        append!(summaries, result.summary; cols=:union)
        append!(interims, result.interim; cols=:union)
    end

    sort!(summaries, :rep)
    sort!(interims, [:rep, :analysis_day])

    total_elapsed = time() - start_seconds
    performance = performance_summary(
        summaries;
        n_reps = n_reps,
        theta1 = theta1,
        theta2 = theta2,
        true_t_star = true_t_star,
        p_effect_threshold = p_effect_threshold,
        guard_days = guard_days,
        analysis_every = analysis_every,
        min_analysis_day = min_analysis_day,
        stop_after_detection = stop_after_detection,
        track_timing = track_timing,
        tolerance = tolerance,
        started_at = started_at,
        total_elapsed = total_elapsed,
        completed = true,
    )

    return (
        performance = performance,
        summaries = summaries,
        interims = interims,
    )
end

theta1 = 0.10
theta2 = 0.15
true_t_star = 15
n_reps = 1000
n_iter = 5000
burn_in = 1000
p_effect_threshold = 0.70
guard_days = 6
tolerance = 3

prefix = scenario_prefix(
    theta1 = theta1,
    theta2 = theta2,
    true_t_star = true_t_star,
    n_reps = n_reps,
    n_iter = n_iter,
    burn_in = burn_in,
    p_effect_threshold = p_effect_threshold,
    guard_days = guard_days,
    tolerance = tolerance,
)

println("Running scenario: ", prefix)
println("Workers: ", nworkers())

total_elapsed_seconds = @elapsed begin
    results = parallel_simulation_study(
        n_reps = n_reps,
        seed = 2026,
        theta1 = theta1,
        theta2 = theta2,
        true_t_star = true_t_star,
        analysis_every = 10,
        min_analysis_day = 20,
        n_iter = n_iter,
        burn_in = burn_in,
        thin = 1,
        prop_sd = (theta1=0.05, theta2=0.05, tau=3),
        p_effect_threshold = p_effect_threshold,
        guard_days = guard_days,
        stop_after_detection = true,
        track_timing = false,
        tolerance = tolerance,
    )

    save_results_csv(results; prefix = prefix)
end

println("Total simulation_parallel.jl elapsed time: ", round(total_elapsed_seconds; digits=2), " seconds")
println("Total simulation_parallel.jl elapsed time: ", round(total_elapsed_seconds / 60; digits=2), " minutes")

show(results.performance; allcols=true)
println()
