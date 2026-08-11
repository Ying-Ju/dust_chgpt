using CSV
using DataFrames
using Statistics
using StatsPlots
using Printf


include("all_functions.jl")

const ANALYSIS_OUTPUT_PREFIX = "final_n1000_p070_guard6_tol3"
const ANALYSIS_FILE_PATTERN = r"theta1_010_theta2_(\d+)_tau(\d+)_n1000_iter5000_burn1000_p070_guard6_tol3_(performance|summaries)\.csv$"

# Load once, then rerun only the output you are editing.
data = load_analysis_data(file_pattern = ANALYSIS_FILE_PATTERN)
performance = data.performance
summaries = data.summaries

show_key_performance(performance)


save_combined_tables(
     performance,
     summaries;
     output_prefix = ANALYSIS_OUTPUT_PREFIX,
)

save_rate_figures(
     performance;
     output_prefix = ANALYSIS_OUTPUT_PREFIX,
)

save_boxplots(
     summaries;
     output_prefix = ANALYSIS_OUTPUT_PREFIX,
)


within5 = combine(
    groupby(summaries, [:true_theta1, :true_theta2, :true_t_star]),
    nrow => :n_reps,
    :detected => sum => :detected_count,
    :abs_tau_error => (x -> mean(collect(skipmissing(x)) .<= 5)) => :tau_within_5days_rate,
)

sort!(within5, [:true_theta2, :true_t_star])

show(within5; allcols=true, allrows=true)