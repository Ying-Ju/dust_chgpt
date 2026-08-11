using CSV
using DataFrames
using Statistics
using StatsPlots
using Printf
using Dates


include("all_functions.jl")

const ANALYSIS_OUTPUT_PREFIX = "final_n1000_p070_guard6_tol5"
const ANALYSIS_FILE_PATTERN = r"theta1_010_theta2_(\d+)_tau(\d+)_n1000_iter5000_burn1000_p070_guard6_tol3_(performance|summaries)\.csv$"
const TABLE_TAU_TOLERANCE = 5

# Load once, then rerun only the output you are editing.
data = load_analysis_data(file_pattern = ANALYSIS_FILE_PATTERN)
performance = data.performance
summaries = data.summaries

update_tau_within_tolerance!(
    performance,
    summaries;
    tolerance = TABLE_TAU_TOLERANCE,
)

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
