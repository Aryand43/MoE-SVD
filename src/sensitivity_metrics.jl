# sensitivity_metrics.jl

# This script doesn't compute the inputs (fᵢ, rᵢ, aᵢ) — it assumes they're passed in

# fᵢ comes from gating.jl — it’s how often expert i was selected across calibration data
# aᵢ comes from activation_stats.jl — it's how spiky or extreme an expert's activations are
# rᵢ will come from svd_decompose.jl — number of significant singular values, i.e., model complexity

# Computes layer sensitivity using S_L = sum(fᵢ × rᵢ × aᵢ)
# This gives a scalar signal for how important a layer is to preserve

function compute_layer_sensitivity(frequencies::Vector{Float64},
                                   ranks::Vector{Int},
                                   outlier_ratios::Vector{Float64})::Float64
    # Sanity check: ensure all input vectors have same length
    N = length(frequencies)
    @assert N == length(ranks) == length(outlier_ratios) "All input vectors must be same length"

    # Each term fᵢ × rᵢ × aᵢ reflects one expert’s contribution to total layer importance
    contributions = [frequencies[i] * ranks[i] * outlier_ratios[i] for i in 1:N]

    # Final sensitivity score: higher means the layer is more important to preserve
    return sum(contributions)
end

# This script is part of the broader MoE-SVD pipeline but doesn’t integrate other modules directly —
# that comes later in compression_runner.jl

# Final S_L score helps decide whether to compress a layer or leave it intact

# Test function with dummy data
function run_dummy_sensitivity_test()
    # 4 dummy experts
    f = rand(4)                          # Routing frequencies (fᵢ)
    r = rand(10:100, 4)                 # SVD ranks (rᵢ)
    a = rand(4)                          # Outlier ratios (aᵢ)

    S_L = compute_layer_sensitivity(f, r, a)
    println("Dummy Routing Frequencies (fᵢ): ", f)
    println("Dummy SVD Ranks (rᵢ): ", r)
    println("Dummy Outlier Ratios (aᵢ): ", a)
    println("Layer Sensitivity Score (S_L): ", round(S_L, digits=4))
end

# Uncomment to run standalone
run_dummy_sensitivity_test()
