# activation_stats.jl

using Statistics

# Computes the activation outlier ratio for each expert
# This helps measure how "spiky" or intense expert activations are

function compute_outlier_ratios(activations::Vector{<:AbstractMatrix}, τ::Real)
    outlier_ratios = Float64[]

    for A in activations
        # For each expert activation matrix:
        # Calculates mean absolute activation
        abs_vals = abs.(A)
        mean_abs = mean(abs_vals)

        # Counts how many elements exceed mean × τ
        threshold = τ * mean_abs
        outlier_count = count(x -> x > threshold, abs_vals)
        total_count = length(abs_vals)

        # Outlier ratio = how spiky or activated the expert is
        # Why? aᵢ reflects how intense or extreme expert responses are
        push!(outlier_ratios, outlier_count / total_count)
    end

    return outlier_ratios
end

# S_L = ∑ (fᵢ × rᵢ × aᵢ)
# Where:
# fᵢ = routing frequency (from gating.jl)
# rᵢ = number of significant singular values in expert i's weights (via SVD)
# aᵢ = outlier ratio (computed here)

# rᵢ reflects how information-rich or complex the expert’s weights are
# Calculated by keeping singular values above 0.01 × max(singular_values)
# This whole process helps determine which experts/layers to compress or preserve

# Example test case to visualize outlier ratios
function run_dummy_activation_stats()
    num_experts = 4
    # Create dummy activations: 128 units × 5 tokens per expert
    activations = [randn(128, 5) for _ in 1:num_experts]
    τ = 2.0  # Outlier threshold multiplier

    outlier_ratios = compute_outlier_ratios(activations, τ)

    println("Outlier Ratios per Expert:")
    println(outlier_ratios)
end

# Uncomment to test standalone
run_dummy_activation_stats()
