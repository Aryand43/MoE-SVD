# compression_runner.jl

include("gating.jl")
include("activation_stats.jl")
include("svd_decompose.jl")
include("sensitivity_metrics.jl")
include("u_matrix_trimming.jl")
include("v_matrix_sharing.jl")

using LinearAlgebra

"""
    run_compression_pipeline(weights, activations, frequencies;
                             k_top=2, rank_threshold=0.01, tau=2.0, sensitivity_threshold=50)

Runs the full MoE-SVD pipeline. If the layer is deemed important (S_L ≥ threshold), it's left untouched.
Otherwise, U-matrix trimming and V-matrix sharing are applied for compression.

Returns a named tuple:
    (compressed = Bool, experts = final_expert_mats, V_shared = maybe_Vs, S_L = score)
"""
function run_compression_pipeline(weights::Vector{Matrix{Float64}},
                                  activations::Vector{Matrix{Float64}},
                                  frequencies::Vector{Float64};
                                  k_top::Int = 2,
                                  rank_threshold::Float64 = 0.01,
                                  tau::Float64 = 2.0,
                                  sensitivity_threshold::Float64 = 50)

    # Step 1: Compute activation outlier ratios aᵢ
    outlier_ratios = compute_outlier_ratios(activations, tau)

    # Step 2: Compute principal ranks rᵢ using SVD
    ranks = compute_principal_ranks(weights, rank_threshold)

    # Step 3: Compute sensitivity score S_L
    S_L = compute_layer_sensitivity(frequencies, ranks, outlier_ratios)

    println("Layer Sensitivity Score (S_L): ", round(S_L, digits=2))

    if S_L ≥ sensitivity_threshold
        println("Skipping compression — layer deemed important.")
        return (compressed = false, experts = weights, V_shared = nothing, S_L = S_L)
    end

    println("Applying compression — sensitivity below threshold.")

    # Step 4: Run U-matrix trimming (preserves top-k diversity)
    trimmed = u_matrix_trim(weights, frequencies, k_top)

    # Step 5: Apply V-matrix sharing
    svd_data = [svd(W) for W in weights]
    v_share_result = v_matrix_sharing(svd_data, frequencies)
    shared_Vᵀ = v_share_result.V_shared

    # Step 6: Form final compressed expert matrices: Eᵢ = UΣ_combined * shared_Vᵀ
    compressed_experts = [UΣ * shared_Vᵀ for (UΣ, _, _) in trimmed]

    println("Compression complete. Experts compressed: ", length(compressed_experts))

    orig_params = sum(length.(weights))
    new_params = sum(length.(compressed_experts))
    println("Parameter count reduced from $orig_params to $new_params.")

    return (compressed = true, experts = compressed_experts, V_shared = shared_Vᵀ, S_L = S_L)
end

# Test
function run_dummy_compression_test()
    num_experts = 4
    weights = [randn(128, 64) for _ in 1:num_experts]
    activations = [randn(128, 5) for _ in 1:num_experts]
    frequencies = rand(4)

    result = run_compression_pipeline(weights, activations, frequencies;
                                      k_top=2, rank_threshold=0.01, tau=2.0,
                                      sensitivity_threshold=100.0)

    println("Was layer compressed? ", result.compressed)
    println("Shape of expert 1 matrix: ", size(result.experts[1]))
end

# Uncomment to run test
run_dummy_compression_test()
