# layer_compress.jl

include("activation_stats.jl")
include("svd_decompose.jl")
include("sensitivity_metrics.jl")
include("u_matrix_trimming.jl")
include("v_matrix_sharing.jl")
include("aw_svd.jl")

using LinearAlgebra

"""
    compress_layer(weights, activs, route_freqs; ...)

Compresses a single MoE FFN layer using activation-weighted SVD, U-matrix trimming,
and V-matrix sharing. Skips compression if the layer's sensitivity is above threshold.

Returns a named tuple:
    (compressed, experts, V_shared, S_L)
"""
function compress_layer(
        weights::Vector{Matrix{Float64}},
        activs::Vector{Matrix{Float64}},
        route_freqs::Vector{Float64};
        k_top::Int             = 2,
        tau::Float64           = 2.0,
        rank_thresh::Float64   = 0.01,
        target_rank::Union{Int,Nothing} = 16,
        sens_thresh::Float64   = 50.0
    )

    # Step 1: Compute activation outlier ratios
    a = compute_outlier_ratios(activs, tau)

    # Step 2: Compute principal ranks
    r = compute_principal_ranks(weights, rank_thresh)

    # Step 3: Compute sensitivity score
    S_L = compute_layer_sensitivity(route_freqs, r, a)

    println("Sensitivity Score S_L = ", round(S_L, digits=2))

    if S_L ≥ sens_thresh
        println("Layer skipped (high sensitivity).")
        return (
            compressed = false,
            experts = weights,
            V_shared = nothing,
            S_L = S_L
        )
    end

    println("Compressing layer (low sensitivity).")

    # Step 4: Run activation-weighted SVD on each expert
    aw_results = [activation_weighted_svd(W, X; target_rank=target_rank) for (W, X) in zip(weights, activs)]
    W_compressed = [res.W_compressed for res in aw_results]

    # Step 5: Trim UΣ using routing frequency
    trimmed = u_matrix_trim(W_compressed, route_freqs, k_top)

    # Step 6: Rebuild SVDs from truncated weights for V-sharing
    svd_data = [svd(Wc) for Wc in W_compressed]
    v_share_result = v_matrix_sharing(svd_data, route_freqs)
    V_shared = v_share_result.V_shared

    # Step 7: Rebuild final experts using UΣ_combined and top-r slice of V_shared
    experts = [UΣ * V_shared[:, 1:size(UΣ, 2)] for (UΣ, _, _) in trimmed]

    return (
        compressed = true,
        experts = experts,
        V_shared = V_shared,
        S_L = S_L
    )
end

# Dummy test
function run_dummy_layer_compress_test()
    weights = [randn(128, 64) for _ in 1:4]
    activs  = [randn(64, 64) for _ in 1:4]
    freqs   = rand(4)

    result = compress_layer(weights, activs, freqs;
                            sens_thresh=100.0)

    println("Was layer compressed? ", result.compressed)
    println("Shape of first expert: ", size(result.experts[1]))
end

# Uncomment to run test
run_dummy_layer_compress_test()
