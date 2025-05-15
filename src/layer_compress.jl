# layer_compress.jl

include("activation_stats.jl")
include("svd_decompose.jl")
include("sensitivity_metrics.jl")
include("u_matrix_trimming.jl")
include("v_matrix_sharing.jl")
include("aw_svd.jl")

using LinearAlgebra

function compress_layer(
        weights::Vector{Matrix{Float64}},
        activs::Vector{Matrix{Float64}},
        route_freqs::Vector{Float64};
        k_top::Int             = 2,
        tau::Float64           = 2.0,
        rank_thresh::Float64   = 0.6,
        target_rank::Union{Int,Nothing} = 16,
        sens_thresh::Float64   = 50.0
    )

    a = compute_outlier_ratios(activs, tau)
    r = compute_principal_ranks(weights, rank_thresh)
    S_L = compute_layer_sensitivity(route_freqs, r, a)

    println("Sensitivity Score S_L = ", round(S_L, digits=2))

    if S_L ≥ sens_thresh
        println("Layer skipped (high sensitivity).")
        return (compressed = false, experts = weights, V_shared = nothing, S_L = S_L)
    end

    println("Compressing layer (low sensitivity).")

    aw_results = [activation_weighted_svd(W, X; target_rank=target_rank) for (W, X) in zip(weights, activs)]
    W_compressed = [res.W_compressed for res in aw_results]

    trimmed = u_matrix_trim(W_compressed, route_freqs, k_top)
    svd_data = [svd(Wc) for Wc in W_compressed]
    v_share_result = v_matrix_sharing(svd_data, route_freqs)
    V_shared = v_share_result.V_shared

    experts = [UΣ * V_shared[:, 1:target_rank] for (UΣ, _, _) in trimmed]

    return (compressed = true, experts = experts, V_shared = V_shared, S_L = S_L)
end

function run_dummy_layer_compress_test()
    println("MoE-SVD Compression Test")

    # Use small-valued weights for clearer singular value decay
    weights = [0.01 * randn(128, 64) for _ in 1:4]
    activs  = [0.01 * randn(64, 64) for _ in 1:4]
    freqs   = rand(4)

    result = compress_layer(weights, activs, freqs;
                            sens_thresh = 100.0,
                            rank_thresh = 0.01,
                            target_rank = 16)

    for i in 1:2
        original_shape = size(weights[i])
        compressed_shape = size(result.experts[i])
        original_svals = svd(weights[i]).S
        compressed_svals = svd(result.experts[i]).S
        retained_original = sum(original_svals .> 0.01 * maximum(original_svals))
        retained_compressed = sum(compressed_svals .> 1e-6)

        println("Expert $i Original Shape: $original_shape")
        println("Expert $i Compressed Shape: $compressed_shape")
        println("Expert $i Original Rank (σ > 1% max): $retained_original")
        println("Expert $i Compressed Rank (σ > 1e-6): $retained_compressed")
        println("---")
    end
end

# Uncomment to run
run_dummy_layer_compress_test()
