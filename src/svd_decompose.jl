# svd_decompose.jl

# For each expert’s weight matrix Wᵢ:
# - Runs SVD to get singular values
# - Flags values ≥ threshold_ratio × max(Σ)
# - threshold_ratio is typically 0.01 (1%)
# - Counts how many singular values pass → defines principal rank rᵢ
# - rᵢ reflects how complex or information-rich expert i is

using LinearAlgebra

function compute_principal_ranks(weights::Vector{<:AbstractMatrix}, threshold_ratio::Float64)::Vector{Int}
    ranks = Int[]

    for W in weights
        # Step 1: Run SVD on expert's weight matrix
        svd_result = svd(W)

        # Step 2: Get singular values
        Σ = svd_result.S

        # Step 3: Set threshold = threshold_ratio × max(singular value)
        threshold = threshold_ratio * maximum(Σ)

        # Step 4: Count how many singular values ≥ threshold → this is rᵢ
        push!(ranks, count(s -> s ≥ threshold, Σ))
    end

    return ranks
end

# Test function with dummy weight matrices
function run_dummy_svd_test()
    num_experts = 4
    weights = [randn(128, 64) for _ in 1:num_experts]
    threshold_ratio = 0.01

    println("SVD Principal Ranks per Expert:")
    ranks = compute_principal_ranks(weights, threshold_ratio)

    for (i, W) in enumerate(weights)
        Σ = svd(W).S
        println("Expert $i: Singular Values (rounded) = ", round.(Σ, digits=2))
        println("→ rᵢ = ", ranks[i])
    end
end

# Uncomment to run standalone
run_dummy_svd_test()
