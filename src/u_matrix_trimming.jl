# u_matrix_trimming.jl

using LinearAlgebra

"""
    u_matrix_trim(weights::Vector{Matrix{Float64}}, frequencies::Vector{Float64}, k::Int)

Reduces model complexity while maintaining diversity by combining the top-k 
frequently routed experts' (U * Σ) components for each expert.

Returns a vector of tuples: 
    (UΣ_combined, Vᵀ, full_combined_matrix) 
where full_combined_matrix = UΣ_combined * Vᵀ
"""
function u_matrix_trim(weights::Vector{Matrix{Float64}}, frequencies::Vector{Float64}, k::Int)
    N = length(weights)
    @assert N == length(frequencies) "weights and frequencies must match"

    # Precompute SVD for each expert
    svd_data = [svd(W) for W in weights]

    # Store final compressed representations
    compressed_experts = []

    for i in 1:N
        f_i = frequencies[i]

        # We gather all experts that are routed as often or more frequently than expert i
        eligible_indices = findall(j -> frequencies[j] ≥ f_i, 1:N)

        # Efficient top-k selection based on routing frequency
        if length(eligible_indices) <= k
            selected = eligible_indices
        else
            topk_indices = partialsortperm(eligible_indices, k, by = j -> frequencies[j], rev=true)
            selected = eligible_indices[topk_indices]
        end
        # UΣ_combined will store the weighted sum of top experts' U * Σ
        UΣ_combined = zero(weights[i])

        # Combine selected U * Σ matrices, scaled by each expert's routing frequency
        for j in selected
            U, Σ = svd_data[j].U, Diagonal(svd_data[j].S)
            UΣ_combined += frequencies[j] * (U * Σ)
        end

        # We retain the Vᵀ from the current expert for consistency
        Vᵀ = svd_data[i].V'

        # Full combined matrix = (U₁Σ₁ + U₂Σ₂) * Vᵀ
        combined_matrix = UΣ_combined * Vᵀ

        push!(compressed_experts, (UΣ_combined, Vᵀ, combined_matrix))
    end

    return compressed_experts
end

# Test Example
function run_dummy_u_trim_test()
    weights = [randn(128, 64) for _ in 1:4]
    frequencies = rand(4)
    k = 2

    compressed = u_matrix_trim(weights, frequencies, k)

    # Show shapes and a sample output for expert 1
    UΣ, Vᵀ, W_trimmed = compressed[1]
    println("UΣ_combined shape: ", size(UΣ))
    println("Vᵀ shape: ", size(Vᵀ))
    println("Combined Matrix shape: ", size(W_trimmed))

    x = randn(64)
    println("Output from compressed expert 1: ", W_trimmed * x)
end

# Uncomment to run test
run_dummy_u_trim_test()
