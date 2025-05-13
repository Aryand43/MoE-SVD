# v_matrix_sharing.jl

using LinearAlgebra

"""
    v_matrix_sharing(svd_components::Vector{<:SVD}, frequencies::Vector{Float64})

This function implements V-matrix sharing:
- Computes router sampling frequency for each expert.
- Selects the Vᵀ matrix from the most frequently used expert.
- Updates all expert decompositions to use this shared Vᵀ.

Returns a named tuple:
    (experts = updated_experts, V_shared = shared_Vᵀ)
"""
function v_matrix_sharing(svd_components::Vector{<:SVD}, frequencies::Vector{Float64})
    N = length(svd_components)
    @assert N == length(frequencies) "Number of SVD components must match frequency vector"

    # Frequencies can be raw counts or normalized — only their relative sizes matter
    max_idx = argmax(frequencies)  # Find most frequently used expert

    # Use Vᵀ from the highest-frequency expert
    shared_Vᵀ = svd_components[max_idx].V'

    updated_experts = []

    # Rebuild each expert using its U and Σ, and the shared Vᵀ
    for i in 1:N
        U = svd_components[i].U
        Σ = Diagonal(svd_components[i].S)
        Eᵢ = U * Σ * shared_Vᵀ
        push!(updated_experts, Eᵢ)
    end

    return (experts = updated_experts, V_shared = shared_Vᵀ)
end

# Dummy test
function run_dummy_v_sharing_test()
    weights = [randn(128, 64) for _ in 1:4]
    frequencies = [0.15, 0.30, 0.10, 0.45]
    svd_data = [svd(W) for W in weights]

    result = v_matrix_sharing(svd_data, frequencies)

    println("Shape of shared Vᵀ: ", size(result.V_shared))
    println("Shape of updated expert 1 matrix: ", size(result.experts[1]))

    x = randn(64)
    println("Expert 1 output (after V-matrix sharing): ", result.experts[1] * x)
end

# Uncomment to run test
run_dummy_v_sharing_test()
