# aw_svd.jl

using LinearAlgebra

"""
    activation_weighted_svd(W, X; target_rank=nothing)

Performs activation-weighted SVD:
1. Scales weight matrix W using Cholesky of X Xᵀ.
2. Runs SVD on scaled weights.
3. Optionally truncates to target rank.
4. Returns compressed components and inverse scale.

Returns a NamedTuple:
(U, Σ, Vt, Sinv, W_compressed)
"""
function activation_weighted_svd(
        W::Matrix{Float64},
        X::Matrix{Float64};
        target_rank::Union{Int,Nothing}=nothing
    )

    # Step 1 – compute scaling matrix (activations' Gram matrix)
    S = cholesky(X * X').U

    # Step 2 – scale the weights
    W_aw = W * S

    # Step 3 – run SVD on the scaled matrix
    U, σ, V = svd(W_aw)

    # Step 4 – truncate to desired rank (safety: cap at length of σ)
    r = target_rank === nothing ? length(σ) : min(target_rank, length(σ))
    U_r = U[:, 1:r]
    Σ_r = Diagonal(σ[1:r])
    # Keep the top-r right singular vectors
    V_r = V[:, 1:r]

    # Safer inversion via triangular solve
    Sinv = UpperTriangular(S) \ I(size(S, 1))

    # Final compressed matrix
    W_compressed = U_r * Σ_r * V_r' * Sinv

    return (
        U = U_r,
        Σ = Σ_r,
        Vt = V_r',
        Sinv = Sinv,
        W_compressed = W_compressed
    )
end

# Dummy test
function run_dummy_aw_svd_test()
    W = randn(128, 64)
    X = randn(64, 64)
    result = activation_weighted_svd(W, X, target_rank=16)

    println("Shape of U: ", size(result.U))
    println("Shape of W_compressed: ", size(result.W_compressed))
end

# Uncomment to test
run_dummy_aw_svd_test()
