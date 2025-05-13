# gating.jl
using Flux
using Statistics

# Gating mechanism in a Mixture-of-Experts (MoE) model
# This module routes input tokens to a subset of experts based on learned scores

mutable struct GatingLayer
    dense::Dense               # Dense layer computes gating scores per expert
    k::Int                     # Top-k expert selection (e.g., 2 experts per token)
    expert_counts::Vector{Int}  # Tracks how often each expert is routed to
end

# Creates the gating layer; input = token embedding, output = score per expert
function GatingLayer(input_dim::Int, num_experts::Int, k::Int)
    dense = Dense(input_dim, num_experts)
    expert_counts = zeros(Int, num_experts)
    return GatingLayer(dense, k, expert_counts)
end

# Resets routing frequency stats before reuse or recalibration
function reset!(g::GatingLayer)
    fill!(g.expert_counts, 0)
end

# Top-k expert selection based on gating scores
function top_k_experts(scores::AbstractMatrix, k::Int)
    return mapslices(col -> partialsortperm(col, rev=true, 1:k), scores; dims=1)
end

# Forward pass of the gating layer
function (g::GatingLayer)(x::AbstractMatrix)
    # Input: matrix of token embeddings [input_dim, batch_size]

    # Dense layer projects token embeddings to expert scores
    scores = softmax(g.dense(x))  # Softmax converts scores to [0,1]; normalized gate scores

    # Selects top-k experts with highest softmax scores per input
    topk_indices = top_k_experts(scores, g.k)

    # Tracks expert selection frequency across inputs
    for idxs in eachcol(topk_indices)
        for i in idxs
            g.expert_counts[i] += 1
        end
    end

    # Outputs top-k expert indices and raw softmax scores
    return (topk_indices=topk_indices, scores=scores)
end

# Returns expert routing frequency normalized over all inputs
function normalized_counts(g::GatingLayer)
    total = sum(g.expert_counts)
    total == 0 && return zeros(Float64, length(g.expert_counts))
    return g.expert_counts ./ total  # Routing freq used in compression (e.g., trimming)
end

# Example calibration run on dummy data
function run_gating_calibration()
    input_dim = 10
    num_experts = 4
    top_k = 2
    gating = GatingLayer(input_dim, num_experts, top_k)

    # Calibration dataset: small sample set to observe routing behavior
    for _ in 1:100
        x = rand(Float32, input_dim, 5)  # Batch of 5 token embeddings
        gating(x)
    end

    println("Normalized Expert Routing Frequencies:")
    println(normalized_counts(gating))  # Key signal for compression decisions
end

# Uncomment to run standalone
run_gating_calibration()
