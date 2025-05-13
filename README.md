# MoE-SVD: Mixture-of-Experts Compression via SVD

*MoE-SVD is a modular Julia framework for compressing Mixture-of-Experts (MoE) models with Singular Value Decomposition.  
It implements the core ideas from the MoE-SVD paper to shrink model size while preserving representational capacity and expert diversity.*

---

## Modules

### `gating.jl`
* Computes routing frequencies **fᵢ** for each expert using a dense soft-max gating layer.  
* Tracks top-k expert selections and normalized routing counts.

### `activation_stats.jl`
* Calculates mean absolute activations and outlier ratios **aᵢ**.  
* Quantifies how “spiky” or extreme each expert’s response is.

### `svd_decompose.jl`
* Finds the principal rank **rᵢ** for each expert’s weight matrix via SVD.  
* Retains only singular values above a configurable threshold.

### `sensitivity_metrics.jl`
* Computes the layer sensitivity score  

  \[
    S_L = \sum_i fᵢ \times rᵢ \times aᵢ
  \]

* Guides the decision to compress or skip a layer.

### `u_matrix_trimming.jl`
* Applies top-k **U**-matrix trimming based on routing frequency.  
* Preserves the most diverse expert directions with a weighted **UΣ** combination.

### `v_matrix_sharing.jl`
* Selects the most frequently used **V**-matrix across experts.  
* Replaces all expert **V** matrices with this shared one to cut memory use.

### `aw_svd.jl`
* Implements activation-weighted SVD using a Cholesky factor of the activation Gram matrix.  
* Optionally truncates to a target rank and returns the compressed expert representation.

### `compression_runner.jl`
* Orchestrates the full MoE-SVD compression pipeline—trimming, sharing, and compression decisions.

### `layer_compress.jl`
* Wraps all logic needed to compress a single MoE feed-forward layer end-to-end.

---

## Usage

### Requirements
* Julia ≥ 1.6  
* Standard libraries: `LinearAlgebra`, `Statistics`

### Run the Full Pipeline

1. **Uncomment** the dummy test call at the bottom of `compression_runner.jl`:

   ```julia
   # Build fresh SVD objects from already-compressed weights
   run_dummy_compression_test()
