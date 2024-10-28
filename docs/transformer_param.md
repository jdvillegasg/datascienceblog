# Global parameters

- `d_model`: $M$
- `batch_size`: $B$
- `vocab_mz_size`: $Z_{max}$
- Max. number of characters in SMILES numbered encoded: $SM_{max} = 100$
- `d_k`: $D_{K}$
- `d_v`: $D_{V}$
- `n_heads`: $H$
- `d_ff`: $F$
- `n_layers `: $N_{L}$

# torch.nn.Embedding

## Parameters:

- `num_embeddings`: $E = Z_{max}$
- `embedding_dim`: $D_{E} = M$

## Inputs:

- $x \in \mathbb{R}^{B \times SM_{max}}$

## Outputs:

- $x \in \mathbb{R}^{B \times SM_{max} \times M}$


# Positional Encoder:

## Parameters:

- `d_model`: $D_{P} = M$

## Inputs

- $x \in \mathbb{R}^{SM_{max} \times B \times M}$ 

## Outputs

- $x \in \mathbb{R}^{SM_{max} \times B \times M}$ 

# `get_attn_pad_mask`

## Inputs:

- $x_{1} \in \mathbb{R}^{B \times L_{Q}}$, with $L_{Q} = SM_{max}$
- $x_{2} \in \mathbb{R}^{B \times L_{K}}$, with $L_{k} = SM_{max}$

## Outputs

- $x \in \mathbb{R}^{B \times SM_{max} \times SM_{max}}$

# MultiHeadAttention

## Inputs

- $Q \in \mathbb{R}^{SM_{max} \times B \times M}$
- $V \in \mathbb{R}^{SM_{max} \times B \times M}$
- $K \in \mathbb{R}^{SM_{max} \times B \times M}$

## Linear layers

### Inputs

- `input_Q` $ \in \mathbb{R}^{SM_{max} \times B \times M}$
- `input_K` $ \in \mathbb{R}^{SM_{max} \times B \times M}$
- `input_V` $ \in \mathbb{R}^{SM_{max} \times B \times M}$

### Outputs

- $Q \in \mathbb{R}^{B \times SM_{max} \times D_{K}*H}$
- $K \in \mathbb{R}^{B \times SM_{max} \times D_{K}*H}$
- $V \in \mathbb{R}^{B \times SM_{max} \times D_{V}*H}$

## Dimensionality handling 1

### Inputs

- $Q \in \mathbb{R}^{B \times SM_{max} \times D_{K}*H}$
- $K \in \mathbb{R}^{B \times SM_{max} \times D_{K}*H}$
- $V \in \mathbb{R}^{B \times SM_{max} \times D_{V}*H}$

### Outputs

- $Q \in \mathbb{R}^{B \times H \times SM_{max} \times D_{K}}$
- $K \in \mathbb{R}^{B \times H \times SM_{max} \times D_{K}}$
- $V \in \mathbb{R}^{B \times H \times SM_{max} \times D_{V}}$

## Dimensionality handling 2

### Inputs

- $x \in \mathbb{R}^{B \times SM_{max} \times SM_{max}}$

### Outputs

- $x \in \mathbb{R}^{B \times H \times SM_{max} \times SM_{max}}$

## ScaledDotProductAttention

### MatMul

**Inputs**

- $Q \in \mathbb{R}^{B \times H \times SM_{max} \times D_{K}}$
- $K \in \mathbb{R}^{B \times H \times SM_{max} \times D_{K}}$


**Outputs**

- $QK^{T} \in \mathbb{R}^{B \times H \times SM_{max} \times SM_{max}}$

### Softmax

**Inputs**

- $x \in \mathbb{R}^{B \times H \times SM_{max} \times SM_{max}}$

**Outputs**

- $x \in \mathbb{R}^{B \times H \times SM_{max} \times SM_{max}}$

### MatMul

**Inputs**

- $x \in \mathbb{R}^{B \times H \times SM_{max} \times SM_{max}}$
- $V \in \mathbb{R}^{B \times H \times SM_{max} \times D_{V}}$

**Outputs**

- $x \in \mathbb{R}^{B \times H \times SM_{max} \times D_{V}}$

## Dimensionality handling 3 (transpose, reshape)

**Inputs**

- $x \in \mathbb{R}^{B \times H \times SM_{max} \times D_{V}}$

**Outputs**

- $x \in \mathbb{R}^{B \times SM_{max} \times H*D_{V}}$

## Linear Layer

### Inputs

- $x \in \mathbb{R}^{B \times SM_{max} \times H*D_{V}}$

### Outputs

- $x \in \mathbb{R}^{B \times SM_{max} \times M}$

# PoswiseFeedForwardNet

## Inputs

- $x \in \mathbb{R}^{B \times SM_{max} \times M}$

## Linear Layer

### Inputs

- $x \in \mathbb{R}^{B \times SM_{max} \times M}$

### Outputs

- $x \in \mathbb{R}^{B \times SM_{max} \times F}$

## ReLu

### Inputs

- $x \in \mathbb{R}^{B \times SM_{max} \times F}$

### Outputs

- $x \in \mathbb{R}^{B \times SM_{max} \times F}$

## Linear Layer

### Inputs

- $x \in \mathbb{R}^{B \times SM_{max} \times F}$

### Outputs

- $x \in \mathbb{R}^{B \times SM_{max} \times M}$