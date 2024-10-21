
Workflow:

1. Normalize
2. (Optional) Trend removal
2. Pass through model
3. Denormalize

# Deep Time Series Models: a comprehensive survey and benchmark

## Architecture classes

- MLP-based
- RNN-based
- CNN-based
- GNN-based
- Transformer-based

### MLP-based

- [DLinear](https://arxiv.org/pdf/2205.13504) (In Time-Series-Library)
- [TiDE](https://arxiv.org/pdf/2304.08424) (In Time-Series-Library)
- [FreTS](https://arxiv.org/pdf/2311.06184) (In Time-Series-Library)
- [Koopa](https://arxiv.org/pdf/2305.18803) (In Time-Series-Library)
- [TimeMixer](https://openreview.net/pdf?id=7oLshfEIC2) (In Time-Series-Library)

### RNN-based

- [LSTNet](https://arxiv.org/pdf/1703.07015) 
- [Mamba](https://arxiv.org/pdf/2312.00752) (In Time-Series-Library)

## CNN-based

- [TimesNet](https://openreview.net/pdf?id=ju_Uqw384Oq) (In Time-Series-Library)
- [MICN](https://openreview.net/pdf?id=zt53IDUR1U) (In Time-Series-Library)

## GNN-based

- [MTGNN](https://arxiv.org/pdf/2005.11650)
- [StemGNN](https://arxiv.org/pdf/2103.07719)

## Transformer-based

- [Autoformer](https://openreview.net/pdf?id=I55UqU-M11y) (In Time-Series-Library)
- [Informer](file:///home/julian/Downloads/17325-Article%20Text-20819-1-2-20210518.pdf) (In Time-Series-Library)
- [FEDFormer](https://proceedings.mlr.press/v162/zhou22g/zhou22g.pdf) (In Time-Series-Library)
- [Crossformer](https://openreview.net/pdf?id=vSVLM2j9eie) (In Time-Series-Library)
- [PatchTST](https://openreview.net/pdf?id=Jbdc0vTOcol) (In Time-Series-Library)
- [iTransformer](https://arxiv.org/pdf/2310.06625) (In Time-Series-Library)

 
## Effectiveness of deep architectures

!!! tip "Important quote"
    "Recent research by DLinear [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504), also referred to as LTSF-Linear, challenges the effectiveness of complicated deep architecture in temporal modeling".