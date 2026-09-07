---
license: apache-2.0
metrics:
- mse
- mae
- mase
- wql
- crps
pipeline_tag: time-series-forecasting
datasets:
- thuml/UTSD
- Salesforce/lotsa_data
- Salesforce/GiftEvalPretrain
- autogluon/chronos_datasets
tags:
- time series
- time-series
- forecasting
- foundation models
- pretrained models
- time series foundation models
library_name: transformers
---

# Timer-S1

Timer-S1 is a time series foundation model with **8.3B** total parameters, **0.75B** activated parameters per token, and a context length of  **11,520**.

The model supports **zero-shot forecasting** (predicting without dataset-specific training) at different quantile levels.

For more details, please refer to our [technical report](https://arxiv.org/pdf/2603.04791).

![image](https://cdn-uploads.huggingface.co/production/uploads/64fbe24a2d20ced4e91de38a/7Udz1nO2V1Nk0pw5cW4gG.png)

**Architecture**: Timer-S1 is a decoder-only Mixture-of-Experts (MoE) Transformer. For time series forecasting (a sequential problem where each step depends on previous ones), we propose **TimeSTP**, enabling multi-step prediction with cost-effective **serial computations**.
![image](https://cdn-uploads.huggingface.co/production/uploads/64fbe24a2d20ced4e91de38a/1XsUZDPw8DJebZwH-Ievd.png)

**Performance**: Timer-S1 achieves state-of-the-art results on [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval). The model excels particularly at **medium-term** and **long-term** forecasting tasks.

![image](https://cdn-uploads.huggingface.co/production/uploads/64fbe24a2d20ced4e91de38a/XDOekWBIGBoc8nTDI-WBI.png)

![image](https://cdn-uploads.huggingface.co/production/uploads/64fbe24a2d20ced4e91de38a/r7eGVKBIRI8h7lMre4-lP.png)

**Post Training**: Timer-S1 undergoes post-training, including continued pre-training (**CPT**) and long-context extension (**LCE**), which improves short-term and long-context performance.

![image](https://cdn-uploads.huggingface.co/production/uploads/69ce7cea1430d60211285e20/9KqUVPPkA6DMr_EnhpD_O.png)


## Quickstart

```
pip install torch accelerate transformers~=4.57.1
```

```python
import torch
from transformers import AutoModelForCausalLM

# load pretrain model
# supports different lookback/forecast lengths
model = AutoModelForCausalLM.from_pretrained(
    'thuml/Timer-S1',
    trust_remote_code=True,
    device_map="auto"
)

# use local model
# model = AutoModelForCausalLM.from_pretrained(
#     'path_to_timer_s1',
#     trust_remote_code=True,
#     device_map="auto"
# )

# prepare input
batch_size, lookback_length = 64, 11520 
seqs = torch.randn(batch_size, lookback_length).to(model.device)

# Note that Timer-S1 generates predictions at fixed quantile levels
forecast_length = 256

output = model.generate(seqs, max_new_tokens=forecast_length, revin=True)

# produce quantile forecasts in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print(output.shape) # batch_size x quantile_num(9) x forecast_length

# produce the median forecast of the first sample
print(output[0][4])
```


This model support inference using either CPU or GPU. To load this model on GPU, we recommend a GPU with **at least 40GB VRAM** (e.g., A100 40GB/80GB, or H100). 

>  **Encounter out-of-memory at runtime?** Try the following options:
> ```python
> # Option 1: reduce batch size or context length
> batch_size, lookback_length = 1, 2880
>
> # Option 2: disable KV cache at runtime (or edit it in config.json for a permanent change)
> model.config.use_cache = False # there is no efficiency impact for cases where the prediction horizon does not exceed 256.
> ```

## Specification

* **Architecture**: decoder-only Transformer with MoE
* **Context Length**: up to 11,520
* **ReNorm**: default=True
* **KV Cache**: default=True
* **Patch Length**: 16
* **Total Parameters**: 8.3B
* **Activated Parameters**: 0.75B
* **Number of Layers**: 40


## License Agreement

This model is licensed under the Apache-2.0 License.

## Citation

If you find Timer-S1 helpful for your research, please cite our paper:
```
@article{liu2026timer,
  title={Timer-S1: A Billion-Scale Time Series Foundation Model with Serial Scaling},
  author={Liu, Yong and Su, Xingjian and Wang, Shiyu and Zhang, Haoran and Liu, Haixuan and Wang, Yuxuan and Ye, Zhou and Xiang, Yang and Wang, Jianmin and Long, Mingsheng},
  journal={arXiv preprint arXiv:2603.04791},
  year={2026}
}
```