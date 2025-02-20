# Partial Embedding Matrix Adaptation

<p align="center">
    <img src="images/figure.svg" alt="Diagram of Partial Embedding Matrix Adaptation" />
</p>

## Introduction

This repository contains the implementation for the paper [Vocabulary-level Memory Efficiency for Language Model Fine-tuning](https://arxiv.org/abs/2309.08708).

Partial Embedding Matrix Adaptation is a simple technique that can reduce the memory footprint of language model fine-tuning without impacting performance.

## Installation

```
pip install git+https://github.com/mlsw/partial-embedding-matrix-adaptation.git
```

## Usage

### Hugging Face Transformers

There is a high-level API for Hugging Face Transformers PyTorch models via the [`HFEmbeddingPruner`](src/partial_embedding_matrix_adaptation/hf_embedding_pruner.py) class. 

```python
from partial_embedding_matrix_adaptation import HFEmbeddingPruner

embedding_pruner = HFEmbeddingPruner(model)
dataset, _ = embedding_pruner.prepare_model(tokenizer, dataset)
```

Please see [examples/distilbert_sst2.py](examples/distilbert_sst2.py) for a complete example. Additionally, the scripts in the [utils](utils) directory show how to use this API with the Hugging Face Transformers [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer).

### PyTorch

Alternatively, the [`EmbeddingPruner`](src/partial_embedding_matrix_adaptation/embedding_pruner.py) class can be used directly for PyTorch models. Please see [`HFEmbeddingPruner`](src/partial_embedding_matrix_adaptation/hf_embedding_pruner.py) for an example of how to use this.

## Reproducibility

The following scripts can be used to reproduce the results from the paper. These are adapted from Hugging Face Transformers [PyTorch Examples](https://github.com/huggingface/transformers/tree/v4.37.2/examples/pytorch) with support for Partial Embedding Matrix Adaptation.

| Task | Script | Documentation |
| ---- | ------ | --------------|
| GLUE | [run_glue_pema.py](utils/run_glue_pema.py) | [Here](https://github.com/huggingface/transformers/tree/v4.37.2/examples/pytorch/text-classification#glue-tasks) |
| XNLI | [run_xnli_pema.py](utils/run_xnli_pema.py) | [Here](https://github.com/huggingface/transformers/tree/v4.37.2/examples/pytorch/text-classification#xnli) |

## License

This project is licensed under the terms of the MIT license. Please see [LICENSE](LICENSE) for more details.

## Citation

If you found this work useful, please consider citing our paper:

```bibtex
@misc{williams-aletras-2025-vocabulary,
  title={Vocabulary-level Memory Efficiency for Language Model Fine-tuning}, 
  author={Miles Williams and Nikolaos Aletras},
  year={2025},
  eprint={2309.08708},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2309.08708}
}
```
