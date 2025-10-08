---
language:
- en
license: cc-by-nc-4.0
task_categories:
- text-generation
configs:
- config_name: default
  data_files: "*.jsonl"
---

<div align="center">
<span style="font-family: default; font-size: 1.5em;">ClimbMix Dataset</span>
<div>
🚀 Creating the highest-quality pre-training datasets for LLMs 🌟
</div>
</div>

<div style="display: flex; gap: 10px; margin-top: 15px; justify-content: center;">
  <a href="https://arxiv.org/abs/2504.13161" style="display: inline-block; background-color: #0d1117; color: white; text-decoration: none; padding: 10px 20px; border-radius: 4px;">
    📄 PAPER
  </a>
  <a href="https://huggingface.co/datasets/nvidia/ClimbLab" style="display: inline-block; background-color: #0d1117; color: white; text-decoration: none; padding: 10px 20px; border-radius: 4px;">
    🤗 CLIMBLAB
  </a>
  <a href="https://huggingface.co/datasets/nvidia/ClimbMix" style="display: inline-block; background-color: #0d1117; color: white; text-decoration: none; padding: 10px 20px; border-radius: 4px;">
    🤗 CLIMBMIX
  </a>
  <a href="https://research.nvidia.com/labs/lpr/climb/" style="display: inline-block; background-color: #0d1117; color: white; text-decoration: none; padding: 10px 20px; border-radius: 4px;">
    🏠 HOMEPAGE
  </a>
</div>


<table>
  <tr>
    <td align="center">
      <img src="assets/cont_pretrain.png" width="300"/><br/>
      <sub><b>Figure 1:</b> Continuously training a 1B model yields a 2.0% improvement over Llama-3.2-1B, demonstrating a more efficient scaling trend compared to prior models. </sub>
    </td>
    <td align="center">
      <img src="assets/pretrain_from_scratch.png" width="360"/><br/>
      <sub><b>Figure 2:</b> Pre-training a 1B model from scratch on ClimbMix shows better scaling effects than training on other datasets. </sub>
    </td>
  </tr>
</table>

## Dataset Description
ClimbMix is a compact yet powerful 400-billion-token dataset designed for efficient pre-training that delivers superior performance under an equal token budget.  It was introduced in [this paper](https://huggingface.co/papers/2504.13161).
We proposed a new algorithm to filter and mix the dataset. First, we grouped the data into 1,000 groups based on topic information. Then we applied two classifiers: one to detect advertisements and another to assess the educational value of the text. Each group was scored accordingly, and low-quality data with low scores was removed. Finally, the remaining high-quality groups were mixed using certain weights to generate the final dataset.

This dataset is for research and development only.

## Dataset Details

* **Owner(s):** NVIDIA
* **Creation Date:** Feb. 1, 2025
* **License/Terms of Use:** CC BY-NC 4.0
* **Intended Usage:** Pre-training language models.
* **Format:** Text in parquet format
* **Size:** 400 billion tokens
* **Data Collection Method:** Automated
* **Labeling Method:** Automated

## Usage

The ClimbMix dataset we released contains token sequences that have been tokenized using the GPT-2 tokenizer. If you wish to obtain the raw text, please use the provided script `detokenize_climbmix.py`. For example:

```bash
python detokenize_climbmix.py --input_folder <tokenized_folder> --output_folder <raw_text_folder>
```

We also noticed that some community members have converted and released a raw text version of ClimbMix on Hugging Face: https://huggingface.co/datasets/OptimalScale/ClimbMix. You may consider using this version to save the effort of manual conversion. However, please note that this is not the official release, and we are not responsible for the content or maintenance of community-hosted datasets.


## Training

To help reproduce the results, we provide the training script for ClimbMix in `nanoGPT/train.sh`. The code is based on the [nanoGPT](https://github.com/karpathy/nanoGPT) project and we do not make any changes to the model definition and training process. The main changes are:

1. Preprocessed and tokenized the ClimbMix dataset in `nanoGPT/data/climbmix/prepare.sh`.
2. Modified the training configuration in `nanoGPT/config/train_gpt2_climbmix.py`.

Note: in our paper, we used Llama-2 tokenizer and Llama-2 model architecture, so the results are different but we verified that the scaling trend against other public datasets is the same.

Here we display the training curves of the `gpt-2-xl` model on ClimbMix and other datasets. The validation data is openwebtext. With the above script, you could easily reproduce the results.

<img src="assets/wandb.png" width="500"/>

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## Citation:
If you find our dataset helpful, please cite the following [paper](https://arxiv.org/abs/2504.13161):

```
@article{diao2025climb,
  author    = {Shizhe Diao and Yu Yang and Yonggan Fu and Xin Dong and Dan Su and Markus Kliegl and Zijia Chen and Peter Belcak and Yoshi Suhara and Hongxu Yin and Mostofa Patwary and Celine Lin and Jan Kautz and Pavlo Molchanov},
  title={CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training}, 
  journal   = {arXiv preprint},
  year      = {2025},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url={https://arxiv.org/abs/2504.13161}, 
}
```