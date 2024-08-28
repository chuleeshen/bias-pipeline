# Bias Pipeline

This is a repo for the pipeline to detect bias.

---

⚠️ **IMPORTANT**⚠️

This code is work in progress and experimental.

---

## Author

This pipeline has been developed by:
- Chu Lee Shen

## Development
### Prerequisites
In order for the image captioning part (LLaMA 3 with vision capabilities using SIGLIP) to work, you need to clone the [llama-3-vision-alpha-hf](https://huggingface.co/qresearch/llama-3-vision-alpha-hf/tree/main) model:

```
# Make sure you have git-lfs installed
git lfs install
```
```
git clone https://huggingface.co/qresearch/llama-3-vision-alpha-hf
```
> NOTE: For the section *Generated Images VQA*, MiniGPT-v2 and MiniGPT 4 is not working at the moment. Those are part of my tests for different image captioning tools :smile:

**HPC Processing Specs** \
GPU Profile: 3g.40gb\
CPUs: 16\
RAM: 64GB

> NOTE: These are the settings that I used when using Monash HPC. This is **NOT** the minimum processing specs!
