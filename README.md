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

> NOTE: Currently at the section for Generated Images VQA, set up MiniGPT-v2 and MiniGPT 4 is currently not working properly and is just part of my experiments on different image captioning tools :smile:
