# BiasLens

This is a repo for the pipeline to detect bias.

## Author

This pipeline has been developed by:
- Chu Lee Shen

## Quickstart

1. Open terminal and clone the repository:
```
git clone https://github.com/chuleeshen/bias-pipeline.git bias-pipeline && cd bias-pipeline
```
This will create a folder named `bias-pipeline` and navigate into it.


2. Move up one directory level:
```
cd ..
```

3. Ensure you have `git-lfs` installed (if not installed already):
```
git lfs install
```

4. Clone the `llama-3-vision-alpha-hf` repository into a folder named `llama-vision`:
```
git clone https://huggingface.co/qresearch/llama-3-vision-alpha-hf llama-vision
```

5. Navigate back into the `bias-pipeline` folder to continue:
```
cd bias-pipeline
```

6. Before proceeding, you need to install **Streamlit**, run in terminal:

```bash
pip install streamlit
```

7. Open the secrets file for editing:
```
nano .streamlit/secrets.toml
```

8. Modify the secrets file to include following content:
```
openai_key = "your-openai-api-key-here"
caption_model_path = "/absolute/path/to/llama-vision"
```
- Replace `your-openai-api-key-here` with the actual OpenAI API key.
- Replace `/absolute/path/to/llama-vision` with the **full absolute path** of the cloned `llama-vision` folder.

9. You can now start the Streamlit app by running:
```
streamlit run app.py
```

---

**HPC Processing Specs** \
GPU Profile: 3g.40gb\
CPUs: 16\
RAM: 64GB

> NOTE: These are the settings when using the HPC infrastructure by Monash University Malaysia. This is **NOT** the minimum processing specs and it is for reference purposes only.
