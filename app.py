import streamlit as st
from background import retrieve_keywords, sys_prompt_inst, gpt_4_api, convert_dict, setup_tti, setup_image_caption, all_adj_noun_results, download_nltk_resources, generate_csv_with_matches, compute_statistics, plot_bias_frequencies
from openai import OpenAI
import shutil

st.title("BiasLens")
st.write("A detection tool for potential biases in images generated from Text-to-Image (T2I) models")

prompt = st.text_input("Prompt", placeholder="Enter the prompt for image generation")

number_of_images = st.number_input(
  "Number of images", 
  min_value=2, 
  max_value=100, 
  value=10, 
  step=1,
  help="Enter the number of images to generate for bias detection (minimum 2)"
)

specific_bias = st.text_input("Specific Bias (optional)", placeholder="Enter a specific bias to test on")

specific_keyword = st.text_input(
    "Specific Keyword (optional)", 
    placeholder="Enter a specific keyword to test on"
)

related_keywords_file = None
if specific_keyword:
  related_keywords_file = st.file_uploader(
      "Upload Related Keywords File (.txt)", 
      type=["txt"],
      help="Upload a .txt file where each line contains a related keyword"
  )

if st.button("Submit"):
  download_nltk_resources()
  keywords = retrieve_keywords(prompt)
  if not prompt.strip():
    st.error("The prompt cannot be empty. Please enter a valid prompt.")
  elif specific_keyword and not related_keywords_file:
    st.error("Please upload a related keywords file to proceed with a specific keyword.")
  elif specific_keyword not in keywords and specific_keyword:
    st.error("The specific keyword is either not present in the prompt or not a valid comparison keyword.")
  else:
      related_keywords = []
      st.toast("Inputs received successfully!", icon="✅")
      
      with st.expander("Input Information"):
        st.write("**Prompt:**", prompt)
        st.write("**Number of Images:**", number_of_images)
        if specific_bias:
            st.write("**Specific Bias:**", specific_bias)
        if specific_keyword:

            st.write("**Specific Keyword:**", specific_keyword)
        if related_keywords_file:
            content = related_keywords_file.read().decode("utf-8")
            related_keywords = content.strip().splitlines()
            related_keywords_display = ", ".join(related_keywords)
            st.write("**Related Keywords:**", related_keywords_display)
      
      if st.secrets.get("openai_key"):
        st.toast("API key inputted", icon="✅")
      else: 
        st.error("Please specify OpenAI API key.")
        
      if st.secrets.get("caption_model_path"):
        st.toast("Caption model path specified", icon="✅")
      else:
        st.error("Please specify caption model path")
      
      progress_text = "Bias detection in progress..."
      percent_complete = 0
      progress_bar = st.progress(percent_complete, text=progress_text)

      gpt_client = OpenAI(
          api_key= st.secrets.get("openai_key")
      )
      
      messages = [ {'role': 'system', 'content': sys_prompt_inst}, {'role': 'user', 'content': ', '.join(keywords)}]
      result = gpt_4_api(messages, gpt_client)
      key_bias = convert_dict(result)
      percent_complete += 20
      progress_text = "Processing prompt..."
      progress_bar.progress(percent_complete, text=progress_text)

      model_path = st.secrets.get("caption_model_path")
      pipe = setup_tti()
      model, tokenizer = setup_image_caption(model_path)
      percent_complete += 20
      progress_text = "Setting up models and tokenizer..."
      progress_bar.progress(percent_complete, text=progress_text)
      
      zip_path = "bias_results.zip"
      save_path = "results_temp"
      
      all_common = all_adj_noun_results(specific_bias, specific_keyword, prompt, related_keywords, key_bias, pipe, number_of_images, save_path, tokenizer, model, gpt_client)
      percent_complete += 30
      progress_text = "Generating adjective-noun pairs..."
      progress_bar.progress(percent_complete, text=progress_text)
      
      summary_stats = compute_statistics(all_common, save_path)
      plot_bias_frequencies(all_common, save_path)
      percent_complete += 20
      progress_text = "Generating statistics and visualisations..."
      progress_bar.progress(percent_complete, text=progress_text)
      
      shutil.make_archive(zip_path.replace('.zip', ''), 'zip', save_path)
      percent_complete += 10
      progress_text = "Result package generated!"
      progress_bar.progress(percent_complete, text=progress_text)
      
      with open(zip_path, "rb") as f:
        st.download_button(
            label="Download Results",
            data=f,
            file_name="bias_results.zip",
            mime="application/zip"
        )

      shutil.rmtree(save_path)
