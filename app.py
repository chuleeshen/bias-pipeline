import streamlit as st
from background import retrieve_keywords, sys_prompt_inst, gpt_4_api, convert_dict, setup_tti, setup_image_caption, bias_results, generate_top_k_phrases
from openai import OpenAI

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

save_path = st.text_input("Results file path", placeholder="Enter a file path")

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
        
      if st.secrets.get():
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
      progress_bar.progress(percent_complete + 20, text=progress_text)

      model_path = st.secrets.get("caption_model_path")
      pipe = setup_tti()
      model, tokenizer = setup_image_caption(model_path)
      progress_bar.progress(percent_complete + 20, text=progress_text)
      
      all_common = bias_results(specific_bias, specific_keyword, prompt, related_keywords, key_bias, pipe, number_of_images, save_path, tokenizer, model, gpt_client)
      progress_bar.progress(percent_complete + 60, text=progress_text)
      progress_bar.empty()
      
      if percent_complete == 100:
        st.toast(f'Results saved in {save_path}', icon="✨")
      
      # figure = generate_top_k_phrases(all_common, 10, save_path)
      # st.pyplot(figure)