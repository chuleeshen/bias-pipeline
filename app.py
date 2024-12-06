import streamlit as st
from background import retrieve_keywords

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
  keywords = retrieve_keywords(prompt)
  if not prompt.strip():
    st.error("The prompt cannot be empty. Please enter a valid prompt.")
  elif specific_keyword and not related_keywords_file:
    st.error("Please upload a related keywords file to proceed with a specific keyword.")
  elif specific_keyword not in keywords:
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
            related_keywords_display = ", ".join(content.strip().splitlines())
            st.write("**Related Keywords:**", related_keywords_display)
            related_keywords = content.strip().splitlines()
      
      if st.secrets.get("openai_key"):
        st.toast("API key inputted", icon="✅")
      
      print(related_keywords)
      
      