import spacy
import torch
from diffusers import DiffusionPipeline, AutoencoderKL
import os
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from collections import Counter
import re
import csv
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

captioning_prompt = "Describe the image"

sys_prompt_inst = """
You will be provided keywords where each keyword is separated with commas.
Your task is to provide common biases related to the keywords and say nothing else.

Output only the keyword and its associated biases where each bias is separated with commas, as shown in the format.
If there are no related biases, put a '-' after the keyword.

### Format ###
keyword 1: bias 1, bias 2, bias 3 ...
keyword 2: bias 1, bias 2, bias 3 ...
keyword 3: bias 1, bias 2, bias 3 ...
keyword 4: -
"""

phrase_inst = """
You will be provided some sentences and a topic.
Your task is to provide phrases from the sentences related to the given topic and say nothing else.
If there are no related phrases, put a '-'. 

Output only the phrases where each phrase is separated with slashes, as shown in the format.

### Format ###
phrase 1 / phrase 2 / phrase 3 ...
"""

phrase_user = """
Sentences: {sentences}
Topic: {topic}
"""

def retrieve_keywords(prompt):
    nlp = spacy.load("en_core_web_md")
    doc = nlp(prompt)
    keywords = []
    for token in doc:
        if token.pos_ in {'NOUN', 'VERB', 'PROPN', 'ADJ'}:
            keywords.append(token.text)
        elif token.ent_type_ in {'NORP', 'PERSON', 'GPE', 'ORG'}:
            keywords.append(token.text)
    return keywords

def gpt_4_api(messages, gpt_client):
    completion = gpt_client.chat.completions.create(model="gpt-4o", messages=messages)
    return completion.choices[0].message.content


def clean_string(s):
    return re.sub(r'^[\d\W]+|[\d\W]+$', '', s)

def convert_dict(inp):
    entries = inp.split('\n')
    result_dict = {}
    for entry in entries:
        if not entry.strip():
          continue
        key, values = entry.split(': ', 1)
        if values.strip() == '-':
          result_dict[key] = []
        else:
          values_list = [clean_string(value).strip() for value in values.split(',')]
          result_dict[key] = values_list
    return result_dict

def setup_tti():
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16
    )
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    
    return pipe.to("cuda")

def generate_images(pipe, prompt, number_of_images, save_path):
    image = pipe(prompt=prompt, num_inference_steps=25, num_images_per_prompt = number_of_images)
    
    images = [img.resize((256, 256)) for img in image.images]
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for i, img in enumerate(images):
        img.save(f"{save_path}/{i}.png")

def setup_image_caption(model_path):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_skip_modules=["mm_projector", "vision_model"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=bnb_cfg,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
    )
    
    return model, tokenizer

def process_introductory(input_text):
    words = re.findall(r'\w+|[.,!?;]', input_text.lower())
    i = 0
    while i < len(words) - 2:
        if words[i] == 'the' and words[i+1] == 'image' and words[i+2] == 'shows':
            del words[i:i+3]
        else:
            i += 1

    processed_text = " ".join(words)

    processed_text = re.sub(r'\s([.,!?;])', r'\1', processed_text)

    def capitalize_sentences(text):
        sentences = re.split(r'([.!?]\s*)', text)
        sentences = [s.capitalize() for s in sentences]
        return ''.join(sentences)

    result = capitalize_sentences(processed_text)

    return result

def generate_captions(captioning_prompt, number_of_images, tokenizer, model, img_save_path): 
  caption_dict = {}
  for i in range(number_of_images):
    gen_image_path = f'{img_save_path}/{str(i)}.png'
    image = Image.open(gen_image_path)

    desc_output = tokenizer.decode(model.answer_question(image, captioning_prompt, tokenizer), skip_special_tokens=True)
    
    caption_dict[i] = process_introductory(desc_output)
  
  return caption_dict

def topic_related_phrases(caption_dict, topic, number_of_images, gpt_client):
  phrase_col = []
  for i in range(number_of_images):
    caption = caption_dict.get(i)
    sentences = phrase_user.format(sentences=caption, topic=topic)
    messages = [{'role': 'system', 'content': phrase_inst}, {'role': 'user','content': sentences}]

    result = gpt_4_api(messages, gpt_client)
    phrase_col.append((i, result))
  
  return phrase_col

def topic_combo(input_dict):
  combined = []
  for key in input_dict:
    for value in input_dict[key]:
      cleaned = value.replace('bias', '').strip()
      combined.append(cleaned)
  
  output = list(set(combined))
  output.sort()
  
  return output

def extract_adj_noun_pairs(phrases):
    nlp = spacy.load('en_core_web_sm')
    sentences = re.split(r'\s*/\s*|\s*,\s*', phrases)
    adj_noun_pairs = []
    
    for sentence in sentences:
        doc = nlp(sentence)
        for chunk in doc.noun_chunks:
            adj = []
            noun = ""
            for tok in chunk:
                if tok.pos_ == "NOUN":
                    noun = tok.text
                if tok.pos_ == "ADJ":
                    adj.append(tok.text)
            if noun:
                for i in  adj:
                    adj_noun_pairs.append(f"{i} {noun}")
    
    return adj_noun_pairs

def extract_subject(sentence):
    nlp = spacy.load("en_core_web_md")
    doc = nlp(sentence)
    subjects = []
    
    for token in doc:
        if token.dep_ == "nsubj" or (token.dep_ == "ROOT" and token.pos_ in {"NOUN", "PROPN"}):
            subject_parts = [token.text]
            for child in token.children:
                if child.dep_ in {"amod", "compound"}:
                    subject_parts.insert(0, child.text)
            subjects.append(" ".join(subject_parts))
            
            for sibling in token.conjuncts:
                sibling_parts = [sibling.text]
                for child in sibling.children:
                    if child.dep_ in {"amod", "compound"}:
                        sibling_parts.insert(0, child.text)
                subjects.append(" ".join(sibling_parts))          
    return subjects

def generate_csv_with_matches(caption_list, prompt, save_path):
    prompt_subject = extract_subject(prompt)

    csv_rows = []
    for img_key, caption in caption_list.items():
        img_file = f"{img_key}.png"
        
        first_sentence = caption.split('.')[0] + "."
        caption_subject = extract_subject(first_sentence)

        csv_rows.append({
            "img_file": img_file,
            "prompt_subject": ", ".join(prompt_subject),
            "caption_subject": ", ".join(caption_subject),
        })


    csv_path = os.path.join(save_path, "caption_subject_matches.csv")
    os.makedirs(save_path, exist_ok=True)

    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["img_file", "prompt_subject", "caption_subject"])
        writer.writeheader()
        writer.writerows(csv_rows)

    return csv_path

def write_file(folder, filename, content):
    file_path = os.path.join(folder, filename)
    os.makedirs(folder, exist_ok=True)
    
    with open(file_path, 'w') as file:
        if filename.endswith('.json'):
            json.dump(content, file, indent=4)
        else:
            file.write(str(content))
        
def all_adj_noun_results(specific_bias, specific_keyword, prompt, related_keywords, key_bias, pipe, number_of_images, save_path, tokenizer, model, gpt_client):
  captioning_prompt = "Describe the image with the introductory phrase 'The image shows'. Avoid any mention of the image's stylistic aspects."
  
  if not specific_bias:
    topics = topic_combo(key_bias)
  else:
    topics = [specific_bias.replace('bias', '').strip()]
    
  generate_images(pipe, prompt, number_of_images, f'{save_path}/{prompt.replace(' ', '_')}/images')
    
  has_other_keywords = specific_keyword != '' and len(related_keywords) != 0
  
  if has_other_keywords:
    for i in related_keywords:
      gen_prompt = prompt.replace(specific_keyword, i)
      generate_images(pipe, gen_prompt, number_of_images, f'{save_path}/{gen_prompt.replace(' ', '_')}/images')
  
  index = 0
  current_prompt  = prompt
  prev_keyword = specific_keyword
  results_dict = {}
  
  while index <= len(related_keywords):
    current_prompt_path = f'{save_path}/{current_prompt.replace(' ', '_')}'
    img_save_path = f'{current_prompt_path}/images'
    caption_list = generate_captions(captioning_prompt, number_of_images, tokenizer, model, img_save_path)
    write_file(current_prompt_path, 'caption_list.txt', caption_list)
    generate_csv_with_matches(caption_list, current_prompt, current_prompt_path)
    
    for topic in topics:
      related_phrases = topic_related_phrases(caption_list, topic, number_of_images, gpt_client)
      write_file(current_prompt_path, f'{topic}_related_phrases.txt', related_phrases)
    
      all_pairs = []
      for _, phrases in related_phrases:
        adj_noun_pairs = extract_adj_noun_pairs(phrases)
        all_pairs.extend(adj_noun_pairs)
      
      pair_counts = Counter(all_pairs)
      result = [[pair, freq] for pair, freq in pair_counts.items()]
      
      if topic not in results_dict:
          results_dict[topic] = {}
      
      results_dict[topic][current_prompt] = result
      write_file(current_prompt_path, f'{topic}_adj_noun_pairs.txt', result)
    
    if has_other_keywords and index != len(related_keywords):
      current_prompt = current_prompt.replace(prev_keyword, related_keywords[index])
      prev_keyword = related_keywords[index]
      
    index = index + 1
  
  write_file(save_path, "all-results.txt", results_dict)
  return results_dict

def compute_statistics(data, save_path):
    summary = {}
    
    for bias_type, prompts in data.items():
        prompt_list = []
        adjectives_by_prompt = {}
        nouns_by_prompt = {}
        pairs_by_prompt = {}

        for prompt, pairs in prompts.items():
            if not pairs:
                continue

            prompt_list.append(prompt)
            df = pd.DataFrame(pairs, columns=['adjective_noun', 'frequency'])
            df[['adjective', 'noun']] = df['adjective_noun'].str.split(' ', expand=True, n=1)
            
            adjectives_by_prompt[prompt] = set(df['adjective'])
            nouns_by_prompt[prompt] = set(df['noun'])
            pairs_by_prompt[prompt] = set(df["adjective_noun"])
        
        common_adj_all, common_nouns_all, common_pairs_all = [], [], []
        common_adj_partial, common_nouns_partial, common_pairs_partial = {}, {}, {}
        unique_adjectives, unique_nouns, unique_pairs = {}, {}, {}
        
        if len(prompt_list) > 1:
            if adjectives_by_prompt:
                common_adj_set = set.intersection(*[adjectives_by_prompt[prompt] for prompt in prompt_list])
                common_adj_all = list(common_adj_set)
                
                all_adjectives = set.union(*[adjectives_by_prompt[prompt] for prompt in prompt_list])
                for adj in all_adjectives:
                    sharing_prompts = [prompt for prompt in prompt_list if adj in adjectives_by_prompt[prompt]]
                    if len(sharing_prompts) >= 2:
                        common_adj_partial[adj] = sharing_prompts
                
                for prompt in prompt_list:
                    unique_adjectives[prompt] = list(
                        adjectives_by_prompt[prompt] - common_adj_set - set(common_adj_partial.keys())
                    )
                
            
            if nouns_by_prompt:
                common_noun_set = set.intersection(*[nouns_by_prompt[prompt] for prompt in prompt_list])
                common_nouns_all = list(common_noun_set)
                
                all_nouns = set.union(*[nouns_by_prompt[prompt] for prompt in prompt_list])
                for noun in all_nouns:
                    sharing_prompts = [prompt for prompt in prompt_list if noun in nouns_by_prompt[prompt]]
                    if len(sharing_prompts) >= 2:
                        common_nouns_partial[noun] = sharing_prompts

                for prompt in prompt_list:
                    unique_nouns[prompt] = list(
                        nouns_by_prompt[prompt] - common_noun_set - set(common_nouns_partial.keys())
                    )
            
            if pairs_by_prompt:
                common_pairs_set = set.intersection(*[pairs_by_prompt[prompt] for prompt in prompt_list])
                common_pairs_all = list(common_pairs_set)

                all_pairs = set.union(*[pairs_by_prompt[prompt] for prompt in prompt_list])
                for pair in all_pairs:
                    sharing_prompts = [prompt for prompt in prompt_list if pair in pairs_by_prompt[prompt]]
                    if len(sharing_prompts) >= 2:
                        common_pairs_partial[pair] = sharing_prompts

                for prompt in prompt_list:
                    unique_pairs[prompt] = list(
                        pairs_by_prompt[prompt] - common_pairs_set - set(common_pairs_partial.keys())
                    )
    
        elif len(prompt_list) == 1:
            single_prompt = prompt_list[0]
            unique_adjectives = {single_prompt: list(adjectives_by_prompt[single_prompt])}
            unique_nouns = {single_prompt: list(nouns_by_prompt[single_prompt])}
            unique_pairs = {single_prompt: list(pairs_by_prompt[single_prompt])}

        summary[bias_type] = {
            'common_adjectives': {
                'all': common_adj_all,
                'partial': common_adj_partial
            },
            'common_nouns': {
                'all': common_nouns_all,
                'partial': common_nouns_partial
            },
            "common_pairs": {
                "all": common_pairs_all, 
                "partial": common_pairs_partial
            },
            'unique_adjectives': unique_adjectives,
            'unique_nouns': unique_nouns,
            'unique_pairs': unique_pairs,
        }
            
    write_file(save_path, "summary_stats.json", summary)
    return summary

def plot_bias_frequencies(data, save_path):
    graph_path = f'{save_path}/graphs'
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)
        
    for bias_type, prompts in data.items():
        combined_data = []
        for prompt, pairs in prompts.items():
            for item in pairs:
                combined_data.append({
                    'adjective_noun': item[0],
                    'frequency': item[1],
                    'prompt': prompt
                })
        
        if not combined_data:
            continue
        
        df = pd.DataFrame(combined_data)

        pivot_table = df.pivot_table(
            index='prompt',
            columns='adjective_noun',
            values='frequency',
            aggfunc='sum',
            fill_value=0
        )
        
        max_frequency = int(pivot_table.values.max())
        
        plt.figure(figsize=(30, 25))
        heatmap = sns.heatmap(
            pivot_table,
            annot=True,
            fmt="g",
            annot_kws={"size": 10},
            cmap="rocket",
            cbar_kws={'label': 'Frequency', 'ticks': range(0, max_frequency + 2, 1)},
            vmin=0,
            vmax=max_frequency
        )
        heatmap.set_title(f"Heatmap of Adjective-Noun Frequencies for Bias: {bias_type}", fontsize=16)
        heatmap.set_xlabel("Adjective-Noun Pair", fontsize=12)
        heatmap.set_ylabel("Prompt", fontsize=12)
        heatmap.set_ylim()

        output_path = os.path.join(graph_path, f"{bias_type}_heatmap.png")
        
        plt.savefig(output_path)
        plt.close()


def generate_html_dashboard(results, summary_stats, save_path):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Bias Visualisation Dashboard</title>
        <script src="https://d3js.org/d3.v6.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/d3-cloud/1.2.5/d3.layout.cloud.min.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
        <style>
            body {{
                font-family: Helvetica, sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
            }}

            label {{
                margin-bottom: 10px;
            }}
            
            li {{
                margin-bottom: 10px;
            }}

            #welcome-section {{
                margin: 0 200px 30px;
            }}
            
            .collapsible {{
                background-color: #ccb7e5;
                color: black;
                cursor: pointer;
                padding: 18px;
                width: 100%;
                border: none;
                outline: none;
                font-size: 15px;
                font-weight: bold;
                border-radius: 10px;
            }}
              
            .active, .collapsible:hover {{
                background-color: #ccb7e5;
            }}
            
            .content {{
                padding: 40px;
                display: none;
                overflow: hidden;
                background-color: #f1f1f1;
                text-align: left;
                margin-top: 10px;
                border-radius: 10px;
            }}

            #bias-section {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
                background-color: #afeeee;
                width: 500px;
                padding: 30px 0;
                margin: 20px 0;
                border-radius: 10px;
            }}


            #upset-section {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
                border-radius: 10px;
                border-style: solid;
                border-width: 2px;
                padding: 30px 60px 50px;
            }}
        
            .chart-container {{
                display: flex;
                flex-direction: column; 
                align-items: center; 
                justify-content: center;
                text-align: center;
            }}

            .bar-charts-container {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
                border-radius: 10px;
                border-style: solid;
                border-width: 2px;
                padding: 40px 0;
                margin: 20px 0;
            }}
        
            .word-cloud, .heatmap {{
                width: 100%;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                margin: 20px 0;
                border-style: solid;
                border-width: 2px;
                border-radius: 10px;
            }}
            
            .slider-container {{ 
                margin-top: 20px; 
            }}
            
            svg {{ display: block; margin: auto; }}
            
            .bar-chart {{
                width: auto;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 10px;
            }}
            
            #upset-container {{
                align-items: center;
                justify-content: center;
                .bar {{
                    fill: #50887f;
                }}
                .intersection-size .bar:hover {{
                    fill: #2b524c;
                    transition: fill 250ms ease-in-out;
                }}

                .set-size .bar:hover {{
                    fill: #2b524c;
                    transition: fill 250ms ease-in-out;
                }}
                
                .combination circle {{
                    fill: #e0e0e0;
                }}
                .combination circle.member {{
                    fill: #50887f;
                }}
                .combination .connector {{
                    stroke: #50887f;
                    stroke-width: 3px;
                }}
                
                .axis-title {{
                    font-size: .8rem;
                }}
            }}

            #upset-tooltip {{
                position: absolute;
                opacity: 0;
                background: #fff;
                box-shadow: 2px 2px 3px 0px rgb(92 92 92 / 0.5);
                border: 1px solid #ddd;
                font-size: .8rem;
                font-weight: 600;
                padding: 2px 8px;
            }}

            .info-icon {{
                cursor: pointer;
                margin-left: 8px;
                font-size: 18px;
                color: #007BFF;
            }}
            
            .info-icon:hover {{
                color: #0056b3;
            }}
            
            .info-popup {{
                display: none;
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: white;
                padding: 20px;
                border: 1px solid #ccc;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
                border-radius: 8px;
                width: 400px;
                z-index: 1000;
                text-align: left;
            }}
            
            .info-popup .close-btn {{
                position: absolute;
                top: 5px;
                right: 10px;
                cursor: pointer;
                font-size: 20px;
                color: #333;
            }}
            
            .info-overlay {{
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.5);
                z-index: 999;
            }}
        </style>
    </head>
    <body>
        <h2>Bias Visualisation Dashboard</h2>
        <div id="welcome-section">
            <b>Welcome to the Bias Visualisation Dashboard!</b>
            <br/>
            This interactive dashboard is designed to help you explore and analyse potential biases detected by <b>BiasLens</b>.
            It visualises <b>adjective-noun pairs</b>, which are words that describe elements in images generated by the text-to-image model.
            The frequency of these elements indicates how often they appear across images, allowing you to uncover patterns in descriptions of different groups or identities. 
            Click the <b>'Help Section'</b> below to get started.
            <br/><br/>
            <button type="button" class="collapsible">Help Section<i id="helpArrow" class="fas fa-chevron-down" style="margin-left: 10px;"></i></button>
            <div class="content">
                <i class="fa-solid fa-thumbtack" style="margin-right: 10px; color: blue"></i><b style="color: blue">How to Use the Dashboard</b>
                <br/>
                <h4>Select a Bias</h4>
                <ul>
                    <li>Use the <b>"Choose Bias"</b> dropdown to select a bias category. This list contains biases detected by BiasLens or specified by you. </li>
                    <li>The dashboard will automatically update the visualisations based on your selection.</li>
                </ul>
                <h4>Explore the Visualisations</h4>
                <ul>
                    <li><b>Word Cloud</b>: Highlights the most frequent adjective-noun pairs associated with the selected bias.</li>
                    <li><b>Heatmap</b>: Shows how elements are distributed across different test cases.</li>
                    <li><b>UpSet Plot</b>: Reveals common elements (adjectives, nouns, or pairs) shared between test cases.</li>
                    <li><b>Bar Charts</b>: Displays the top adjective-noun pairs for each test case (adjustable via a slider).</li>
                </ul>
                <i>Tip: Click the info icon<i class="fa-solid fa-circle-info info-icon"></i> next to each chart title for more details.</i>
                <h4>Interact with the Data</h4>
                <ul>
                    <li>Use <b>dropdowns and sliders</b> to customize your view and focus on specific details.</li>
                    <li><b>Hover over charts</b> to see additional information about each element.</li>
                </ul>
    
                <h4>Gain Insights</h4>
                <ul>
                    <li><b>Spot patterns</b> in how different groups are described.</li>
                    <li><b>Compare biases</b> to check for disproportionate representations..</li>
                    <li><b>Analyze word usage trends</b> and detect potential bias influences.</li>
                </ul>
                <br/>
                <i class="fa-solid fa-rocket" style="margin-right: 10px;"></i><b>Start by selecting a bias category</b> from the dropdown menu, and let the visualizations guide your insights! 
    
            </div>
        </div>
        <div id="bias-section">
            <label for="biasSelect"><b>Choose Bias:</b></label>
            <select id="biasSelect"></select>
        </div>
        <div id="charts" hidden="false">
            <div class="chart-container">
                <div class="word-cloud">
                    <h3>Word Cloud
                        <i class="fa-solid fa-circle-info info-icon" onclick="showInfo('wordCloudInfo')"></i>
                    </h3>
                    <label for="cloudSelect"><b>Show Word Cloud For:</b></label>
                    <select id="cloudSelect" style="font-size: medium;"></select>
                    <svg id="wordCloud" width="500" height="500"></svg>
                </div>
                <div class="heatmap">
                    <h3>Overall Heatmap of Adjective-Noun Pairs
                        <i class="fa-solid fa-circle-info info-icon" onclick="showInfo('heatmapInfo')"></i>
                    </h3>
                    <svg id="heatmap"></svg>
                </div>
            </div>
            <div id="upset-section">
                <h3>UpSet Plot
                    <i class="fa-solid fa-circle-info info-icon" onclick="showInfo('upsetInfo')"></i>
                </h3>
                <label for="upsetSelect"><b>Show UpSet Plot For:</b></label>
                <select id="upsetSelect" style="font-size: medium;">
                    <option value="all" selected>Adjective-Noun Pairs</option>
                    <option value="adj">Adjectives</option>
                    <option value="noun">Nouns</option>
                </select>
                <div id="jsi-section">
                    <p id="jsi-value">Select a bias to calculate JSI...</p>
                </div>
                <div id="upset-container"></div>
                <div id="upset-tooltip"></div>
            </div>
            <div class="bar-charts-container">
                <h3>Top Adjective-Noun Pairs per Test Case
                    <i class="fa-solid fa-circle-info info-icon" onclick="showInfo('barChartInfo')"></i>
                </h3>
                <div class="slider-container">
                    <label for="topK">Top K Adjective-Noun Pairs:</label>
                    <input type="range" id="topK" min="1" max="10" value="5">
                    <span id="topKValue">5</span>
                </div>
                <div id="barCharts"></div>
            </div>
        </div>
        <h3 id="no-related-elements" hidden="true">No related elements to bias selected.</h3>
        
        <!-- Background Overlay -->
        <div id="infoOverlay" class="info-overlay" onclick="hideInfo()"></div>

        <!-- Word Cloud Info -->
        <div id="wordCloudInfo" class="info-popup">
            <span class="close-btn" onclick="hideInfo()"><i class="fas fa-times"></i></span>
            <h3><i class="fa-solid fa-thumbtack" style="margin-right: 10px;"></i>Word Cloud</h3>
            <ul>
                <li>Visualizes the <b>most frequently occurring adjective-noun pairs</b> in images associated with the selected bias.</li>
                <li><b>Larger words</b> indicate higher frequency.</li>
                <li><b>Hover over a word</b> to see its exact frequency.</li>
                <li>Use the <b>dropdown menu</b> to filter the word cloud by a specific test case or view data across all test cases.</li>
                <li>If the word cloud is empty, it means no elements were found for the selected bias.</li>
            </ul>
        </div>

        <!-- Heatmap Info -->
        <div id="heatmapInfo" class="info-popup">
            <span class="close-btn" onclick="hideInfo()"><i class="fas fa-times"></i></span>
            <h3><i class="fa-solid fa-thumbtack" style="margin-right: 10px;"></i>Heatmap</h3>
            <ul>
                <li>Displays the <b>frequency of adjective-noun pairs</b> across different test cases.</li>
                <li><b>Darker shades</b> indicate higher frequency; white tiles mean no occurrences.</li>
                <li><b>Hover over a tile</b> to see:
                    <ul>
                        <li>The associated test case.</li>
                        <li>The adjective-noun pair.</li>
                        <li>The frequency of the pair.</li>
                    </ul>
                </li>
            </ul>
        </div>

        <!-- UpSet Plot Info -->
        <div id="upsetInfo" class="info-popup">
            <span class="close-btn" onclick="hideInfo()"><i class="fas fa-times"></i></span>
            <h3><i class="fa-solid fa-thumbtack" style="margin-right: 10px;"></i>UpSet Plot</h3>
            <ul>
                <li>Shows <b>overlapping elements (adjectives, nouns, or pairs)</b> across different test cases.</li>
                <li>Helps identify:
                    <ul>
                        <li><b>Common elements</b> appearing in multiple test cases.</li>
                        <li><b>Unique elements</b> that only appear in a single test case.</li>
                    </ul>
                </li>
                <li>Key components:
                    <ul>
                        <li><b>Intersection size bar chart</b> – Displays the number of shared elements. <b>Hover over a bar to see details.</b></li>
                        <li><b>Set size bar chart</b> – Shows the total number of elements in each test case. <b>Hover over a bar to see details.</b></li>
                        <li><b>Matrix view</b> – Provides a visual representation of overlaps between test cases.</li>
                    </ul>
                </li>
                <li>Use the <b>dropdown menu</b> to choose which aspect to analyze (adjectives, nouns, or adjective-noun pairs),and see how they overlap across test cases.</li>
            </ul>
        </div>

        <!-- Bar Charts Info -->
        <div id="barChartInfo" class="info-popup">
            <span class="close-btn" onclick="hideInfo()"><i class="fas fa-times"></i></span>
            <h3><i class="fa-solid fa-thumbtack" style="margin-right: 10px;"></i>Bar Charts</h3>
            <ul>
                <li>Displays the <b>most frequently occurring adjective-noun pairs</b> in each test case.</li>
                <li>Use the <b>"Top K" slider</b> to adjust the number of displayed pairs.</li>
            </ul>
        </div>
        <script>
            var coll = document.getElementsByClassName("collapsible");
            var arrow = document.getElementById("helpArrow");
            var i;
            
            for (i = 0; i < coll.length; i++) {{
              coll[i].addEventListener("click", function() {{
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.display === "block") {{
                  content.style.display = "none";
                  arrow.classList.remove("fa-chevron-up");
                  arrow.classList.add("fa-chevron-down");
                }} else {{
                  content.style.display = "block";
                  arrow.classList.remove("fa-chevron-down");
                  arrow.classList.add("fa-chevron-up");
                }}
              }});
            }}
            
            function showInfo(id) {{
                document.getElementById(id).style.display = "block";
                document.getElementById("infoOverlay").style.display = "block";
            }}
            
            function hideInfo() {{
                document.querySelectorAll(".info-popup").forEach(popup => popup.style.display = "none");
                document.getElementById("infoOverlay").style.display = "none";
            }}
        </script>
        <script>
            const resultData = {json.dumps(results)};
            const summary = {json.dumps(summary_stats)};
            
            function updateWordCloud(testCase) {{
                const bias = document.getElementById("biasSelect").value;
                const wordFreq = {{}};

                if (testCase == 'all') {{
                    Object.values(resultData[bias]).forEach(testCase => {{
                        testCase.forEach(([pair, freq]) => {{
                            wordFreq[pair] = (wordFreq[pair] || 0) + freq;
                        }});
                    }});
                }} else {{
                    resultData[bias][testCase].forEach(([pair, freq]) => {{
                        wordFreq[pair] = (wordFreq[pair] || 0) + freq;
                    }});
                }}

                const words = Object.entries(wordFreq).map(([word, size]) => {{ return {{ text: word, size: size * (testCase == 'all' ? 10 : 20) }} }});

                d3.select("#wordCloud").selectAll("text").remove();
                d3.layout.cloud()
                    .size([500, 500])
                    .words(words)
                    .padding(5)
                    .rotate(0)
                    .fontSize(d => d.size)
                    .on("end", drawWordCloud)
                    .start();

                function drawWordCloud(words) {{
                    const tooltip = d3.select("body").append("div")
                    .attr("class", "tooltip")
                    .style("position", "absolute")
                    .style("background", "white")
                    .style("border", "1px solid black")
                    .style("padding", "5px")
                    .style("border-radius", "5px")
                    .style("visibility", "hidden");
                    
                    d3.select("#wordCloud")
                        .append("g")
                        .attr("transform", "translate(250,250)")
                        .selectAll("text")
                        .data(words)
                        .enter().append("text")
                        .style("font-size", d => d.size + "px")
                        .attr("text-anchor", "middle")
                        .attr("transform", d => `translate(${{d.x}}, ${{d.y}}) rotate(${{d.rotate}})`)
                        .attr("fill", "#C23B22")
                        .attr("opacity", d => d.size / d3.max(words, d => d.size))
                        .text(d => d.text)
                        .on("mouseover", function (event, d) {{
                            tooltip.style("visibility", "visible")
                                   .text(`Element: ${{d.text}} | Frequency: ${{d.size / (testCase == 'all' ? 10 : 20)}}`)
                                   .style("font-size", "14px");
                            d3.select(this).style("fill", "black");
                        }})
                        .on("mousemove", function (event) {{
                            tooltip.style("top", (event.pageY - 10) + "px")
                                   .style("left", (event.pageX + 10) + "px");
                        }})
                        .on("mouseout", function () {{
                            tooltip.style("visibility", "hidden");
                            d3.select(this).style("fill", "#C23B22");
                        }});
                }}
            }}
            
            function updateBarCharts() {{
                const bias = document.getElementById("biasSelect").value;
                const topK = document.getElementById("topK").value;
                const container = document.getElementById("barCharts");
                container.innerHTML = "";
            
                Object.entries(resultData[bias]).forEach(([testCase, values]) => {{
                    if (values.length === 0) return;
            
                    const sortedPairs = values.sort((a, b) => b[1] - a[1]).slice(0, topK);
                    const numBars = sortedPairs.length;
            
                    const div = document.createElement("div");
                    div.className = "bar-chart";
                    div.innerHTML = `<h4>${{testCase}}</h4><svg></svg>`;
                    container.appendChild(div);
            
                    const svg = d3.select(div).select("svg");
                    const margin = {{ top: 20, right: 30, bottom: 40, left: 150 }};
                    const barHeight = 25; 
                    const width = 400; 
                    const height = numBars * barHeight + margin.top + margin.bottom; 
            
                    svg.attr("width", width + margin.left + margin.right)
                       .attr("height", height)
                       .style("display", "block");
            
                    const g = svg.append("g")
                        .attr("transform", `translate(${{margin.left}},${{margin.top}})`);
            
                    const yScale = d3.scaleBand()
                        .domain(sortedPairs.map(d => d[0]))
                        .range([0, numBars * barHeight])
                        .padding(0.1);
            
                    const xScale = d3.scaleLinear()
                        .domain([0, d3.max(sortedPairs, d => d[1])])
                        .nice()
                        .range([0, width]);
            
                    g.selectAll("rect")
                        .data(sortedPairs)
                        .enter().append("rect")
                        .attr("y", d => yScale(d[0]))
                        .attr("x", 0)
                        .attr("width", d => xScale(d[1]))
                        .attr("height", yScale.bandwidth())
                        .style("fill", "orange");
            
                    g.append("g")
                        .attr("transform", `translate(0,${{numBars * barHeight}})`)
                        .call(d3.axisBottom(xScale));
            
                    g.append("g")
                        .call(d3.axisLeft(yScale));
                }});
            }}

            function updateHeatmap(bias) {{
                if (!resultData[bias]) return;
            
                const svg = d3.select("#heatmap");
                svg.selectAll("*").remove();
            
                const margin = {{ top: 50, right: 30, bottom: 150, left: 250 }};
            
                const prompts = Array.from(new Set(Object.keys(resultData[bias])));
                const pairs = Array.from(new Set([].concat(...Object.values(resultData[bias]).map(arr => arr.map(pair => pair[0])))));
            
                const numRows = prompts.length;
                const numCols = pairs.length;
                const cellSize = 35; 
                const width = numCols * cellSize;
                const height = numRows * cellSize;

                svg.attr("width", width + margin.left + margin.right)
                   .attr("height", height + margin.top + margin.bottom)
                   .attr("viewBox", `0 0 ${{width + margin.left + margin.right}} ${{height + margin.top + margin.bottom}}`)
                   .style("max-width", "100%")
                   .style("display", "block");
            
                const g = svg.append("g").attr("transform", `translate(${{margin.left}},${{margin.top}})`);
            
                const xScale = d3.scaleBand().domain(pairs).range([0, width]).padding(0.05);
                const yScale = d3.scaleBand().domain(prompts).range([0, height]).padding(0.05);
            
                const colorScale = d3.scaleSequential(d3.interpolateOranges)
                    .domain([0, d3.max(Object.values(resultData[bias]).flat(), d => d[1] || 1)]);

                g.selectAll("rect")
                    .data(prompts.flatMap(prompt =>
                        (resultData[bias][prompt] || []).map(([pair, freq]) => ({{ prompt, pair, freq }}))
                    ))
                    .enter()
                    .append("rect")
                    .attr("x", d => xScale(d.pair))
                    .attr("y", d => yScale(d.prompt))
                    .attr("width", xScale.bandwidth())
                    .attr("height", yScale.bandwidth())
                    .style("fill", d => colorScale(d.freq))
                    .style("stroke", "#fff");
            
                g.append("g")
                    .attr("transform", `translate(0,${{height}})`)
                    .call(d3.axisBottom(xScale))
                    .selectAll("text")
                    .attr("transform", "rotate(-45)")
                    .style("text-anchor", "end");
                g.append("g").call(d3.axisLeft(yScale));
            
                const tooltip = d3.select("body").append("div")
                    .attr("class", "tooltip")
                    .style("position", "absolute")
                    .style("background", "white")
                    .style("border", "1px solid black")
                    .style("padding", "5px")
                    .style("border-radius", "5px")
                    .style("visibility", "hidden");
            
                g.selectAll("rect")
                    .on("mouseover", function (event, d) {{
                        tooltip.style("visibility", "visible")
                            .text(`${{d.prompt}} → ${{d.pair}}: ${{d.freq}}`);
                        d3.select(this).style("stroke", "black");
                    }})
                    .on("mousemove", function (event) {{
                        tooltip.style("top", (event.pageY - 10) + "px")
                            .style("left", (event.pageX + 10) + "px");
                    }})
                    .on("mouseout", function () {{
                        tooltip.style("visibility", "hidden");
                        d3.select(this).style("stroke", "#fff");
                    }});
            }}
            
            function calculateJSI(bias) {{
                let commonNodes = summary[bias]?.common_pairs || {{}};
                let uniqueNodes = summary[bias]?.unique_pairs || {{}};
                
                let totalIntersections = 0;
                let multiNationalityIntersections = 0;

                if (Array.isArray(commonNodes.all)) {{
                    totalIntersections += commonNodes.all.length;
                    multiNationalityIntersections += commonNodes.all.length;
                }}

                if (commonNodes.partial) {{
                    for (const [pair, nationalities] of Object.entries(commonNodes.partial)) {{
                        totalIntersections++;
                        if (nationalities.length > 1) {{
                            multiNationalityIntersections++;
                        }}
                    }}
                }}
                
                if (uniqueNodes) {{
                    for (const [_, pairs] of Object.entries(uniqueNodes)) {{
                        totalIntersections += pairs.length;
                    }}
                }}

                let JSI = totalIntersections > 0 ? (multiNationalityIntersections / totalIntersections).toFixed(2) : 0;

                document.getElementById("jsi-value").innerText = `Jaccard Similarity Index (JSI) for ${{bias}}: ${{JSI}}`;
            }}

            function updateUpSetPlot(value) {{
                const bias = document.getElementById("biasSelect").value;
                const upSetContainer = document.getElementById("upset-container");
                upSetContainer.innerHTML = "";

                let commonNodes = {{}};
                let uniqueNodes = {{}};
            
                if (value == 'all') {{
                    commonNodes = summary[bias]?.common_pairs || {{}};
                    uniqueNodes = summary[bias]?.unique_pairs || {{}};
                }} else if (value == 'adj') {{
                    commonNodes = summary[bias]?.common_adjectives || {{}};
                    uniqueNodes = summary[bias]?.unique_adjectives || {{}};
                }} else if (value == 'noun') {{
                    commonNodes = summary[bias]?.common_nouns || {{}};
                    uniqueNodes = summary[bias]?.unique_nouns || {{}};
                }}
                const testCases = Object.keys(resultData[bias]);
            
                function prepareUpSetData(commonNodes, uniqueNodes) {{
                    const combinations = new Map();
                    const sets = new Map();

                    function addCombination(set, name) {{
                        const key = set.sort().join(",");
                        if (!combinations.has(key)) {{
                            combinations.set(key, {{ combinationId: key, setMembership: set, values: [] }});
                        }}
                        combinations.get(key).values.push(name);
                    }}

                    function addSet(set) {{
                        set.forEach((s) => {{
                            if(!sets.has(s)) {{
                                sets.set(s, {{ setId: s, size: 0 }});
                            }}
                            sets.get(s).size++;
                        }});
                    }}

                    
                    if (commonNodes) {{
                        const {{ all, partial }} = commonNodes;
                        if (all && all.length > 0) {{
                            all.forEach(word => {{ addCombination(testCases, word); addSet(testCases); }});
                        }}
            
                        if (partial) {{
                            for (const word in partial) {{
                                addCombination(partial[word], word);
                                addSet(partial[word]);
                            }}
                        }}
                    }}
                    if (uniqueNodes) {{
                        for (const testCase in uniqueNodes) {{
                            uniqueNodes[testCase].forEach(word => {{ addCombination([testCase], word); addSet([testCase]); }});
                        }}
                    }}

                    return {{ combinations: Array.from(combinations.values()).sort((a, b) => b.values.length - a.values.length), 
                        sets: Array.from(sets.values()).sort((a, b) => b.size - a.size) }};
                }}

                function renderUpSetPlot(target, commonNodes, uniqueNodes) {{
                    const data = prepareUpSetData(commonNodes, uniqueNodes);
                    if (data.combinations.length === 0) return;

                    const allSetIds = data.sets.map(d => d.setId);

                    data.combinations.forEach(combination => {{
                        combination.sets = [];
                        allSetIds.forEach(d => {{
                          combination.sets.push({{
                            setId: d, 
                            member: combination.setMembership.includes(d) 
                          }});
                        }});
                    
                        if (combination.setMembership.length > 1) {{
                          combination.connectorIndices = d3.extent(combination.setMembership, d => allSetIds.indexOf(d));
                        }} else {{
                          combination.connectorIndices = [];
                        }}
                    }});

                    const containerWidth = 1400;
                    const containerHeight = 500;
                  
                    const margin = {{ top: 5, right: 0, bottom: 0, left: 5 }};
                    const innerMargin = 12;
                    const tooltipMargin = 10;
                  
                    const width = containerWidth - margin.left - margin.right;
                    const height = containerHeight - margin.top - margin.left;
                  
                    const leftColWidth = 530;
                    const setIdWidth = 370;
                    const setSizeChartWidth = leftColWidth - setIdWidth;
                    const rightColWidth = width - leftColWidth;
                  
                    const topRowHeight = 130;
                    const bottomRowHeight = height - topRowHeight - innerMargin;

                    const intersectionSizeScale = d3.scaleLinear()
                    .range([topRowHeight, 0])
                    .domain([0, d3.max(data.combinations, (d) => d.values.length)]);
              
                    const setSizeScale = d3.scaleLinear()
                        .range([setSizeChartWidth, 0])
                        .domain([0, d3.max(data.sets, (d) => d.size)]);
                
                    const xScale = d3.scaleBand()
                        .range([0, rightColWidth])
                        .domain(data.combinations.map((d) => d.combinationId))
                        .paddingInner(0.2);
                
                    const yScale = d3.scaleBand()
                        .range([0, bottomRowHeight])
                        .domain(allSetIds)
                        .paddingInner(0.2);

                    const svg = d3.select(target)
                        .append('svg')
                            .attr('width', containerWidth)
                            .attr('height', containerHeight)
                        .append('g')
                            .attr('transform', `translate(${{margin.left}}, ${{margin.top}})`);
                
                    const setSizeChart = svg.append('g')
                        .attr('class', 'set-size')
                        .attr('transform', `translate(0, ${{topRowHeight + innerMargin}})`);
                
                    const intersectionSizeChart = svg.append('g')
                        .attr('class', 'intersection-size')
                        .attr('transform', `translate(${{leftColWidth}}, 0)`);
                
                    const combinationMatrix = svg.append('g')
                        .attr('transform', `translate(${{leftColWidth}}, ${{topRowHeight + innerMargin}})`);
                    
                    const combinationGroup = combinationMatrix.selectAll('.combination')
                    .data(data.combinations)
                    .join('g')
                        .attr('class', 'combination')
                        .attr('transform', (d) => `translate(${{xScale(d.combinationId) + xScale.bandwidth()/2}}, 0)`);

                    const circle = combinationGroup.selectAll('circle')
                    .data((combination) => combination.sets)
                    .join('circle')
                        .classed('member', (d) => d.member)
                        .attr('cy', (d) => yScale(d.setId) + yScale.bandwidth()/2)
                        .attr('r', (d) => yScale.bandwidth()/2);
                    
                    const connector = combinationGroup
                    .filter((d) => d.connectorIndices.length > 0)
                    .append('line')
                        .attr('class', 'connector')
                        .attr('y1', (d) => yScale(allSetIds[d.connectorIndices[0]]) + yScale.bandwidth()/2)
                        .attr('y2', (d) => yScale(allSetIds[d.connectorIndices[1]]) + yScale.bandwidth()/2);
                    
                    const setSizeAxis = d3.axisTop(setSizeScale).ticks(5).tickFormat(d3.format('d'));
                    svg.append('g')
                        .attr('transform', (d) => `translate(0, ${{topRowHeight}})`)
                        .call(setSizeAxis);
                    
                    setSizeChart.selectAll('rect')
                        .data(data.sets)
                        .join('rect')
                            .attr('class', 'bar')
                            .attr('width', (d) => setSizeChartWidth - setSizeScale(d.size))
                            .attr('height', yScale.bandwidth())
                            .attr('x', (d) => setSizeScale(d.size))
                            .attr('y', (d) => yScale(d.setId))
                            .on('mouseover', (event, d) => {{
                                d3.select('#upset-tooltip')
                                  .style('opacity', 1)
                                  .html(`Set Size: ${{d.size}}`);
                            }})
                            .on('mousemove', (event) => {{
                                d3.select('#upset-tooltip')
                                  .style('left', (event.pageX + 10) + 'px')
                                  .style('top', (event.pageY + 10) + 'px');
                            }})
                            .on('mouseout', () => {{
                                d3.select('#upset-tooltip').style('opacity', 0);
                            }});
                    
                    setSizeChart.selectAll('.set-name')
                        .data(data.sets)
                        .join('text')
                            .attr('class', 'set-name')
                            .attr('text-anchor', 'middle')
                            .attr('x', leftColWidth - setIdWidth/2)
                            .attr('y', (d) => yScale(d.setId) + yScale.bandwidth()/2)
                            .attr('dy', '0.35em')
                            .text((d) => d.setId);
                    
                    const intersectionSizeAxis = d3.axisLeft(intersectionSizeScale).tickFormat(d3.format('d')).ticks(3);

                    intersectionSizeChart.append('g')
                    .attr('transform', (d) => `translate(${{-innerMargin}},0)`)
                    .call(intersectionSizeAxis);

                    intersectionSizeChart.selectAll('rect')
                    .data(data.combinations)
                    .join('rect')
                        .attr('class', 'bar')
                        .attr('height', (d) => topRowHeight - intersectionSizeScale(d.values.length))
                        .attr('width', xScale.bandwidth())
                        .attr('x', (d) => xScale(d.combinationId))
                        .attr('y', (d) => intersectionSizeScale(d.values.length))
                        .on('mouseover', (event,d) => {{
                            d3.select('#upset-tooltip')
                              .style('opacity', 1)
                              .html(d.values.join('<br/>'));
                          }})
                          .on('mousemove', (event) => {{
                            d3.select('#upset-tooltip')
                              .style('left', (event.pageX + tooltipMargin) + 'px')   
                              .style('top', (event.pageY + tooltipMargin) + 'px')
                          }})
                          .on('mouseout', () => {{
                            d3.select('#upset-tooltip').style('opacity', 0);
                          }});
                    
                    svg.append('text')
                    .attr('class', 'axis-title')
                    .attr('dy', '0.35em')
                    .attr('text-anchor', 'middle')
                    .attr('y', topRowHeight - 30)
                    .attr('x', setSizeChartWidth / 2)
                    .text('Set Size');

                    svg.append('text')
                    .attr('transform', `translate(${{leftColWidth - innerMargin - 30}}, ${{topRowHeight / 2}}) rotate(-90)`)
                    .attr('class', 'axis-title')
                    .attr('dy', '0.35em')
                    .attr('text-anchor', 'middle')
                    .text('Intersection Size');
                }}
            
                let selectElement = document.getElementById("upsetSelect");
                let selectedLabel = selectElement.options[selectElement.selectedIndex].innerHTML;
                const plotDiv = document.createElement("div");
                plotDiv.innerHTML = `<h3>${{selectedLabel}} UpSet Plot</h3>`;
                upSetContainer.appendChild(plotDiv);
                renderUpSetPlot(plotDiv, commonNodes, uniqueNodes);
            }}
            

            function updateVisualisations(bias) {{
                if (!resultData[bias]) {{
                    document.getElementById("no-related-elements").hidden = false;
                    document.getElementById("charts").hidden = true;
                    return;
                }}

                const hasPairs = Object.values(resultData[bias]).some(arr => arr.length > 0);
            
                if (!hasPairs) {{
                    document.getElementById("no-related-elements").hidden = false;
                    document.getElementById("charts").hidden = true;
                    return;
                }}
            
                document.getElementById("no-related-elements").hidden = true;
                document.getElementById("charts").hidden = false;
                
                document.getElementById('upsetSelect').value = 'all';
                document.getElementById('cloudSelect').value = 'all';

                updateWordCloud('all');
                updateHeatmap(bias);
                updateUpSetPlot('all');
                updateBarCharts();
            }}

            const biasSelect = document.getElementById("biasSelect");
            Object.keys(resultData).forEach(bias => {{
                const option = document.createElement("option");
                option.value = bias;
                option.textContent = bias;
                biasSelect.appendChild(option);
            }});

            const bias = document.getElementById("biasSelect").value;

            const cloudSelect = document.getElementById("cloudSelect");
            cloudSelect.innerHTML = '<option value="all" selected>All Prompts</option>';
            Object.keys(resultData[bias]).forEach(testCase => {{
                const option = document.createElement("option");
                option.value = testCase;
                option.textContent = testCase;
                cloudSelect.appendChild(option);
            }})

            updateVisualisations(bias);
            calculateJSI(bias);
            
            document.getElementById("biasSelect").onchange = function() {{
                let selectedBias = this.value;
                updateVisualisations(selectedBias);
                calculateJSI(selectedBias);
            }};
            document.getElementById("topK").oninput = function() {{ document.getElementById("topKValue").innerText = this.value; updateBarCharts(); }};
            document.getElementById("upsetSelect").onchange = function() {{ updateUpSetPlot(this.value); }};
            document.getElementById("cloudSelect").onchange = function() {{ updateWordCloud(this.value); }};
        </script>
    </body>
    </html>
    """
    write_file(save_path, "vis_dashboard.html", html_content)
    

def save_readme(save_path):
    readme_content = """# BiasLens Results

This zip file contains the results from BiasLens, a pipeline for detecting biases in images generated from Text-to-Image (T2I) models.

## Contents
- **all-results.txt**: Stores the Python dictionary of the biases detected and it's related adjective-noun pairs for each test case.
- **summary_stats.json**: A JSON file summarizing the common and unique adjective-noun pairs detected across different test cases.
- **vis_dashboard.html**: An interactive HTML dashboard for exploring the bias visualizations.
- **graphs/**: Contains heatmaps visualisations for each bias, showing the adjective-noun pair frequency.
- There will be a folder for each prompt generated, in each folder:
    - **images/**: The images generated from the T2I model based on the input prompts.
    - **caption_list.txt**: A list of captions generated from the images.
    - **(bias)_related_phrases.txt**: A list of phrases generated from the captions that are related to the specific bias.
    - **(bias)_adj_noun_pairs.txt**: Contains the adjective-noun pairs extracted from the phrases for the specific bias.
    - **caption_subject_matches.csv**: A CSV file mapping subjects detected in the captions to those in the prompts for image generation accuracy checking.

## Examples of Usage
- Explore bias visualizations interactively with **vis_dashboard.html**.
- Identify common adjective-noun pairs using **summary_stats.json** to detect potential biases.
- Use **caption_subject_matches.csv** to verify whether generated images align with prompt expectations.
- Analyze **(bias)_related_phrases.txt** for specific bias patterns in captions.

---
Generated by BiasLens
"""

    write_file(save_path, "README.md", readme_content)