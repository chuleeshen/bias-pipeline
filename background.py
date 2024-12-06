import spacy
from spacy.cli import download

def retrieve_keywords(prompt):
    download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")
    doc = nlp(prompt)
    keywords = []
    for token in doc:
        if token.pos_ in {'NOUN', 'VERB', 'PROPN', 'ADJ'}:
            keywords.append(token.text)
        elif token.ent_type_ in {'NORP', 'PERSON', 'GPE', 'ORG'}:
            keywords.append(token.text)
    return keywords
