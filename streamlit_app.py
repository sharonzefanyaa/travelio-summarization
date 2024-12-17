import streamlit as st
import torch
import nltk
from nltk.tokenize import sent_tokenize
from transformers import (
    BertTokenizer, 
    BertModel,
    BartTokenizer, 
    BartForConditionalGeneration
)
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import re
import pandas as pd
from rouge_score import rouge_scorer
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set random seed
def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)

# Text cleaning function
def clean_text(text):
    # Normalize the text
    text = text.lower()
    
    # Common replacements dictionary
    replacements = {
        r'\badv\b': 'advertisement',
        r'\bads\b': 'advertisement',
        r'\bupreally\b': 'up really',
        r'\bntap\b': 'great',
        r'\bn\b': 'and',
        r'\bgreat markotop\b': 'very good',
        r'\btopmarkotop\b': 'very good',
        r'\bunitthanksother\b': 'unit thanks other',
        r'\bapt\b': 'apartment',
        r'\baprt\b': 'apartment',
        r'\bgc\b': 'fast',
        r'\bsatset\b': 'fast',
        r'\bapk\b': 'application',
        r'\bapp\b': 'application',
        r'\bapps\b': 'application',
        # Add other replacements from your original code
    }
    
    # Apply replacements
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    # Remove special characters but keep numbers
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove repeating characters
    text = re.sub(r'(\b\w*?)(\w)\2{2,}(\w*\b)', r'\1\2\3', text)
    
    return text.strip()

# Transformer model
class TransformerModel(nn.Module):
    def __init__(self, nhead, num_encoder_layers, d_model, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers,
            num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, src):
        output = self.transformer_encoder(src)
        return self.fc(output)

# Load models
@st.cache_resource
def load_models():
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    return bert_tokenizer, bert_model, bart_tokenizer, bart_model

# Get embeddings
def get_embeddings(sentences, tokenizer, model, batch_size=2):
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        tokenized_batch = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        with torch.no_grad():
            embedding = model(**tokenized_batch).last_hidden_state
        embeddings.append(embedding.cpu())
    return embeddings

# Padding function
def pad_embeddings(embeddings):
    max_length = max(embedding.shape[1] for embedding in embeddings)
    max_batch_size = 2
    padded_embeddings = []
    
    for embedding in embeddings:
        current_batch_size = embedding.shape[0]
        current_length = embedding.shape[1]
        
        if current_batch_size < max_batch_size:
            embedding = embedding.repeat(max_batch_size // current_batch_size, 1, 1)
            
        if current_length < max_length:
            padding_tensor = torch.zeros(
                (embedding.shape[0], max_length - current_length, embedding.shape[2])
            )
            padded_embedding = torch.cat((embedding, padding_tensor), dim=1)
        else:
            padded_embedding = embedding
            
        padded_embeddings.append(padded_embedding)
    
    return torch.stack(padded_embeddings)

# Generate summary
def generate_summary_in_batches(model, input_embeddings, batch_size):
    model.eval()
    summaries = []
    with torch.no_grad():
        for i in range(0, input_embeddings.size(0), batch_size):
            batch_embeddings = input_embeddings[i:i + batch_size]
            summary = model(batch_embeddings)
            summaries.append(summary.cpu())
    return torch.cat(summaries, dim=0)

# Extract important sentences
def extract_important_sentences(embeddings, original_sentences, top_k):
    sentence_scores = []
    
    for i in range(embeddings.shape[0]):
        max_values_per_sentence = embeddings[i].max(dim=0).values
        mean_value_per_sentence = torch.mean(max_values_per_sentence)
        sentence_scores.append((mean_value_per_sentence, i))
    
    sentence_scores.sort(reverse=True, key=lambda x: x[0])
    top_indices = [index for _, index in sentence_scores[:top_k]]
    important_sentences = [original_sentences[index] for index in top_indices]
    
    return important_sentences

# BART summarization
def bart_summarize(text, tokenizer, model, max_length=50, min_length=20):
    inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        num_beams=2,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ROUGE evaluation
def compute_rouge(reference, summary):
    if isinstance(reference, pd.Series):
        reference = reference.astype(str).tolist()
    elif isinstance(reference, str):
        reference = [reference]
    
    if isinstance(summary, pd.Series):
        summary = summary.astype(str).tolist()
    elif isinstance(summary, str):
        summary = [summary]
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference[0], summary[0])
    return scores

# Main Streamlit app
def main():
    st.title("Multi-Sentiment Text Summarization")
    st.write("Generate summaries from text based on sentiment analysis")
    
    # Text input
    text_input = st.text_area(
        "Enter your text to summarize:",
        height=200
    )
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        sentiment = st.selectbox(
            "Select sentiment type:",
            ["Positive", "Neutral", "Negative"]
        )
    with col2:
        top_k = st.slider(
            "Number of key sentences:",
            5, 10, 15
        )
    with col3:
        max_length = st.slider(
            "Maximum summary length:",
            50, 150, 250
        )
    
    if st.button("Generate Summary"):
        if text_input:
            with st.spinner("Processing text..."):
                # Clean text
                cleaned_text = clean_text(text_input)
                sentences = sent_tokenize(cleaned_text)
                
                # Load models
                bert_tokenizer, bert_model, bart_tokenizer, bart_model = load_models()
                
                # Get embeddings
                embeddings = get_embeddings(sentences, bert_tokenizer, bert_model)
                
                # Initialize transformer model
                transformer_model = TransformerModel(
                    nhead=2,
                    num_encoder_layers=4,
                    d_model=768,
                    dim_feedforward=512,
                    dropout=0.1
                )
                
                # Process embeddings
                padded_embeddings = pad_embeddings(embeddings)
                reshaped_embeddings = padded_embeddings.view(
                    -1,
                    padded_embeddings.shape[2],
                    padded_embeddings.shape[3]
                )
                
                summary_embeddings = generate_summary_in_batches(
                    transformer_model,
                    reshaped_embeddings,
                    batch_size=2
                )
                
                # Extract important sentences
                important_sentences = extract_important_sentences(
                    summary_embeddings,
                    sentences,
                    top_k
                )
                
                # Generate final summary
                combined_sentences = " ".join(important_sentences)
                final_summary = bart_summarize(
                    combined_sentences,
                    bart_tokenizer,
                    bart_model,
                    max_length=max_length
                )
                
                # Display results
                st.subheader("Key Sentences:")
                for i, sent in enumerate(important_sentences, 1):
                    st.write(f"{i}. {sent}")
                
                st.subheader("Final Summary:")
                st.write(final_summary)
                
                # Optional: Compute ROUGE scores if reference summary is provided
                reference_summary = st.text_area(
                    "Enter reference summary for ROUGE evaluation (optional):",
                    height=100
                )
                
                if reference_summary:
                    scores = compute_rouge(reference_summary, final_summary)
                    st.subheader("ROUGE Scores:")
                    st.write(f"ROUGE-1: {scores['rouge1'].fmeasure:.3f}")
                    st.write(f"ROUGE-2: {scores['rouge2'].fmeasure:.3f}")
                    st.write(f"ROUGE-L: {scores['rougeL'].fmeasure:.3f}")
        else:
            st.error("Please enter some text to summarize.")

if __name__ == "__main__":
    main()
