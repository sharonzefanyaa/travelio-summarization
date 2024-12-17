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
import re
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import warnings

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Review Summarizer",
    page_icon="📝",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'bert_tokenizer' not in st.session_state:
    st.session_state['bert_tokenizer'] = None
if 'bert_model' not in st.session_state:
    st.session_state['bert_model'] = None
if 'bart_tokenizer' not in st.session_state:
    st.session_state['bart_tokenizer'] = None
if 'bart_model' not in st.session_state:
    st.session_state['bart_model'] = None

def clean_text(text):
    """Clean and normalize text"""
    # Convert to lowercase
    text = text.lower()
    
    # Common replacements
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
        r'\bgoib\b': 'hidden',
        r'\bthx u\b': 'thankyou',
        r'\bthx\b': 'thanks',
        r'\brmboy\b': 'roomboy',
        r'\bmantaaaap\b': 'excellent',
        r'\btop\b': 'excellent'
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    # Remove special characters but keep numbers
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove repeating characters
    text = re.sub(r'(\b\w*?)(\w)\2{2,}(\w*\b)', r'\1\2\3', text)
    
    return text.strip()

class TransformerModel(nn.Module):
    """Transformer model for text processing"""
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

@st.cache_resource
def load_models():
    """Load and cache all required models"""
    # BERT
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    
    # BART
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    
    # Custom Transformer
    model = TransformerModel(
        nhead=2,
        num_encoder_layers=4,
        d_model=768,
        dim_feedforward=512,
        dropout=0.1
    )
    
    return bert_tokenizer, bert_model, bart_tokenizer, bart_model, model

def get_embeddings(sentences, tokenizer, model, batch_size=2):
    """Get BERT embeddings for sentences"""
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        tokenized_batch = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            embedding = model(**tokenized_batch).last_hidden_state
        embeddings.append(embedding.cpu())
    return embeddings

def pad_embeddings(embeddings):
    """Pad embeddings to same length"""
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

def generate_summary_in_batches(model, input_embeddings, batch_size):
    """Generate summary embeddings in batches"""
    model.eval()
    summaries = []
    with torch.no_grad():
        for i in range(0, input_embeddings.size(0), batch_size):
            batch_embeddings = input_embeddings[i:i + batch_size]
            summary = model(batch_embeddings)
            summaries.append(summary.cpu())
    return torch.cat(summaries, dim=0)

def extract_important_sentences(embeddings, original_sentences, top_k=5):
    """Extract most important sentences based on embeddings"""
    sentence_scores = []
    
    for i in range(embeddings.shape[0]):
        max_values_per_sentence = embeddings[i].max(dim=0).values
        mean_value_per_sentence = torch.mean(max_values_per_sentence)
        sentence_scores.append((mean_value_per_sentence, i))
    
    sentence_scores.sort(reverse=True, key=lambda x: x[0])
    top_indices = [index for _, index in sentence_scores[:top_k]]
    important_sentences = [original_sentences[index] for index in top_indices]
    
    return important_sentences

def bart_summarize(text, tokenizer, model, max_length=50, min_length=20):
    """Generate abstractive summary using BART"""
    inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        num_beams=2,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def process_review(review_text, sentiment):
    """Process and summarize a review"""
    # Clean text
    cleaned_text = clean_text(review_text)
    
    # Tokenize into sentences
    sentences = sent_tokenize(cleaned_text)
    
    # Get embeddings
    embeddings = get_embeddings(
        sentences,
        st.session_state.bert_tokenizer,
        st.session_state.bert_model
    )
    
    # Pad embeddings
    padded_embeddings = pad_embeddings(embeddings)
    
    # Reshape embeddings
    reshaped_embeddings = padded_embeddings.view(
        -1,
        padded_embeddings.shape[2],
        padded_embeddings.shape[3]
    )
    
    # Generate summary embeddings
    summary_embeddings = generate_summary_in_batches(
        st.session_state.model,
        reshaped_embeddings,
        batch_size=2
    )
    
    # Extract important sentences
    important_sentences = extract_important_sentences(
        summary_embeddings,
        sentences,
        top_k=5
    )
    
    # Combine sentences
    combined_text = " ".join(important_sentences)
    
    # Generate final summary
    final_summary = bart_summarize(
        combined_text,
        st.session_state.bart_tokenizer,
        st.session_state.bart_model
    )
    
    return final_summary, important_sentences

def main():
    st.title("Review Summarizer")
    st.write("Generate summaries from customer reviews using advanced NLP techniques")
    
    # Load models if not already loaded
    if (st.session_state.bert_tokenizer is None or
        st.session_state.bert_model is None or
        st.session_state.bart_tokenizer is None or
        st.session_state.bart_model is None or
        st.session_state.model is None):
        
        with st.spinner("Loading models... This may take a few moments."):
            (st.session_state.bert_tokenizer,
             st.session_state.bert_model,
             st.session_state.bart_tokenizer,
             st.session_state.bart_model,
             st.session_state.model) = load_models()
    
    # User inputs
    col1, col2 = st.columns([2, 1])
    
    with col1:
        review_text = st.text_area(
            "Enter your review text",
            height=200,
            placeholder="Type or paste your review here..."
        )
    
    with col2:
        sentiment = st.selectbox(
            "Select review sentiment",
            ["Positive", "Neutral", "Negative"]
        )
        
        if st.button("Generate Summary", type="primary"):
            if review_text:
                with st.spinner("Generating summary..."):
                    summary, important_sentences = process_review(review_text, sentiment)
                    
                    st.subheader("Summary")
                    st.write(summary)
                    
                    with st.expander("View Important Sentences"):
                        for i, sentence in enumerate(important_sentences, 1):
                            st.write(f"{i}. {sentence}")
            else:
                st.error("Please enter some review text first.")

if __name__ == "__main__":
    main()
