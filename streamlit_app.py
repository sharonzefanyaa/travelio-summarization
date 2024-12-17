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
import numpy as np
import warnings
import os
from datetime import datetime
import gc

import io
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Suppress warnings
warnings.filterwarnings('ignore')

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_google_drive_service():
    creds = None
    # Get the absolute path to credentials.json
    credentials_path = os.path.join(os.path.dirname(__file__), 'credentials.json')
    
    if not os.path.exists(credentials_path):
        st.error("credentials.json not found. Please follow the setup instructions.")
        return None
        
    # The file token.json stores the user's access and refresh tokens
    token_path = os.path.join(os.path.dirname(__file__), 'token.json')
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    # If credentials are not valid, let user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
                # Save the credentials
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
            except Exception as e:
                st.error(f"Error during authentication: {str(e)}")
                return None

    try:
        service = build('drive', 'v3', credentials=creds)
        return service
    except Exception as e:
        st.error(f"Error building Drive service: {str(e)}")
        return None

def download_file_from_drive(service, file_id):
    try:
        request = service.files().get_media(fileId=file_id)
        file = io.BytesIO()
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        file.seek(0)
        return file.read().decode('utf-8')
    except Exception as e:
        st.error(f"Error downloading file: {str(e)}")
        return None

# Memory management
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Initialize session state
if 'added_reviews' not in st.session_state:
    st.session_state.added_reviews = {
        'positive': [],
        'neutral': [],
        'negative': []
    }
if 'original_data' not in st.session_state:
    st.session_state.original_data = {
        'positive': None,
        'neutral': None,
        'negative': None
    }
if 'current_data' not in st.session_state:
    st.session_state.current_data = {
        'positive': None,
        'neutral': None,
        'negative': None
    }
if 'device' not in st.session_state:
    st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    
# Load and preprocess data
@st.cache_data
def load_original_data():
    try:
        # Initialize Drive service
        service = get_google_drive_service()
        if not service:
            return None

        # File IDs from your Google Drive
        file_ids = {
            'positive': 'positive_text.txt',  # Replace with actual file ID
            'neutral': 'neutral_text.txt',    # Replace with actual file ID
            'negative': 'negative_text.txt'   # Replace with actual file ID
        }

        data = {}
        for sentiment, file_id in file_ids.items():
            content = download_file_from_drive(service, file_id)
            if content is None:
                raise Exception(f"Failed to download {sentiment} file")
            data[sentiment] = content
        return data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Save new reviews
def save_reviews_to_file(reviews, data_path):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = []
        
        for sentiment, texts in reviews.items():
            if texts:  # Only save if there are new reviews
                filepath = os.path.join(data_path, f'new_reviews_{sentiment}_{timestamp}.txt')
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(texts))
                saved_files.append(filepath)
        
        if saved_files:
            st.success(f"Reviews saved to: {', '.join(saved_files)}")
        return True
    except Exception as e:
        st.error(f"Error saving reviews: {str(e)}")
        return False

# Text cleaning function
def clean_text(text):
    text = text.lower()
    
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
        r'\btop\b': 'excellent',
        r'\bops\b': 'operations',
        r'\bpeni\b': '',
        r'\bdisappointingthe\b': 'disappointing the',
        r'\bcs\b': 'customer service',
        r'\bbtw\b': 'by the way',
        r'\b2023everything\b': '2023 everything',
        r'\b2023its\b': '2023 its',
        r'\bbadthe\b': 'bad the',
        r'\bphotothe\b': 'photo the',
        r'\bh-1\b': 'the day before',
        r'\bac\b': 'air conditioner',
        r'\b30-60\b': '30 to 60',
        r'\b8-9\b': '8 to 9',
        r'\bgb/day\b': 'gb per day',
        r'\bnamethe\b': 'name the',
        r'\bluv\b': 'love',
        r'\bc/i\b': 'checkin',
        r'\+': 'and',
        r'\bwfh\b': 'work from home',
        r'\btl\b': 'team leader',
        r'\bspv\b': 'supervisor',
        r'\b2.5hrs\b': '2 and a half hours',
        r'\b&\b': 'and'
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\b\w*?)(\w)\2{2,}(\w*\b)', r'\1\2\3', text)
    
    return text.strip()

# Model definitions
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

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        
        # Move models to device
        bert_model = bert_model.to(st.session_state.device)
        bart_model = bart_model.to(st.session_state.device)
        
        return bert_tokenizer, bert_model, bart_tokenizer, bart_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

# Summarization functions
def get_embeddings(sentences, tokenizer, model, batch_size=2):
    embeddings = []
    try:
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            tokenized_batch = tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512  # BERT's max length
            ).to(st.session_state.device)
            
            with torch.no_grad():
                embedding = model(**tokenized_batch).last_hidden_state
            embeddings.append(embedding.cpu())
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return None
        
    return embeddings

def pad_embeddings(embeddings):
    try:
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
    except Exception as e:
        st.error(f"Error padding embeddings: {str(e)}")
        return None

def generate_summary_in_batches(model, input_embeddings, batch_size):
    try:
        model = model.to(st.session_state.device)
        model.eval()
        summaries = []
        
        with torch.no_grad():
            for i in range(0, input_embeddings.size(0), batch_size):
                batch_embeddings = input_embeddings[i:i + batch_size].to(st.session_state.device)
                summary = model(batch_embeddings)
                summaries.append(summary.cpu())
                
        return torch.cat(summaries, dim=0)
    except Exception as e:
        st.error(f"Error generating batch summaries: {str(e)}")
        return None

def extract_important_sentences(embeddings, original_sentences, top_k=5):
    try:
        sentence_scores = []
        
        for i in range(embeddings.shape[0]):
            max_values_per_sentence = embeddings[i].max(dim=0).values
            mean_value_per_sentence = torch.mean(max_values_per_sentence)
            sentence_scores.append((mean_value_per_sentence, i))
        
        sentence_scores.sort(reverse=True, key=lambda x: x[0])
        top_indices = [index for _, index in sentence_scores[:top_k]]
        important_sentences = [original_sentences[index] for index in top_indices]
        
        return important_sentences
    except Exception as e:
        st.error(f"Error extracting important sentences: {str(e)}")
        return None

def bart_summarize(text, tokenizer, model, max_length=50, min_length=20):
    try:
        inputs = tokenizer(
            text,
            max_length=1024,
            return_tensors="pt",
            truncation=True
        ).to(st.session_state.device)
        
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
    except Exception as e:
        st.error(f"Error generating BART summary: {str(e)}")
        return None

def generate_summary(text, bert_tokenizer, bert_model, bart_tokenizer, bart_model):
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Clean text
        status_text.text("Cleaning text...")
        cleaned_text = clean_text(text)
        progress_bar.progress(10)
        
        # Tokenize into sentences
        status_text.text("Tokenizing text...")
        sentences = sent_tokenize(cleaned_text)
        progress_bar.progress(20)
        
        # Get BERT embeddings
        status_text.text("Generating BERT embeddings...")
        embeddings = get_embeddings(sentences, bert_tokenizer, bert_model)
        if embeddings is None:
            return None
        progress_bar.progress(40)
        
        # Initialize transformer model
        status_text.text("Initializing transformer model...")
        transformer_model = TransformerModel(
            nhead=2,
            num_encoder_layers=4,
            d_model=768,
            dim_feedforward=512,
            dropout=0.1
        ).to(st.session_state.device)
        progress_bar.progress(50)
        
        # Process embeddings
        status_text.text("Processing embeddings...")
        padded_embeddings = pad_embeddings(embeddings)
        if padded_embeddings is None:
            return None
        
        reshaped_embeddings = padded_embeddings.view(
            -1,
            padded_embeddings.shape[2],
            padded_embeddings.shape[3]
        )
        progress_bar.progress(60)
        
        # Generate summary embeddings
        status_text.text("Generating summary embeddings...")
        summary_embeddings = generate_summary_in_batches(
            transformer_model,
            reshaped_embeddings,
            batch_size=2
        )
        if summary_embeddings is None:
            return None
        progress_bar.progress(70)
        
        # Extract important sentences
        status_text.text("Extracting important sentences...")
        important_sentences = extract_important_sentences(
            summary_embeddings,
            sentences,
            top_k=5
        )
        if important_sentences is None:
            return None
        progress_bar.progress(80)
        
        # Combine sentences
        status_text.text("Generating final summary...")
        combined_sentences = " ".join(important_sentences)
        
        # Generate final summary using BART
        final_summary = bart_summarize(
            combined_sentences,
            bart_tokenizer,
            bart_model,
            max_length=50,
            min_length=20
        )
        progress_bar.progress(100)
        status_text.text("Summary generation complete!")
        
        # Clean up
        clear_memory()
        
        return final_summary
    except Exception as e:
        st.error(f"Error in summary generation pipeline: {str(e)}")
        return None

def main():
    st.title("Interactive Review Summarization")
    
    # Initialize NLTK
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        with st.spinner('Downloading required NLTK data...'):
            nltk.download('punkt')
    
    # Initialize or load data
    if st.session_state.original_data['positive'] is None:
        if st.button("Load Data from Google Drive"):
            with st.spinner("Connecting to Google Drive..."):
                data = load_original_data()
                if data:
                    st.session_state.original_data = data
                    st.session_state.current_data = data.copy()
                    st.success("Data loaded successfully from Google Drive!")
    
    if st.session_state.original_data['positive'] is not None:
        # Main interface
        st.sidebar.title("Actions")
        action = st.sidebar.radio(
            "Choose action:",
            ["View Summary", "Add Review"]
        )
        
        if action == "View Summary":
            sentiment_type = st.selectbox(
                "Select sentiment type:",
                ["Positive", "Neutral", "Negative"]
            )
            
            if st.button("Generate Summary"):
                with st.spinner("Generating summary..."):
                    bert_tokenizer, bert_model, bart_tokenizer, bart_model = load_models()
                    if None in (bert_tokenizer, bert_model, bart_tokenizer, bart_model):
                        st.error("Failed to load models. Please try again.")
                        return
                        
                    current_text = st.session_state.current_data[sentiment_type.lower()]
                    summary = generate_summary(
                        current_text,
                        bert_tokenizer,
                        bert_model,
                        bart_tokenizer,
                        bart_model
                    )
                    
                    if summary:
                        st.subheader(f"{sentiment_type} Review Summary")
                        st.write(summary)
                        
                        # Add download button for summary
                        st.download_button(
                            label="Download Summary",
                            data=summary,
                            file_name=f"{sentiment_type.lower()}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
        
        elif action == "Add Review":
            st.subheader("Add New Review")
            review_text = st.text_area("Enter your review:", height=150)
            sentiment = st.selectbox(
                "Select review sentiment:",
                ["Positive", "Neutral", "Negative"]
            )
            
            if st.button("Submit Review"):
                if review_text:
                    with st.spinner("Processing review..."):
                        # Clean and preprocess review
                        cleaned_review = clean_text(review_text)
                        
                        # Add to session state
                        st.session_state.added_reviews[sentiment.lower()].append(cleaned_review)
                        
                        # Update current data
                        current_text = st.session_state.current_data[sentiment.lower()]
                        st.session_state.current_data[sentiment.lower()] = current_text + " " + cleaned_review
                        
                        # Save new reviews to file
                        if save_reviews_to_file(
                            {sentiment.lower(): [cleaned_review]},
                            os.path.dirname(st.session_state.original_data['positive'])
                        ):
                            st.success("Review added and saved successfully!")
                        
                        # Show options
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Add More"):
                                st.experimental_rerun()
                        with col2:
                            if st.button("Exit"):
                                # Reset to original data
                                st.session_state.current_data = st.session_state.original_data.copy()
                                st.session_state.added_reviews = {
                                    'positive': [],
                                    'neutral': [],
                                    'negative': []
                                }
                                st.experimental_rerun()
                else:
                    st.warning("Please enter a review before submitting.")
        
        # Display stats
        st.sidebar.subheader("Statistics")
        st.sidebar.write(f"Added reviews:")
        st.sidebar.write(f"- Positive: {len(st.session_state.added_reviews['positive'])}")
        st.sidebar.write(f"- Neutral: {len(st.session_state.added_reviews['neutral'])}")
        st.sidebar.write(f"- Negative: {len(st.session_state.added_reviews['negative'])}")

if __name__ == "__main__":
    main()
