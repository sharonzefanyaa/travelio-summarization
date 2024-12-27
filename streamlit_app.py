import streamlit as st
import torch
import re
import pandas as pd
import sqlite3
import requests
from io import StringIO

# Constants
GITHUB_BASE_URL = "https://raw.githubusercontent.com/sharonzefanyaa/travelio-summarization/d8459d9276f0410a4d549a19381dc8e93544195b/"
TEXT_FILES = {
    "Negative": "negative_text.txt",
    "Neutral": "neutral_text.txt",
    "Positive": "positive_text.txt"
}

class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect('reviews.db')
        self.create_tables()
        
    def create_tables(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    sentiment TEXT NOT NULL,
                    cleaned_text TEXT NOT NULL
                )
            ''')
    
    def load_initial_data(self):
        try:
            for sentiment, filename in TEXT_FILES.items():
                url = GITHUB_BASE_URL + filename
                response = requests.get(url)
                if response.status_code == 200:
                    texts = response.text.strip().split('\n')
                    for text in texts:
                        cleaned = clean_text(text)
                        self.add_review(text, sentiment, cleaned)
        except Exception as e:
            st.error(f"Error loading initial data: {str(e)}")
    
    def add_review(self, text, sentiment, cleaned_text):
        with self.conn:
            self.conn.execute(
                'INSERT INTO reviews (text, sentiment, cleaned_text) VALUES (?, ?, ?)',
                (text, sentiment, cleaned_text)
            )
    
    def get_all_reviews(self):
        return pd.read_sql('SELECT * FROM reviews', self.conn)

    def create_temp_dataset(self, new_text, sentiment, cleaned_text):
        all_reviews = self.get_all_reviews()
        new_row = pd.DataFrame({
            'text': [new_text],
            'sentiment': [sentiment],
            'cleaned_text': [cleaned_text]
        })
        return pd.concat([all_reviews, new_row], ignore_index=True)

@st.cache_resource
def initialize_database():
    db = DatabaseManager()
    db.load_initial_data()
    return db

def clean_text(text):
    """Clean and preprocess the input text."""
    # Normalize the text (convert to lowercase)
    text = text.lower()

    # Replace typos with word boundaries to ensure replacement occurs only when surrounded by spaces
    text = re.sub(r'\badv\b', 'advertisement', text)
    text = re.sub(r'\bads\b', 'advertisement', text)
    text = re.sub(r'\bupreally\b', 'up really', text)
    text = re.sub(r'\bntap\b', 'great', text)
    text = re.sub(r'\bn\b', 'and', text)
    text = re.sub(r'\bgreat markotop\b', 'very good', text)
    text = re.sub(r'\btopmarkotop\b', 'very good', text)
    text = re.sub(r'\bunitthanksother\b', 'unit thanks other', text)
    text = re.sub(r'\bapt\b', 'apartment', text)
    text = re.sub(r'\baprt\b', 'apartment', text)
    text = re.sub(r'\bgc\b', 'fast', text)
    text = re.sub(r'\bsatset\b', 'fast', text)
    text = re.sub(r'\bapk\b', 'application', text)
    text = re.sub(r'\bapp\b', 'application', text)
    text = re.sub(r'\bapps\b', 'application', text)
    text = re.sub(r'\bgoib\b', 'hidden', text)
    text = re.sub(r'\bthx u\b', 'thankyou', text)
    text = re.sub(r'\bthx\b', 'thanks', text)
    text = re.sub(r'\brmboy\b', 'roomboy', text)
    text = re.sub(r'\bmantaaaap\b', 'excellent', text)
    text = re.sub(r'\btop\b', 'excellent', text)
    text = re.sub(r'\bops\b', 'operations', text)
    text = re.sub(r'\bpeni\b', '', text)
    text = re.sub(r'\bdisappointingthe\b', 'disappointing the', text)
    text = re.sub(r'\bcs\b', 'customer service', text)
    text = re.sub(r'\bbtw\b', 'by the way', text)
    text = re.sub(r'\b2023everything\b', '2023 everything', text)
    text = re.sub(r'\b2023its\b', '2023 its', text)
    text = re.sub(r'\bbadthe\b', 'bad the', text)
    text = re.sub(r'\bphotothe\b', 'photo the', text)
    text = re.sub(r'\bh-1\b', 'the day before', text)
    text = re.sub(r'\bac\b', 'air conditioner', text)
    text = re.sub(r'\b30-60\b', '30 to 60', text)
    text = re.sub(r'\b8-9\b', '8 to 9', text)
    text = re.sub(r'\bgb/day\b', 'gb per day', text)
    text = re.sub(r'\bnamethe\b', 'name the', text)
    text = re.sub(r'\bluv\b', 'love', text)
    text = re.sub(r'\bc/i\b', 'checkin', text)
    text = re.sub(r'\+', 'and', text)
    text = re.sub(r'\bwfh\b', 'work from home', text)
    text = re.sub(r'\btl\b', 'team leader', text)
    text = re.sub(r'\bspv\b', 'supervisor', text)
    text = re.sub(r'\b2.5hrs\b', '2 and a half hours', text)
    text = re.sub(r'\b&\b', 'and', text)

    # Remove special characters but keep numbers
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove repeating characters
    text = re.sub(r'(\b\w*?)(\w)\2{2,}(\w*\b)', r'\1\2\3', text)

    return text.strip()  # Ensure no leading/trailing whitespace

def get_embeddings(sentences, tokenizer, model, batch_size=1):
    """Get BERT embeddings for the input sentences."""
    if not sentences:
        return []
        
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        try:
            tokenized_batch = tokenizer(
                batch, 
                return_tensors='pt', 
                padding=True, 
                truncation=True,
                max_length=512  # Add explicit max length
            )
            with torch.no_grad():
                embedding = model(**tokenized_batch).last_hidden_state
            embeddings.append(embedding.cpu())
        except Exception as e:
            st.error(f"Error processing batch: {str(e)}")
            continue
    return embeddings

def pad_embeddings(embeddings):
    """Pad embeddings to the same length."""
    if not embeddings:
        return None
        
    try:
        max_length = max(embedding.shape[1] for embedding in embeddings)
        max_batch_size = 1  # Simplified to handle one sentence at a time

        padded_embeddings = []
        for embedding in embeddings:
            if embedding.shape[0] < max_batch_size:
                embedding = embedding.repeat(max_batch_size, 1, 1)
            
            if embedding.shape[1] < max_length:
                padding_tensor = torch.zeros(
                    (embedding.shape[0], max_length - embedding.shape[1], embedding.shape[2])
                )
                padded_embedding = torch.cat((embedding, padding_tensor), dim=1)
            else:
                padded_embedding = embedding

            padded_embeddings.append(padded_embedding)

        return torch.stack(padded_embeddings)
    except Exception as e:
        st.error(f"Error during padding: {str(e)}")
        return None

def generate_summary_in_batches(model, input_embeddings, batch_size=1):
    """Generate summaries in batches."""
    if input_embeddings is None:
        return None
        
    model.eval()
    summaries = []
    try:
        with torch.no_grad():
            for i in range(0, input_embeddings.size(0), batch_size):
                batch_embeddings = input_embeddings[i:i + batch_size]
                summary = model(batch_embeddings)
                summaries.append(summary.cpu())
        return torch.cat(summaries, dim=0)
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

def extract_important_sentences(embeddings, original_sentences, top_k=3):
    """Extract the most important sentences based on embeddings."""
    if embeddings is None or not original_sentences:
        return []
        
    try:
        sentence_scores = []
        for i in range(min(embeddings.shape[0], len(original_sentences))):
            max_values = embeddings[i].max(dim=0).values
            mean_value = torch.mean(max_values)
            sentence_scores.append((mean_value, i))

        sentence_scores.sort(reverse=True)
        top_k = min(5, len(sentence_scores))
        selected_indices = [idx for _, idx in sentence_scores[:top_k]]
        return [original_sentences[idx] for idx in sorted(selected_indices)]
    except Exception as e:
        st.error(f"Error extracting sentences: {str(e)}")
        return []

def bart_summarize(text, tokenizer, model, max_length=50, min_length=10):
    """Generate summary using BART."""
    try:
        inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error in BART summarization: {str(e)}")
        return ""

@st.cache_resource
def load_models():
    try:
        # BERT
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        
        # BART
        bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        
        # Transformer
        transformer_model = TransformerModel(
            nhead=2,
            num_encoder_layers=4,
            d_model=768,
            dim_feedforward=512,
            dropout=0.1
        )
        
        return (bert_tokenizer, bert_model, bart_tokenizer, bart_model, transformer_model)
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def main():
    st.title("Enhanced Review Summarizer")
    
    # Initialize database and load models
    db = initialize_database()
    models = load_models()
    if models is None:
        st.error("Failed to load models")
        return
        
    bert_tokenizer, bert_model, bart_tokenizer, bart_model, transformer_model = models
    
    # Display dataset statistics
    current_data = db.get_all_reviews()
    st.sidebar.write(f"Database size: {len(current_data)} reviews")
    
    # Input section
    review_type = st.sidebar.selectbox(
        "Select Review Type",
        ["Positive", "Neutral", "Negative"]
    )
    
    review_text = st.text_area("Enter new review:", height=150)
    
    if st.button("Process & Summarize"):
        if not review_text:
            st.warning("Please enter a review.")
            return
            
        with st.spinner("Processing..."):
            try:
                # Preprocess new text
                cleaned_text = clean_text(review_text)
                
                # Create temporary dataset
                temp_dataset = db.create_temp_dataset(review_text, review_type, cleaned_text)
                
                # Process for summarization
                sentences = tokenize_sentences(cleaned_text)
                embeddings = get_embeddings(sentences, bert_tokenizer, bert_model)
                padded_embeddings = pad_embeddings(embeddings)
                
                if padded_embeddings is not None:
                    reshaped_embeddings = padded_embeddings.view(
                        -1, padded_embeddings.shape[2], padded_embeddings.shape[3]
                    )
                    
                    summary_embeddings = generate_summary_in_batches(
                        transformer_model, reshaped_embeddings
                    )
                    
                    important_sentences = extract_important_sentences(
                        summary_embeddings, sentences
                    )
                    
                    if important_sentences:
                        final_summary = bart_summarize(
                            " ".join(important_sentences), 
                            bart_tokenizer, 
                            bart_model
                        )
                        
                        # Add to database
                        db.add_review(review_text, review_type, cleaned_text)
                        
                        # Display results
                        st.subheader("Results")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Original Review:")
                            st.write(review_text)
                        with col2:
                            st.write("Summary:")
                            st.write(final_summary)
                        
                        st.success("Review processed and added to database")
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
