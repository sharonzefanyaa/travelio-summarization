# app.py
import streamlit as st
import torch
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import (
    BertTokenizer, 
    BertModel,
    BartTokenizer, 
    BartForConditionalGeneration
)
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import re
import warnings

# Initialize NLTK at startup
@st.cache_resource
def initialize_nltk():
    try:
        # Try to tokenize a simple sentence to test if punkt is available
        sent_tokenize("This is a test sentence.")
    except LookupError:
        # If punkt is not found, download it
        nltk.download('punkt')
    
    # Verify the download was successful
    try:
        sent_tokenize("This is a test sentence.")
        return True
    except LookupError as e:
        st.error(f"Failed to initialize NLTK tokenizer: {str(e)}")
        return False

# Custom tokenize function with fallback
def safe_tokenize(text):
    try:
        return sent_tokenize(text)
    except LookupError:
        # Fallback tokenization using simple rule-based approach
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

# Initialize NLTK at app startup
nltk_initialized = initialize_nltk()

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED_VALUE = 42
torch.manual_seed(SEED_VALUE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED_VALUE)

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

[Rest of the code remains the same until the main() function]

def main():
    st.title("Review Summarizer")
    
    if not nltk_initialized:
        st.error("Failed to initialize NLTK. Using fallback tokenization.")
    
    # Load models
    models = load_models()
    if models is None:
        st.error("Failed to load required models. Please try again later.")
        return
        
    bert_tokenizer, bert_model, bart_tokenizer, bart_model, transformer_model = models
    
    # Sidebar
    st.sidebar.header("Settings")
    review_type = st.sidebar.selectbox(
        "Select Review Type",
        ["Positive", "Neutral", "Negative"]
    )
    
    # Main content
    st.write(f"Selected Review Type: **{review_type}**")
    
    # Text input
    review_text = st.text_area("Enter your review:", height=150)
    
    if st.button("Generate Summary"):
        if review_text:
            with st.spinner("Generating summary..."):
                try:
                    # Clean text
                    cleaned_text = clean_text(review_text)
                    
                    # Use safe tokenize function
                    sentences = safe_tokenize(cleaned_text)
                    
                    if not sentences:
                        st.warning("No valid sentences found in the input text. Please check your review.")
                        return
                    
                    # Get embeddings
                    embeddings = get_embeddings(sentences, bert_tokenizer, bert_model)
                    
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
                        transformer_model, 
                        reshaped_embeddings, 
                        batch_size=2
                    )
                    
                    # Extract important sentences
                    important_sentences = extract_important_sentences(
                        summary_embeddings, 
                        sentences, 
                        top_k=3
                    )
                    
                    # Generate final summary
                    combined_sentences = " ".join(important_sentences)
                    final_summary = bart_summarize(
                        combined_sentences, 
                        bart_tokenizer, 
                        bart_model
                    )
                    
                    # Display results
                    st.subheader("Summary")
                    st.write(final_summary)
                    
                    st.subheader("Important Sentences")
                    for i, sentence in enumerate(important_sentences, 1):
                        st.write(f"{i}. {sentence}")
                        
                except Exception as e:
                    st.error(f"An error occurred during summarization: {str(e)}")
                    st.error("Please try again with a different review text or contact support.")
        else:
            st.warning("Please enter a review to generate a summary.")

if __name__ == "__main__":
    main()
