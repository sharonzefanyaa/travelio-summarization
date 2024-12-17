# streamlit_app.py
import streamlit as st
import torch
import re
from transformers import (
    BertTokenizer, 
    BertModel,
    BartTokenizer, 
    BartForConditionalGeneration
)
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED_VALUE = 42
torch.manual_seed(SEED_VALUE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED_VALUE)

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_sentences(text):
    """Simple regex-based sentence tokenizer"""
    # Split on period, exclamation mark, or question mark followed by space and uppercase letter
    sentences = re.split(r'[.!?]+\s+(?=[A-Z])', text)
    # Clean up the sentences and remove empty ones
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    if not sentences and text.strip():  # If no sentences found but text exists
        return [text.strip() + '.']  # Return the entire text as one sentence
    return sentences

# Define the TransformerModel class
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

def get_embeddings(sentences, tokenizer, model, batch_size=2):
    """Get BERT embeddings for the input sentences."""
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

def pad_embeddings(embeddings):
    """Pad embeddings to the same length."""
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
    """Generate summaries in batches."""
    model.eval()
    summaries = []
    with torch.no_grad():
        for i in range(0, input_embeddings.size(0), batch_size):
            batch_embeddings = input_embeddings[i:i + batch_size]
            summary = model(batch_embeddings)
            summaries.append(summary.cpu())
    return torch.cat(summaries, dim=0)

def extract_important_sentences(embeddings, original_sentences, top_k=5):
    """Extract the most important sentences based on embeddings."""
    sentence_scores = []

    for i in range(embeddings.shape[0]):
        max_values_per_sentence = embeddings[i].max(dim=0).values
        mean_value_per_sentence = torch.mean(max_values_per_sentence)
        sentence_scores.append((mean_value_per_sentence, i))

    sentence_scores.sort(reverse=True, key=lambda x: x[0])
    top_indices = [index for _, index in sentence_scores[:top_k]]
    important_sentences = [original_sentences[index] for index in top_indices]

    return important_sentences

def bart_summarize(text, tokenizer, model, max_length=50, min_length=20, 
                  length_penalty=2.0, no_repeat_ngram_size=3, num_beams=2):
    """Generate summary using BART."""
    inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
    inputs = inputs.to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=length_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Initialize models
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
    st.title("Review Summarizer")
    
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
                    
                    # Tokenize into sentences
                    sentences = tokenize_sentences(cleaned_text)
                    
                    if not sentences:
                        st.warning("No valid sentences found in the input text. Please check your review.")
                        return
                    
                    st.info(f"Detected {len(sentences)} sentences in the review.")
                    
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
                        top_k=min(3, len(sentences))  # Ensure we don't try to get more sentences than exist
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
                    
                    st.subheader("Key Sentences")
                    for i, sentence in enumerate(important_sentences, 1):
                        st.write(f"{i}. {sentence}")
                        
                except Exception as e:
                    st.error(f"An error occurred during summarization: {str(e)}")
                    st.error("Please try again with a different review text or contact support.")
        else:
            st.warning("Please enter a review to generate a summary.")

if __name__ == "__main__":
    main()
