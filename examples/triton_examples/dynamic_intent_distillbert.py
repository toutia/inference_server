import tritonclient.http as httpclient
from transformers import DistilBertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Triton server details
TRITON_URL = "localhost:8000"  # Change to your Triton server URL
MODEL_NAME = "distilbert"  # The name of the model deployed on Triton

# Initialize Triton HTTP client
client = httpclient.InferenceServerClient(url=TRITON_URL)

# Function to get embeddings from Triton server
def get_embedding_from_triton(text, max_length=128):
    # Tokenize the text input for DistilBERT
    inputs = tokenizer(text, return_tensors='np', padding='max_length', truncation=True,max_length=max_length)

    # Create inference request inputs for Triton
    input_ids = httpclient.InferInput("input_ids", inputs['input_ids'].shape, "INT32")
    attention_mask = httpclient.InferInput("attention_mask", inputs['attention_mask'].shape, "INT32")

    # Set the input data
    input_ids.set_data_from_numpy(inputs['input_ids'].astype(np.int32))
    attention_mask.set_data_from_numpy(inputs['attention_mask'].astype(np.int32))

    # Make inference request
    results = client.infer(MODEL_NAME, [input_ids, attention_mask])

    # Extract the embedding from the output (assuming it's the CLS token embedding)
    embeddings = results.as_numpy('output')
    return embeddings.squeeze()

# Function to get intent based on cosine similarity
def get_intent(user_input, intent_list):
    user_embedding = get_embedding_from_triton(user_input)
    
    max_similarity = 0
    best_intent = None
    
    for intent in intent_list:
        for pattern in intent['patterns']:
            pattern_embedding = get_embedding_from_triton(pattern)
            similarity = cosine_similarity(user_embedding.reshape(1, -1), pattern_embedding.reshape(1, -1))[0][0]
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_intent = intent['intent']
    
    return best_intent if max_similarity > 0.6 else "unknown_intent"

# Example dynamic intent list
intents = [
    {"intent": "greeting", "patterns": ["hello", "hi", "hey"]},
    {"intent": "book_flight", "patterns": ["book a flight", "flight booking", "buy flight ticket"]},
    {"intent": "check_weather", "patterns": ["weather today", "what’s the weather", "temperature"]},
    {"intent": "register", "patterns": ["fill form", "submit registration form", "submit my profile details","sign up"]},

]

# Example user input
user_input = "Iwant to create an account"

# Match the user input to an intent
matched_intent = get_intent(user_input, intents)
print(f"Matched intent: {matched_intent}")
