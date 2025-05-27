import tritonclient.http as httpclient
import numpy as np

# Initialize the client to communicate with Triton
url = "localhost:8000"  # Modify if Triton is running on a different machine or port
model_name = "distilbert"

client = httpclient.InferenceServerClient(url=url)

# Example input data: This would usually come from tokenizing a sentence with a tokenizer (e.g., Huggingface tokenizer).
input_ids = np.ones((1, 128), dtype=np.int32)  # Placeholder input, 1 sequence with 128 tokens
attention_mask = np.ones((1, 128), dtype=np.int32)  # Attention mask (1 for tokens to attend to)

# Prepare input tensors for Triton
input_ids_tensor = httpclient.InferInput("input_ids", input_ids.shape, "INT32")
attention_mask_tensor = httpclient.InferInput("attention_mask", attention_mask.shape, "INT32")

# Set the input tensor data
input_ids_tensor.set_data_from_numpy(input_ids)
attention_mask_tensor.set_data_from_numpy(attention_mask)

# Make the request to Triton
response = client.infer(model_name=model_name, inputs=[input_ids_tensor, attention_mask_tensor])

# Get the output
output_data = response.as_numpy('output')

# Display the result
print("Model Output Shape:", output_data.shape)
print("Output:", output_data)
