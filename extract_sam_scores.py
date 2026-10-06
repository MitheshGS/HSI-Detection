import torch
import cv2
import numpy as np
from models.experimental import attempt_load # Standard YOLOv5 model loader
from utils.general import non_max_suppression

# 1. Load your best trained model from the 52-epoch run
weights_path = 'runsx3/train/s2adet_pretrained_52ep/weights/best.pt'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = attempt_load(weights_path, map_location=device)
model.eval()

# 2. Setup the hook to capture scores
captured_scores = []
def extract_attention_scores(module, input, output):
    # Detach the attention map and move to CPU
    captured_scores.append(output.detach().cpu().numpy())

# 3. Locate the SAM Sigmoid layer
# IMPORTANT: You must print the model first if this exact path throws an AttributeError.
# YOLOv5 nests layers inside 'model.model'. You need to find the index of the SSA module.
# print(model) # Uncomment this to view the architecture tree
# 3. Locate the SplitAttention Softmax layer at the final fusion scale (Layer 26)
try:
    # Navigating the nested PyTorch sequential tree based on your architecture dump
    target_layer = model.model[26].S2Attention_all.model[0][0].fn.split_attention.softmax
    
    hook_handle = target_layer.register_forward_hook(extract_attention_scores)
    print("Successfully hooked into the Layer 26 SplitAttention Softmax!")
except AttributeError as e:
    print(f"Error: Could not find the layer. Details: {e}")
    exit()

# 4. Generate two separate inputs for the two-stream architecture
# Shape: (Batch Size, Channels, Height, Width)
dummy_x1 = torch.randn(1, 3, 512, 512).to(device)  # Spatial input
dummy_x2 = torch.randn(1, 3, 512, 512).to(device)  # Spectral input

# 5. Run the forward pass with both streams
with torch.no_grad():
    _ = model(dummy_x1, dummy_x2)

# 6. Analyze the output
attention_map = captured_scores[0]
print(f"\nIntercepted SAM Attention Map Shape: {attention_map.shape}")

flattened_scores = attention_map.flatten()
print("\n--- Channel Attention Scores ---")
for channel_idx, score in enumerate(flattened_scores):
    print(f"Channel {channel_idx}: {score:.4f}")

# 7. Cleanup
hook_handle.remove()
