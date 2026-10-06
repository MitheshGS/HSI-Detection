import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Step 1: Create a synthetic 16-channel Hyperspectral Image (Height, Width, 16)
# (In a real scenario, this is where you load the raw .mat or .h5 file)
height, width = 256, 256
num_bands = 16
print(f"Original Raw HSI Shape: ({height}, {width}, {num_bands})")

# Generate random spectral data to simulate raw 16-band input
raw_hsi_cube = np.random.rand(height, width, num_bands)

# Step 2: Flatten the spatial dimensions to feed into PCA
# Reshape from (256, 256, 16) -> (65536, 16)
flattened_hsi = raw_hsi_cube.reshape(-1, num_bands)

# Step 3: Apply Principal Component Analysis (PCA) to compress 16 bands -> 3 bands
print("Applying PCA Dimensionality Reduction (16 -> 3 channels)...")
pca = PCA(n_components=3)
compressed_hsi = pca.fit_transform(flattened_hsi)

# Step 4: Reshape back to image format (256, 256, 3)
false_color_image = compressed_hsi.reshape(height, width, 3)

# Step 5: Normalize the data to 0-1 for display purposes
false_color_image = (false_color_image - np.min(false_color_image)) / (np.max(false_color_image) - np.min(false_color_image))

print(f"Final Compressed Image Shape for YOLOv5: {false_color_image.shape}")

# Step 6: Visualize the Before & After pipeline
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Show Band 1 of the raw 16-channel cube
axes[0].imshow(raw_hsi_cube[:, :, 0], cmap='gray')
axes[0].set_title('Raw 16-Band Input\n(Showing Band 1/16 in Grayscale)')
axes[0].axis('off')

# Show the final compressed 3-channel image
axes[1].imshow(false_color_image)
axes[1].set_title('Compressed 3-Channel Output\n(False-Color for YOLOv5)')
axes[1].axis('off')

plt.tight_layout()
plt.show()
