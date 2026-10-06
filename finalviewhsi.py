import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. Load the real HSI data from the binary file
print("Loading raw binary data...")
img = envi.open('scan0003.hdr', 'scan0003.bil').load()

# 2. Extract exactly 16 bands to match the evaluator's requirement
# Slicing 16 contiguous bands starting from index 100
raw_16_bands = img[:, :, 100:116]
height, width, num_bands = raw_16_bands.shape
print(f"Extracted 16-band cube shape: ({height}, {width}, {num_bands})")

# 3. Flatten the spatial dimensions to feed into PCA
flattened_hsi = raw_16_bands.reshape(-1, num_bands)

# 4. Apply Principal Component Analysis (PCA) to compress 16 bands into 3
print("Applying PCA Dimensionality Reduction (16 -> 3 channels)...")
pca = PCA(n_components=3)
compressed_hsi = pca.fit_transform(flattened_hsi)

# 5. Reshape back to standard image dimensions and normalize for display
false_color_image = compressed_hsi.reshape(height, width, 3)
false_color_image = (false_color_image - np.min(false_color_image)) / (np.max(false_color_image) - np.min(false_color_image))

print(f"Final Compressed Image Shape for YOLOv5: {false_color_image.shape}")

# 6. Visualize the Before & After pipeline
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(raw_16_bands[:, :, 0], cmap='gray')
axes[0].set_title('Raw Input\n(Showing 1 of 16 Bands in Grayscale)')
axes[0].axis('off')

axes[1].imshow(false_color_image)
axes[1].set_title('Compressed 3-Channel Output\n(False-Color for YOLOv5)')
axes[1].axis('off')

plt.tight_layout()
plt.show()
