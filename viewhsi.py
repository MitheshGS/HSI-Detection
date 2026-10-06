import spectral.io.envi as envi
import matplotlib.pyplot as plt

# 1. Load the ENVI dataset. 
# The library reads the .hdr file to know how to decode the .bil binary file.
img = envi.open('scan0003.hdr', 'scan0003.bil')

print(f"Successfully loaded HSI Data Cube. Shape: {img.shape}")

# 2. Extract a single band to prove it works (e.g., Band 100 out of 168)
# img.shape will be (400, 320, 168)
single_band = img[:, :, 100]

# 3. Display the raw band in grayscale
plt.imshow(single_band, cmap='gray')
plt.title('Raw HSI - Band 100')
plt.axis('off')
plt.show()
