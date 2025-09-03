import cv2
import numpy as np
import matplotlib.pyplot as plt


def normalize_channel(channel):
    ch = channel - np.min(channel)
    ch = ch / (np.max(ch) + 1e-8)  # evita divisione per zero
    ch = (ch * 255).astype(np.uint8)
    return ch


# --- Crea immagini RGB spettrali ---
def make_rgb_image(channels):
    # channels: lista di 3 array (R,G,B)
    return np.stack([normalize_channel(ch) for ch in channels], axis=-1)

a=2 #1-3
b=-200 #0-100


location = 'images/0006.jpg'

# Carica immagine a colori (BGR → RGB per coerenza)
img_bgr = cv2.imread(location)
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

print("Image dimensions:", img.shape)  # (h, w, 3)

# Lista per i risultati
magnitude_spectra = []
phases = []
dct_channels = []

# Ciclo sui 3 canali: R, G, B
for c in range(3):
    channel = np.float32(img[:, :, c])

    # DFT 2D complessa
    dft = cv2.dft(channel, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Magnitude e Phase
    mag = 50 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    phase = cv2.phase(dft_shift[:, :, 0], dft_shift[:, :, 1])

    # DCT

    dct = cv2.dct(channel)
    dct_log = np.log(np.abs(dct) + 1)

    magnitude_spectra.append(mag)
    phases.append(phase)
    dct_channels.append(dct_log)

# Magnitude DFT
img_dft_mag = make_rgb_image(magnitude_spectra)  # magnitude_spectra = [R,G,B]

# Phase DFT
img_dft_phase = make_rgb_image(phases)  # phases = [R,G,B]

# Magnitude DCT
img_dct_mag = make_rgb_image(dct_channels)  # dct_channels = [R,G,B]

img_dct_mag_contrasted = cv2.convertScaleAbs(img_dct_mag, alpha=a, beta=b)

# --- Visualizza ---
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img_dft_mag)
plt.axis("off")
plt.title("DFT Magnitude RGB")

plt.subplot(1,3,2)
plt.imshow(img_dft_phase)
plt.axis("off")
plt.title("DFT Phase RGB")

plt.subplot(1,3,3)
plt.imshow(img_dct_mag_contrasted)
plt.axis("off")
plt.title("DCT Magnitude RGB")

plt.tight_layout()
plt.show()
