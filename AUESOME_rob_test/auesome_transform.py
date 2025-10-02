import torch
import torch.nn as nn
from torchvision import models, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from plot_utils import plot_training_curves, plot_validation_curves



def normalize_channel(channel):
    ch = channel - np.min(channel)
    ch = ch / (np.max(ch) + 1e-8)
    ch = (ch * 255).astype(np.uint8)
    return ch

def make_rgb_image(channels):
    return np.stack([normalize_channel(ch) for ch in channels], axis=-1)

class DFTDCTTransform:
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, img: Image.Image):
        # Convert PIL → numpy RGB
        img = img.resize((self.img_size, self.img_size))
        img = np.array(img)

        magnitude_spectra, phases, dct_channels = [], [], []

        for c in range(3):
            channel = np.float32(img[:, :, c])

            # DFT
            dft = cv2.dft(channel, flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)

            mag = 50 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
            phase = cv2.phase(dft_shift[:, :, 0], dft_shift[:, :, 1])

            # DCT
            dct = cv2.dct(channel)
            dct_log = np.log(np.abs(dct) + 1)

            magnitude_spectra.append(mag)
            phases.append(phase)
            dct_channels.append(dct_log)

        # Stack: (9, H, W)
        img_dft_mag = make_rgb_image(magnitude_spectra)
        img_dft_phase = make_rgb_image(phases)
        img_dct_mag = make_rgb_image(dct_channels)

        merged = np.concatenate([img_dft_mag, img_dft_phase, img_dct_mag], axis=-1)  # (H, W, 9)
        merged = merged.astype(np.float32) / 255.0  # normalize 0-1
        merged = np.transpose(merged, (2, 0, 1))  # (C, H, W)

        return torch.tensor(merged, dtype=torch.float)
