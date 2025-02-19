import cv2
import numpy as np
from PIL import Image
import io
import logging
import base64

# Constants for frequency and blur analysis thresholds
FREQ_STD_THRESHOLD = 22.7  # Standard deviation threshold for frequency analysis
FREQ_ENTROPY_THRESHOLD = 13.5  # Entropy threshold for frequency analysis
SHARPNESS_THRESHOLD = 150.0  # Threshold for blur analysis (Laplacian variance)

# Configure logging
logger = logging.getLogger(__name__)


def compute_dft_features(gray_image: np.ndarray):
    # Compute 2D DFT and shift so the low frequencies are at the center.
    dft = np.fft.fft2(gray_image)
    dft_shift = np.fft.fftshift(dft)
    mag_spectrum = 20 * np.log(np.abs(dft_shift) + 1)  # Log magnitude for visualization

    features = {}
    features["mean_magnitude"] = np.mean(mag_spectrum)  # Mean of magnitude spectrum
    features["std_magnitude"] = np.std(mag_spectrum)  # Standard deviation of magnitude spectrum

    # Frequency Energy Ratio (low frequency vs high frequency)
    h, w = mag_spectrum.shape
    center_h, center_w = h // 2, w // 2  # Center of the spectrum
    radius = min(center_h, center_w) // 4  # Radius for low-frequency region
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center_w) ** 2 + (Y - center_h) ** 2)
    mask = dist_from_center <= radius  # Mask for low-frequency region
    low_freq_energy = np.mean(mag_spectrum[mask])  # Energy in low-frequency region
    high_freq_energy = np.mean(mag_spectrum[~mask])  # Energy in high-frequency region
    features["freq_energy_ratio"] = low_freq_energy / (high_freq_energy + 1e-6)  # Ratio of energies

    # Angular Symmetry: Compute an angular histogram and compare two halves.
    y_indices, x_indices = np.indices(mag_spectrum.shape)
    angles = np.arctan2(y_indices - center_h, x_indices - center_w)  # Calculate angles
    angles_deg = np.degrees(angles)  # Convert to degrees
    angles_deg[angles_deg < 0] += 360  # Normalize angles to [0, 360]
    bins = np.linspace(0, 360, 361)  # Bins for angular histogram
    angular_hist, _ = np.histogram(
        angles_deg.ravel(), bins=bins, weights=mag_spectrum.ravel())  # Weighted histogram
    angular_hist_norm = angular_hist / (np.sum(angular_hist) + 1e-6)  # Normalize histogram
    first_half = angular_hist_norm[:180]  # First half of the histogram
    second_half = angular_hist_norm[180:]  # Second half of the histogram
    second_half_reversed = second_half[::-1]  # Reverse the second half
    symmetry_corr = np.corrcoef(first_half, second_half_reversed)[0, 1]  # Correlation between halves
    features["angular_symmetry"] = symmetry_corr  # Angular symmetry feature

    # Frequency Entropy: Measures the randomness in the frequency domain.
    flat_mag = mag_spectrum.ravel()  # Flatten the magnitude spectrum
    flat_mag_norm = flat_mag / (np.sum(flat_mag) + 1e-6)  # Normalize the spectrum
    freq_entropy = -np.sum(flat_mag_norm * np.log(flat_mag_norm + 1e-6))  # Calculate entropy
    features["frequency_entropy"] = freq_entropy  # Frequency entropy feature

    # Periodicity analysis via autocorrelation.
    power_spectrum = np.abs(dft_shift) ** 2  # Power spectrum
    auto_corr = np.fft.fftshift(np.abs(np.fft.ifft2(power_spectrum)))  # Autocorrelation
    center_value = auto_corr[center_h, center_w]  # Center value (main peak)
    auto_corr_copy = np.copy(auto_corr)
    # Ignore the main peak to find the second peak.
    auto_corr_copy[center_h - 5:center_h + 6, center_w - 5:center_w + 6] = 0
    second_peak = np.max(auto_corr_copy)  # Second highest peak
    features["periodicity_ratio"] = second_peak / (center_value + 1e-6)  # Ratio of peaks

    logging.info("Computed DFT features successfully.")
    return mag_spectrum, features


def compute_blur_measure(gray_image: np.ndarray):
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()  # Compute Laplacian variance
    logging.info(f"Computed blur measure (Laplacian variance): {laplacian_var}")
    return laplacian_var


def process_image(image_bytes: bytes):
    logging.info("Starting image processing.")
    # Load image using PIL (handles many file formats)
    file_stream = io.BytesIO(image_bytes)
    pil_image = Image.open(file_stream).convert("RGB")  # Convert to RGB format

    # Convert PIL image to OpenCV format and then to grayscale.
    bgr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    logging.info("Image converted to grayscale.")

    # Compute Frequency (DFT) Analysis.
    mag_spectrum, dft_features = compute_dft_features(gray)

    # Compute Blur Analysis (Laplacian Variance).
    blur_measure = compute_blur_measure(gray)

    # --- Frequency Analysis Heuristic ---
    freq_ai = False
    if dft_features["std_magnitude"] < FREQ_STD_THRESHOLD:
        freq_ai = True
    if dft_features["frequency_entropy"] < FREQ_ENTROPY_THRESHOLD:
        freq_ai = True
    dft_result = "AI-generated" if freq_ai else "REAL"
    logging.info(f"Frequency Analysis Decision: {dft_result}")

    # --- Blur Analysis Heuristic ---
    blur_result = "AI-generated" if blur_measure > SHARPNESS_THRESHOLD else "REAL"
    logging.info(f"Blur Analysis Decision: {blur_result}")

    # --- Overall Decision ---
    overall_decision = "REAL"
    if freq_ai:
        overall_decision = "AI-generated"
    logging.info(f"Overall Decision: {overall_decision}")
    
    mag_spectrum_base64 = mag_spectrum_to_base64(mag_spectrum)

    # Prepare the result map
    result_map = {
        "mag_spectrum_base64": mag_spectrum_base64,
        "mean_magnitude": dft_features["mean_magnitude"],
        "std_magnitude": dft_features["std_magnitude"],
        "freq_energy_ratio": dft_features["freq_energy_ratio"],
        "angular_symmetry": dft_features["angular_symmetry"],
        "frequency_entropy": dft_features["frequency_entropy"],
        "periodicity_ratio": dft_features["periodicity_ratio"],
        "laplacian_variance": blur_measure,
        "dft_Analysis_Decision": dft_result,
        "blur_Analysis_Decision": blur_result,
        "overall_decision": overall_decision,
    }

    logging.info("Image processing completed.")
    return result_map

def mag_spectrum_to_base64(mag_spectrum: np.ndarray):
    try:
        # Normalize mag_spectrum for proper display
        mag_spectrum_normalized = np.uint8(np.clip(mag_spectrum, 0, 255))

        # Convert to image (using OpenCV to encode as PNG)
        _, buffer = cv2.imencode('.png', mag_spectrum_normalized)

        # Convert image to base64
        base64_image = base64.b64encode(buffer).decode('utf-8')

        return base64_image

    except Exception as e:
        logging.error(f"Error converting mag_spectrum to base64: {str(e)}")
        return None