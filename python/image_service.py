import cv2
import numpy as np
from PIL import Image
import io
import logging
import base64

# Configure logging
logger = logging.getLogger(__name__)


def compute_dft_features(gray_image: np.ndarray):
    """
    Compute the 2D DFT magnitude spectrum and extract advanced statistical features.
    Returns the magnitude spectrum and a dictionary of features.
    """
    # Compute the 2D DFT and shift to center the zero frequency.
    dft = np.fft.fft2(gray_image)
    dft_shift = np.fft.fftshift(dft)
    # Log scale for better visualization
    mag_spectrum = 20 * np.log(np.abs(dft_shift) + 1)

    features = {}
    features["mean_magnitude"] = np.mean(mag_spectrum)
    features["std_magnitude"] = np.std(mag_spectrum)

    # Compute low vs. high frequency energy ratio.
    h, w = mag_spectrum.shape
    center_h, center_w = h // 2, w // 2
    radius = min(center_h, center_w) // 4
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center_w)**2 + (Y - center_h)**2)
    mask = dist_from_center <= radius
    low_freq_energy = np.mean(mag_spectrum[mask])
    high_freq_energy = np.mean(mag_spectrum[~mask])
    features["freq_energy_ratio"] = low_freq_energy / (high_freq_energy + 1e-6)

    # Angular symmetry: compare the angular distribution in two halves.
    y_indices, x_indices = np.indices(mag_spectrum.shape)
    angles = np.arctan2(y_indices - center_h, x_indices - center_w)
    angles_deg = np.degrees(angles)
    angles_deg[angles_deg < 0] += 360
    bins = np.linspace(0, 360, 361)
    angular_hist, _ = np.histogram(
        angles_deg.ravel(), bins=bins, weights=mag_spectrum.ravel())
    angular_hist_norm = angular_hist / (np.sum(angular_hist) + 1e-6)
    first_half = angular_hist_norm[:180]
    second_half = angular_hist_norm[180:]
    second_half_reversed = second_half[::-1]
    symmetry_corr = np.corrcoef(first_half, second_half_reversed)[0, 1]
    features["angular_symmetry"] = symmetry_corr

    # Frequency entropy.
    flat_mag = mag_spectrum.ravel()
    flat_mag_norm = flat_mag / (np.sum(flat_mag) + 1e-6)
    freq_entropy = -np.sum(flat_mag_norm * np.log(flat_mag_norm + 1e-6))
    features["frequency_entropy"] = freq_entropy

    # Periodicity analysis via autocorrelation.
    power_spectrum = np.abs(dft_shift)**2
    auto_corr = np.fft.fftshift(np.abs(np.fft.ifft2(power_spectrum)))
    center_value = auto_corr[center_h, center_w]
    auto_corr_copy = np.copy(auto_corr)
    # Ignore the main peak.
    auto_corr_copy[center_h-5:center_h+6, center_w-5:center_w+6] = 0
    second_peak = np.max(auto_corr_copy)
    features["periodicity_ratio"] = second_peak / (center_value + 1e-6)

    return mag_spectrum, features


def compute_blur_measure(gray_image: np.ndarray):
    """
    Compute the blur metric using the variance of the Laplacian.
    A higher variance indicates a sharper image.
    """
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()


def process_image(image_bytes: bytes):
    """
    Main entry point for processing an image.
    Takes image bytes as input, performs DFT and blur analysis,
    and returns a dictionary with the results.
    """
    logging.info("Starting image processing...")

    # Convert bytes to a PIL image and then to OpenCV format.
    file_stream = io.BytesIO(image_bytes)
    pil_image = Image.open(file_stream).convert("RGB")
    bgr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    logging.info("Image converted to grayscale for analysis.")

    # --- DFT Analysis ---
    mag_spectrum, dft_features = compute_dft_features(gray)
    logging.info("DFT features computed.")

    # DFT Decision: If std magnitude > 23.7, classify as REAL.
    if dft_features["std_magnitude"] > 23.7:
        dft_result = "REAL"
    else:
        dft_result = "AI-generated"
    logging.info(f"DFT Analysis Decision: {dft_result}")

    # --- Blur Analysis ---
    blur_measure = compute_blur_measure(gray)
    logging.info(f"Blur measure (Laplacian variance): {blur_measure}")

    # Blur Decision: If blur_measure < 150, classify as REAL.
    if blur_measure < 150.0:
        blur_result = "REAL"
    else:
        blur_result = "AI-generated"
    logging.info(f"Blur Analysis Decision: {blur_result}")

    # --- Combined Decision ---
    # If either DFT or Blur analysis returns REAL, classify as REAL.
    if dft_result == "REAL" or blur_result == "REAL":
        overall_decision = "REAL"
    else:
        overall_decision = "AI-generated"
    logging.info(f"Overall Decision: {overall_decision}")

    mag_spectrum_base64 = mag_spectrum_to_base64(mag_spectrum)

    # Prepare the result map with all required keys.
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
        "overall_decision": overall_decision
    }

    logging.info("Image processing completed.")
    return result_map


# Example usage (for testing):
if __name__ == "__main__":
    # Load an image as bytes for testing.
    with open("example_image.jpg", "rb") as f:
        image_bytes = f.read()

    # Process the image and print the results.
    results = process_image(image_bytes)
    print(results)


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
