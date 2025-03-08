# Reconstructing RGB Images with Convex Optimization

This repository demonstrates how to reconstruct RGB images using convex optimization techniques, specifically the Total Variation (TV) method. It provides implementations for denoising, deblurring, and inpainting RGB images.

## Overview

This project leverages the power of convex optimization and the Total Variation method to address common image processing tasks:

*   **Denoising:**  Reduces noise in images while preserving important features.
*   **Deblurring:**  Restores images that have been blurred due to motion or defocus.
*   **Inpainting:**  Fills in missing or damaged parts of an image seamlessly.

## Getting Started

### Prerequisites

*   Python 3.6.2 or higher

### Installation

1.  Clone the repository:

    ```
    git clone https://github.com/naot97/reconstruct-image-by-convex.git
    cd reconstruct-image-by-convex
    ```

2.  (Optional) Create a virtual environment:

    ```
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  Install any necessary dependencies (if you have any, list them in a `requirements.txt` file and install with pip):

    ```
    pip install -r requirements.txt # If you have a requirements.txt file
    ```
    *If you don't have a `requirements.txt` file, consider creating one using `pip freeze > requirements.txt` after installing the necessary packages.*

### Usage

Each task (denoising, deblurring, inpainting) has its own script:

*   **Denoising:**

    ```
    python denoise.py
    ```

*   **Deblurring:**

    ```
    python deblur.py
    ```

*   **Inpainting:**

    ```
    python inpaint.py
    ```

*Note: You might need to modify the scripts to specify the input image and output file paths.*

## Testing and Benchmarking

The repository includes functionality for testing and benchmarking the denoising method.

### Denoising Benchmark Data

The `denoising_benchmark_data` folder contains images for benchmarking the denoising performance.

### Benchmarking Function

The `denoising_real()` function in `denoise.py` can be used to compare the Total Variation denoising method against:

*   Mean filter
*   Median filter
*   Thresholding in the Fourier domain

To run the benchmark, you may need to modify the `denoise.py` script to load the benchmark data and call the `denoising_real()` function appropriately.

## Contributing

(Optional) If you want others to contribute:

Contributions are welcome!  Please feel free to submit pull requests with improvements or new features.

## License

(Optional) Add a license file to your repo and specify it here

This project is licensed under the [License Name] License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

(Optional)

*   Mention any libraries, papers, or resources that were helpful in creating this project.
