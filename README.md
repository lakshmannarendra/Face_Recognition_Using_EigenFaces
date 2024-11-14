
# Face Recognition using Eigenfaces and Principal Component Analysis (PCA)

This project implements a face recognition system using the **eigenfaces** method, leveraging **Principal Component Analysis (PCA)** for dimensionality reduction. The system uses a **k-Nearest Neighbors (k-NN)** classifier within the eigenface space to achieve efficient and accurate face recognition.

> For detailed graphs, explanations, and in-depth analysis, please refer to the [full report](https://github.com/lakshmannarendra/Face_Recognition_Using_EigenFaces/blob/main/report.pdf).

## Dataset

The project uses the **AT&T Face Dataset (ORL dataset)**, which contains 400 grayscale images of 40 individuals, with 10 images per individual. Download the dataset from [here](https://cam-orl.co.uk/facedatabase.html).

## Project Structure

- **Notebook**: A Jupyter Notebook that guides through each stage of data preprocessing, PCA application, and face recognition using k-NN.
- **Image Dataset**: ORL dataset, used to train and evaluate the face recognition model.

> **Note**: For a comprehensive explanation of each stage, including data preprocessing, PCA analysis, and model evaluation, check the [report](https://github.com/lakshmannarendra/Face_Recognition_Using_EigenFaces/blob/main/report.pdf).

## Prerequisites

Ensure the following libraries are installed:

- `numpy`
- `matplotlib`
- `scikit-learn`
- `scipy`

Install them with:
```bash
pip install numpy matplotlib scikit-learn scipy
```

## Project Overview

The project follows these main steps:

1. **Data Loading and Preprocessing**
   - The images are resized, converted to grayscale, and vectorized to prepare them for PCA.

2. **Normalization and Mean Face Calculation**
   - Calculates the mean face of the dataset, centering the data for PCA.

3. **PCA for Eigenface Computation**
   - PCA is applied to compute **eigenfaces**, capturing the main variations among faces in a lower-dimensional space.

4. **Projection into Eigenface Space**
   - Training and test images are projected into the eigenface space, simplifying their representation while retaining essential features for recognition.

5. **Face Recognition Using k-NN Classifier**
   - A k-NN classifier (with `k=1`) classifies images based on the nearest neighbor within the eigenface space.

6. **Real-Time Face Recognition**
   - The system processes and recognizes a new input image by projecting it into the eigenface space and matching it with known faces.

7. **Error Analysis**
   - Misclassifications, error rates, and the impact of varying train-test ratios are analyzed to refine the model's accuracy.

8. **Visualization**
   - Detailed graphs, including the mean face, eigenfaces, 3D projections, and error analysis plots, provide insights into model performance.

> **For detailed visualizations, scree plots, and explained variance graphs, refer to the [report](https://github.com/lakshmannarendra/Face_Recognition_Using_EigenFaces/blob/main/report.pdf).**

## Results

- **Accuracy**: The model achieved approximately **95% accuracy**.
- **Challenges**: Some difficulties were observed in differentiating similar-looking faces and under varying lighting conditions.
- **Future Improvements**: The report suggests exploring **deep learning** techniques (e.g., CNNs) and advanced preprocessing to improve robustness and accuracy.

> For a full breakdown of accuracy metrics, confusion matrix, and classifier performance, consult the [report](https://github.com/lakshmannarendra/Face_Recognition_Using_EigenFaces/blob/main/report.pdf).

## References

- Turk, M., & Pentland, A. (1991). Face recognition using eigenfaces.
- [ORL Database of Faces](https://cam-orl.co.uk/facedatabase.html)

## Usage

1. Clone this repository.
2. Download the ORL dataset and place it in the appropriate directory.
3. Run the Jupyter Notebook to execute each step of the face recognition process.
4. Test real-time face recognition by providing an input image and evaluating the classifier’s predictions.

> **Note**: Detailed instructions, including command-line usage and real-time recognition, are available in the [report](https://github.com/lakshmannarendra/Face_Recognition_Using_EigenFaces/blob/main/report.pdf).



