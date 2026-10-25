# Crop Disease Detection Using Deep Learning

A machine-learning project that accepts an image of a crop leaf and
predicts the disease class learned by the trained model. If the image
belongs to a class labelled as healthy, the system can predict
**Healthy**.

> **Important:** This README assumes a Python project using
> TensorFlow/Keras and a folder-based image dataset. Update script
> names, model filenames, and commands to match your actual project
> files. Predictions are only as reliable as the training data and
> testing; do not treat them as a substitute for advice from an
> agricultural expert.

## Contents

-   [Project overview](#project-overview)
-   [How it works](#how-it-works)
-   [Dataset](#dataset)
-   [Project structure](#project-structure)
-   [Requirements](#requirements)
-   [Installation](#installation)
-   [Prepare the dataset](#prepare-the-dataset)
-   [Train the model](#train-the-model)
-   [Run disease prediction](#run-disease-prediction)
-   [Common issues](#common-issues)
-   [Limitations](#limitations)

## Project overview

**Input:** An image of a crop leaf.\
**Processing:** The image is resized and preprocessed in the same way as
the training images.\
**Output:** The predicted crop/disease class, and optionally a
confidence score.

### Objectives

-   Classify crop leaf images into the disease classes included in the
    training dataset.
-   Include healthy classes where supported by the dataset.
-   Reduce manual effort in preliminary disease identification.
-   Provide a base that can later be integrated into a web or mobile
    application.

## How it works

1.  Collect labelled images of healthy and diseased leaves.
2.  Split the data into training, validation, and test sets.
3.  Preprocess the images and train a CNN or fine-tune a pretrained
    model.
4.  Evaluate the model on images not used for training.
5.  Load the saved model and provide a leaf image.
6.  Display the predicted class and, if implemented, its confidence
    score.

## Dataset

### Recommended dataset: PlantVillage

PlantVillage is a commonly used public dataset for plant-leaf disease
classification. It contains images for multiple crop and disease
categories. The exact number of images and classes depends on the
version downloaded.

-   **Kaggle dataset:**
    https://www.kaggle.com/datasets/emmarex/plantdisease
-   **Alternative project page:** https://plantvillage.psu.edu/

Download the dataset from the source and follow its licence and usage
terms. You may need a free Kaggle account. Dataset files are not
included in this repository by default because they can be large.

### Dataset classes

The model can only predict classes represented in its training labels.
For example, a dataset may include classes such as:

-   Tomato --- healthy
-   Tomato --- early blight
-   Tomato --- late blight
-   Potato --- healthy
-   Potato --- late blight

These are examples only. Check the actual folders in your downloaded
dataset to see which crops and diseases are supported.

### Dataset split

Use separate data for training, validation, and testing. A typical split
is:

-   Training: 70--80%
-   Validation: 10--15%
-   Testing: 10--15%

Avoid placing duplicate or near-duplicate images in different splits.
Keep the test set separate until final evaluation.

## Project structure

One possible structure is:

``` text
crop-disease-detection/
├── dataset/
│   ├── train/
│   │   ├── Tomato___Early_blight/
│   │   ├── Tomato___Late_blight/
│   │   └── Tomato___healthy/
│   ├── val/
│   │   ├── Tomato___Early_blight/
│   │   ├── Tomato___Late_blight/
│   │   └── Tomato___healthy/
│   └── test/
│       ├── Tomato___Early_blight/
│       ├── Tomato___Late_blight/
│       └── Tomato___healthy/
├── models/
│   └── crop_disease_model.keras
├── train.py
├── predict.py
├── app.py                 # Optional user interface
├── requirements.txt
├── .gitignore
└── README.md
```

This is a suggested structure, not a claim about your current files. If
your training code uses a different layout, follow the paths defined in
that code. Do not commit the full dataset or large model files to Git
unless you intend to host them there.

## Requirements

### Software

-   Python 3.10 or 3.11 recommended for broad compatibility with
    TensorFlow packages; use a Python version supported by the
    TensorFlow release you install.
-   pip
-   A code editor such as Visual Studio Code (optional)
-   Windows, Linux, or macOS
-   Internet access for downloading the dataset and installing packages

### Python packages

A typical TensorFlow/Keras implementation may need:

``` text
tensorflow
numpy
pillow
matplotlib
scikit-learn
```

If the project includes a Streamlit interface, also install:

``` text
streamlit
```

The exact packages depend on your code. If a `requirements.txt` file
already exists, use it as the source of truth. Keep package versions
compatible with your Python version and TensorFlow installation.

### Hardware

-   CPU: sufficient for small experiments, though training may be
    slower.
-   RAM: 8 GB or more is helpful for larger datasets.
-   GPU: optional; a compatible GPU setup can speed up training.

## Installation

### 1. Download or clone the project

If the project is on GitHub:

``` bash
git clone <YOUR_REPOSITORY_URL>
cd crop-disease-detection
```

Alternatively, download and extract the project ZIP, then open a
terminal in the project folder.

### 2. Create a virtual environment

**Windows PowerShell:**

``` powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

``` bat
.venv\Scripts\activate.bat
```

**Linux/macOS:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If `requirements.txt` is present:

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Otherwise, install the packages used by the project:

``` bash
pip install tensorflow numpy pillow matplotlib scikit-learn
```

For an optional Streamlit interface:

``` bash
pip install streamlit
```

## Prepare the dataset

1.  Download and extract the dataset.
2.  Arrange the images into class folders, or keep the dataset layout
    expected by your existing loader.
3.  Ensure each class folder contains images of that class only.
4.  Create training, validation, and test sets without leaking duplicate
    images across splits.
5.  Confirm that the folder paths in the training script match your
    actual directories.

Example for a folder-based image loader:

``` text
dataset/
├── train/
│   ├── class_1/
│   └── class_2/
├── val/
│   ├── class_1/
│   └── class_2/
└── test/
    ├── class_1/
    └── class_2/
```

The class folder names are often used as labels. Preserve the
class-to-index mapping used during training so prediction labels remain
correct.

## Train the model

From the project root, activate the virtual environment and run the
training script. If your script is named `train.py`:

``` bash
python train.py
```

The training script should load the dataset, preprocess images, train
the model, evaluate validation performance, and save the trained model.
A common save location is:

``` text
models/crop_disease_model.keras
```

The actual output filename depends on your code. If training uses a
notebook, run the notebook cells in order instead.

### Training recommendations

-   Track training and validation accuracy and loss.
-   Use a separate test set for final evaluation.
-   Review a confusion matrix and per-class precision/recall, not
    accuracy alone.
-   Save the class names in the same order used by the model.
-   Avoid claiming a particular accuracy until it has been measured on
    held-out data.

## Run disease prediction

### Option A: Command-line prediction

If your project includes a `predict.py` script that accepts an image
path, run:

``` bash
python predict.py --image path/to/leaf_image.jpg
```

Replace the example path with your image path. The command-line
arguments depend on your implementation; if `predict.py` does not
support `--image`, use the arguments or input method defined in that
script.

### Option B: Streamlit interface (optional)

If your project has a Streamlit app named `app.py`, run:

``` bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually
`http://localhost:8501`, then upload a leaf image if the interface
provides an upload control.

### Example output

``` text
Predicted class: Tomato___Early_blight
Confidence: 92.4%
```

This is an illustrative example, not a measured result from your model.
Confidence scores are not guarantees that a prediction is correct.

## Common issues

  -----------------------------------------------------------------------
  Issue                               What to check
  ----------------------------------- -----------------------------------
  Dataset folder not found            Confirm the dataset path and folder
                                      names in the training script.

  Model file not found                Train the model first or update the
                                      path to the saved model.

  Incorrect disease label             Ensure the class-index mapping at
                                      inference matches training.

  Low accuracy or one class predicted Check labels, class imbalance,
  repeatedly                          preprocessing, output-layer size,
                                      and train/validation split.

  Out-of-memory error                 Reduce batch size or image
                                      dimensions and close other
                                      applications.

  TensorFlow installation error       Confirm that your Python version is
                                      supported by the TensorFlow version
                                      being installed.

  Poor results on real farm images    Add representative field images and
                                      evaluate under varied lighting,
                                      backgrounds, and leaf conditions.
  -----------------------------------------------------------------------

## Limitations

-   The model can only identify classes it was trained to recognize.
-   Images with poor lighting, complex backgrounds, damaged leaves, or
    unfamiliar crops may produce incorrect predictions.
-   A leaf image alone may not reveal every cause of a plant's symptoms.
-   Treatment advice should be verified with a qualified agricultural
    expert or local extension service.
-   Test the model on real-world images before using it for practical
    decisions.

## Future improvements

-   Add more crops and locally relevant disease classes.
-   Improve performance on field images with varied lighting and
    backgrounds.
-   Add visual explanations or symptom-region highlighting.
-   Provide verified disease-management guidance for supported classes.
-   Deploy the model through a web or mobile application.

## Contributing

1.  Create a branch for your changes.
2.  Keep dataset files and secrets out of version control.
3.  Test changes on validation/test data.
4.  Submit a pull request with a clear description.

## License and dataset attribution

Add a licence for your own code before distributing the project. The
dataset may have separate terms; review and comply with the dataset
provider's licence and attribution requirements.

## Author

**Project:** Crop Disease Detection Using Deep Learning\
**Author(s):** Add your name and team members\
**Institution:** Add your college or institution
