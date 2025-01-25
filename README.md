# Crop and Fertilizer Recommendation System 🌾🌱

![Project Banner](path_to_your_banner_image) <!-- Optional: Add a banner image -->

## Table of Contents

- [📖 Introduction](#-introduction)
- [🚀 Features](#-features)
- [🛠️ Technologies Used](#️-technologies-used)
- [📊 Data](#-data)
- [🤖 Models](#-models)
- [📈 Results](#-results)
- [⚙️ Installation](#️-installation)
- [💻 Usage](#-usage)
- [📁 Project Structure](#-project-structure)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [📧 Contact](#-contact)

---

## 📖 Introduction

Agriculture is a cornerstone of human civilization, and optimizing crop yield while ensuring sustainable farming practices is crucial for food security and environmental health. This project presents a **Crop and Fertilizer Recommendation System** leveraging machine learning to assist farmers in selecting the most suitable crops and corresponding fertilizers based on soil and environmental parameters.

### 🔍 Problem Statement

Farmers often face challenges in selecting the right crop and fertilizer, leading to suboptimal yields and resource wastage. This system aims to:

- **Recommend the optimal crop** to cultivate based on soil nutrients, pH, moisture, and climatic conditions.
- **Suggest appropriate fertilizers** to enhance soil fertility and crop yield, tailored to the recommended crop and current soil deficiencies.

### 🎯 Objectives

- Develop accurate machine learning models for crop and fertilizer recommendations.
- Provide an intuitive interface for farmers to input their soil and environmental data.
- Facilitate informed decision-making to improve agricultural productivity and sustainability.

---

## 🚀 Features

- **Crop Recommendation**: Suggests the best-suited crop for cultivation based on input parameters.
- **Fertilizer Recommendation**: Recommends appropriate fertilizers to address soil deficiencies and support the selected crop.
- **User-Friendly Interface**: Interactive interface for easy data input and result interpretation.
- **Model Deployment**: Ready for deployment as a web application or API for seamless integration.

---

## 🛠️ Technologies Used

- **Programming Language**: Python
- **Libraries & Frameworks**:
  - Data Manipulation: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`
  - Model Serialization: `joblib`
  - Web Interface (Optional): `Streamlit` / `Flask` / `Django`
- **Development Environment**: VS Code with Jupyter Notebook extension
- **Version Control**: Git & GitHub

---

## 📊 Data

### 🔗 Datasets

1. **Crop Recommendation Dataset** (`crop_recommendation.csv`)
   - **Features**:
     - `temparature`: Temperature in °C
     - `humidity`: Humidity in %
     - `moisture`: Soil moisture level
     - `soil type`: Type of soil (e.g., Sandy, Loamy, Clayey)
     - `nitrogen`: Nitrogen level in soil
     - `potassium`: Potassium level in soil
     - `phosphorous`: Phosphorous level in soil
   - **Target**:
     - `label`: Crop type to be recommended

2. **Fertilizer Prediction Dataset** (`fertilizer_prediction.csv`)
   - **Features**:
     - `temparature`: Temperature in °C
     - `humidity`: Humidity in %
     - `moisture`: Soil moisture level
     - `soil type`: Type of soil
     - `crop type`: Type of crop
     - `nitrogen`: Nitrogen level in soil
     - `potassium`: Potassium level in soil
     - `phosphorous`: Phosphorous level in soil
   - **Target**:
     - `fertilizer name`: Recommended fertilizer

### 📑 Data Source

*Provide details about where the datasets originated. If they are publicly available, include links. If they are proprietary or generated, briefly describe their creation.*

---

## 🤖 Models

### 1. **Fertilizer Recommendation Model**

- **Algorithm**: Random Forest Classifier
- **Performance**:
  - **Accuracy**: 100%
  - **Classification Report**:
    ```
                  precision    recall  f1-score   support

        10-26-26       1.00      1.00      1.00         1
        14-35-14       1.00      1.00      1.00         3
        17-17-17       1.00      1.00      1.00         1
           20-20       1.00      1.00      1.00         3
           28-28       1.00      1.00      1.00         3
             DAP       1.00      1.00      1.00         4
            Urea       1.00      1.00      1.00         5

        accuracy                           1.00        20
       macro avg       1.00      1.00      1.00        20
    weighted avg       1.00      1.00      1.00        20
    ```
- **Notes**:
  - Achieved perfect accuracy on the test set. Consider evaluating with a larger dataset or using cross-validation to ensure generalizability.

### 2. **Crop Recommendation Model**

*To be developed. Follow a similar approach as the Fertilizer Recommendation Model.*

---

## 📈 Results

### Fertilizer Recommendation

- **Accuracy**: 100%
- **Confusion Matrix**:

  ![Confusion Matrix](path_to_confusion_matrix_image)

*Include visualizations and detailed metrics once both models are fully developed and evaluated.*

---

## ⚙️ Installation

### Prerequisites

- **Python 3.6+**
- **pip** package manager

### Clone the Repository

```bash
git clone https://github.com/yourusername/crop-fertilizer-recommendation.git
cd crop-fertilizer-recommendation
```

### Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

*Ensure your `requirements.txt` includes all necessary libraries. Example:*

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
streamlit  # If using Streamlit for the interface
```

---

## 💻 Usage

### 1. **Training the Models**

*Run the Jupyter Notebook or Python scripts to train and save the models.*

```bash
jupyter notebook
```

*Open the notebook and execute the cells to train the Crop and Fertilizer Recommendation models.*

### 2. **Making Predictions**

*Use the provided scripts or integrate the models into a web application for real-time recommendations.*

#### Example: Using the Recommendation Function

```python
from recommendation_system import recommend_crop_and_fertilizer

# Example input values
temparature = 25.0      # in degrees Celsius
humidity = 80.0         # in percentage
moisture = 200          # soil moisture level
soil_type = 'Loamy'     # soil type as string
nitrogen = 90           # N level
phosphorous = 42        # P level
potassium = 43          # K level

# Get recommendations
crop, fertilizer = recommend_crop_and_fertilizer(
    temparature, 
    humidity, 
    moisture, 
    soil_type, 
    nitrogen, 
    phosphorous, 
    potassium
)

print(f"Recommended Crop: {crop}")
print(f"Recommended Fertilizer: {fertilizer}")
```

### 3. **Deploying the Web Application (Optional)**

*If you have built a web interface using Streamlit, Flask, or Django, follow the respective framework's instructions to run the application.*

#### Example: Running a Streamlit App

```bash
streamlit run app.py
```

*This will launch the web interface in your default browser, allowing users to input parameters and receive recommendations.*

---

## 📁 Project Structure

```
crop-fertilizer-recommendation/
├── Dataset/
│   ├── crop_recommendation.csv
│   └── fertilizer_prediction.csv
├── models/
│   ├── crop_model.pkl
│   ├── fertilizer_model.pkl
│   └── scalers/
│       ├── crop_scaler.pkl
│       └── fertilizer_scaler.pkl
├── notebooks/
│   ├── Crop_Recommendation.ipynb
│   └── Fertilizer_Recommendation.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train_crop_model.py
│   ├── train_fertilizer_model.py
│   └── recommendation_system.py
├── app.py  # If using Streamlit or another web framework
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🤝 Contributing

Contributions are welcome! Whether it's reporting bugs, suggesting features, or submitting pull requests, your input is valuable.

### Steps to Contribute

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add your message here"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

*Ensure your contributions adhere to the project's code of conduct and guidelines.*

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

*Replace with your chosen license. Ensure you include a `LICENSE` file in your repository.*

---

## 📧 Contact

- **Your Name** - [@yourusername](https://github.com/yourusername)
- **Email**: your.email@example.com

*Feel free to reach out for questions, suggestions, or collaborations!*

---

## Acknowledgements

- [Scikit-Learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Streamlit](https://streamlit.io/) *(if used)*
- [Other Resources](links to tutorials, articles, etc.)

---

## 📷 Screenshots

*Include screenshots of your application interface, model training outputs, or any other relevant visuals to showcase your project.*

![Crop Recommendation](path_to_crop_recommendation_screenshot)
![Fertilizer Recommendation](path_to_fertilizer_recommendation_screenshot)
![Web Interface](path_to_web_interface_screenshot)

---

## Additional Tips

- **Keep It Updated**: Regularly update the README as your project evolves.
- **Use Badges**: Add badges for build status, license, etc., to enhance the README's appearance. Example:

  ```markdown
  ![License](https://img.shields.io/badge/license-MIT-blue.svg)
  ![Python Version](https://img.shields.io/badge/python-3.8-blue.svg)
  ```

- **Highlight Key Achievements**: Mention if your model achieved significant accuracy or any other milestones.

---

**Happy Coding! 🚀**

---
