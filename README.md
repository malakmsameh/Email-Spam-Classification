# **Email Spam Classification using Naive Bayes**

This project implements a Naive Bayes classifier to classify emails as spam or not spam (ham). It is part of a Pattern Recognition course and showcases how to approach spam email detection using machine learning.

## **Project Description**

The objective of this project is to develop a model that can effectively classify emails as spam or non-spam. The Naive Bayes algorithm is chosen due to its simplicity, efficiency, and effectiveness in handling text classification tasks, particularly in the case of spam detection.

## **Libraries Used**

The following Python libraries are used in this project:

- **`numpy`**: For numerical computations.
- **`pandas`**: For data manipulation and analysis.
- **`matplotlib.pyplot`**: For creating visualizations.
- **`seaborn`**: For enhanced data visualization.
- **`scikit-learn`**:
  - **`naive_bayes`**: Implements the Naive Bayes classifier.
  - **`svm`**: Support Vector Machine classifier (imported, but not the primary focus).
  - **`tree`**: Decision Tree classifier (imported, but not the primary focus).
  - **`model_selection`**: For splitting data into training and testing sets.
  - **`base`**: Base classes for scikit-learn models.
  - **`preprocessing`**: For preprocessing tasks such as scaling and encoding.
  - **`feature_extraction`**: For converting text data into numerical format.
  - **`metrics`**: For evaluating the performance of the model.

## **Dataset**

The dataset used in this project is `combined_data.csv`. It contains email data labeled as either spam (1) or non-spam (0). The dataset has two key columns:

- **`label`**: Indicates whether the email is spam (1) or not spam (0).
- **`text`**: The text content of the email.

## **Exploratory Data Analysis (EDA)**

The EDA process involves:

- **Loading the dataset**: Using `pandas` to load the data.
- **Inspecting the data**: Displaying the first few rows (`df.head()`) and the last few rows (`df.tail()`).
- **Understanding the dataset**: Checking the shape of the dataset (`df.shape`) to see how many rows and columns are present.
- **Data info**: Using `df.info()` to get a summary of data types and non-null counts.
- **Target variable analysis**: Analyzing the distribution of the target variable (`label`) to assess the balance between spam and non-spam emails.

## **Feature Extraction**

To prepare the text data for machine learning models, it is transformed into a numerical format using text vectorization techniques such as:

- **TF-IDF (Term Frequency-Inverse Document Frequency)**
- **Count Vectorization**

Further details about the vectorization method used in the project can be added here.

## **Model Training and Evaluation**

The steps for model training and evaluation include:

1. **Data Split**: The dataset is divided into training and testing sets.
2. **Model Training**: A Naive Bayes model (e.g., Multinomial Naive Bayes) is trained on the training set.
3. **Prediction**: The trained model is used to predict labels for the test data.
4. **Evaluation**: The model is evaluated using various metrics such as:
   - **Accuracy**
   - **Precision**
   - **Recall**
   - **F1-score**
   - **Confusion matrix**

## **Prediction on New Sentences**

The project includes functionality to use the trained model to predict whether new, unseen email text is spam or not.

## **How to Run the Project**

1. **Install Dependencies**: Ensure you have Python 3.x installed and the required libraries. You can install them using `pip`:

    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```

2. **Dataset**: Download the `combined_data.csv` dataset and place it in the same directory as the notebook, or update the file path in the notebook.

3. **Run the Notebook**: Open and run the `Email Spam Classification Project (1).ipynb` notebook using Jupyter Notebook or JupyterLab. Execute the cells sequentially to replicate the analysis and train/test the model.

## **Results**

The notebook will display the performance metrics of the Naive Bayes classifier, which will give insight into how well the model classifies emails as spam or not. Additionally, predictions on new sentences will demonstrate the model's ability to classify previously unseen email content.

## **Further Improvements**

- **Experiment with different vectorization techniques**: Compare the performance of various text vectorization methods, such as TF-IDF and CountVectorizer.
- **Explore other classification algorithms**: Try classifiers like Support Vector Machines (SVM) or Logistic Regression, and compare their performance to Naive Bayes.
- **Hyperparameter tuning**: Perform hyperparameter optimization to enhance the performance of the Naive Bayes model.
- **Advanced text preprocessing**: Implement steps like stemming, lemmatization, and stop word removal to improve model accuracy.
- **Dataset expansion**: Consider using a larger and more diverse dataset to improve the model's generalization.

