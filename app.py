import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, matthews_corrcoef
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

# Function to load the model and make predictions
def load_and_predict(model_path, img_path):
    model = load_model(model_path)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    prediction = model.predict(img_array)
    return (prediction > 0.5).astype(int)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    st.pyplot()

# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, auc_score):
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    st.pyplot()

# Streamlit App
st.title('Acne Detection Model Evaluation')

# Load the model
model_path = 'VGG16Model.h5'
model = load_model(model_path)


# Display confusion matrix
st.header('Confusion Matrix')
test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    "dataset/testing/",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)
true_labels = test_generator.classes
predictions = model.predict(test_generator)
predicted_labels = (predictions > 0.5).astype(int).reshape(-1)
cm = confusion_matrix(true_labels, predicted_labels)
plot_confusion_matrix(cm, class_names=['Non-Acne', 'Acne'])

# Display classification report
st.header('Classification Report')
report = classification_report(true_labels, predicted_labels, target_names=['Non-Acne', 'Acne'])
st.text(report)

# Display ROC Curve
st.header('ROC Curve')
fpr, tpr, _ = roc_curve(true_labels, predictions)
roc_auc = auc(fpr, tpr)
plot_roc_curve(fpr, tpr, roc_auc)

# Display MCC
st.header('Matthews Correlation Coefficient (MCC)')
mcc = matthews_corrcoef(true_labels, predicted_labels)
st.write(f'MCC: {mcc:.2f}')

# Function to predict single image
st.header('Single Image Prediction')
uploaded_file = st.file_uploader("Choose an image...", type="jpeg")
if uploaded_file is not None:
    img_path = 'temp_image.jpg'
    with open(img_path, 'wb') as f:
        f.write(uploaded_file.getvalue())
    prediction_class = load_and_predict(model_path, img_path)[0][0]
    class_labels = ['Non-Acne', 'Acne']
    predicted_label = class_labels[prediction_class]
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write(f'Predicted Label: {predicted_label}')
