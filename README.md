# 🚀 Diabetic Retinopathy Detection using Vision Transformers  

## 📌 Overview  
This project was developed for the **Infyma AI Hackathon 2025**. It focuses on detecting **Diabetic Retinopathy (DR)** from retinal images using **Vision Transformers (ViT)**. The model classifies retinal images into different severity levels, aiding in early diagnosis and treatment.  

## 📂 Dataset  
- **Source**: [Diabetic Retinopathy Balanced Dataset (Kaggle)](https://www.kaggle.com/datasets/kushagratandon12/diabetic-retinopathy-balanced/data)  
- **Categories**:  
  - **0**: No DR (Healthy)  
  - **1**: Mild DR  
  - **2**: Moderate DR  
  - **3**: Severe DR  
  - **4**: Proliferative DR  
- **Format**: JPEG/PNG images with structured CSV metadata.  

## 📁 File Structure  
```
├── team_name/
│   ├── model/
│   │   ├── diabetic_retinopathy_transformer_balanced.h5   # Trained model
│   ├── notebooks/
│   │   ├── model_training.ipynb  # Jupyter Notebook  
│   ├── report.pdf   # Explanation of approach & results  
│   ├── README.md   # Project documentation  
│   ├── requirements.txt   # Dependencies  
```

## 🛠 Installation  
1. **Clone the repository**:  
   ```bash
   git clone https://github.com/ZayanRashid295/Infyma.git
   cd Infyma
   ```
2. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```
3. **Download and place the dataset** in the `data/` directory.  

## 📊 Model Pipeline  
### 1️⃣ **Data Processing & Augmentation**   
- The dataset is **preprocessed** by selecting **50% of images per class** for training and validation.  
- **Test set remains unchanged** as provided in the dataset.  
- Apply **image resizing, normalization, and augmentation** (rotation, flipping, contrast adjustments).  
  

### 2️⃣ **Building the Model**  
- **Model Architecture**: Vision Transformer (ViT) with Transfer Learning.  
- **Optimizer**: Adam with learning rate scheduling.  
- **Loss Function**: Categorical Cross-Entropy.  

### 3️⃣ **Model Training**  
Run the following command to train the model:  
```bash
jupyter notebook notebooks/model_training.ipynb
```
- Trains for **10 epochs**.  
- Uses **early stopping & batch normalization**.  
- Saves the trained model as `.h5`.  

### 4️⃣ **Evaluation & Explainability**  
- **Metrics**: F1-score, Precision, Recall.  
- **Model Explainability**: Uses **Grad-CAM** to visualize affected areas in retinal images.  
- **Computational Efficiency**: Optimized inference speed using model quantization.  

## 🏆 Evaluation Criteria  
Your submission is judged based on:  
- **40% Accuracy & Performance**: Model precision & recall.  
- **20% Explainability**: Interpretation of predictions.  
- **20% Computational Efficiency**: Speed & optimization.  
- **20% Innovation**: Hybrid models, novel architectures.  

## 🚀 Deployment  
You can deploy the model using:  
To deploy the model for real-time predictions:

1. Install dependencies:
   ```bash
   pip install streamlit tensorflow numpy opencv-python pillow
   ```
2. Run the web app:
   ```bash
   streamlit run app.py
   ```
3. Upload a **retinal image**, and the app will classify its severity.
## 🖼️ Sample Output (App Working)
Here is a preview of the app running with a test image:

![Diabetic Retinopathy Detection App](ModelWorking.png)

📌 *Upload a retinal image to get a prediction!*



(If Streamlit or Flask implementation is added)  

## 📜 Hackathon Rules & Guidelines  
✅ **Allowed Frameworks**: TensorFlow, PyTorch, OpenCV, FastAI, Scikit-Learn.  
✅ **Submission Format**:  
  - Jupyter Notebook (`.ipynb`)  
  - Model Weights (`.h5` or `.pt`)  
  - Short Report (`report.pdf`)  
✅ **Plagiarism**: Unauthorized use of existing solutions will lead to disqualification.  

## 📤 Submission Instructions  
1. **Push your project to GitHub**.  
2. **Submit your GitHub repository link** as per hackathon guidelines.  
3. **Ensure the following files are included**:  
   - ✅ `model_training.ipynb` (Notebook)  
   - ✅ `trained_model.h5` (Saved model)  
   - ✅ `README.md` (Documentation)  
   - ✅ `requirements.txt` (Dependencies)  
   - ✅ `report.pdf` (Explaining approach)  

## 📜 License  
This project is open-source and available under the **MIT License**.  

  

---  
🔥 **Developed for Infyma AI Hackathon 2025**  

## Evalvation and Results

