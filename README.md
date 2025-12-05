# ❤️ Heart Disease Prediction Using Machine Learning (Random Forest Classifier)

This project predicts the **risk of heart disease** (Low / Moderate / High) using a **Random Forest Classifier**.  
Users enter their health details through an interactive web interface, and the system shows:

- Heart disease risk prediction  
- Model confidence  
- Full ML visualizations (heatmap, boxplot, histograms, scatterplots)  
- Medical feature explanations  
- Prediction history (stored in database)

---

## 🚀 Project Workflow

### 1️⃣ **Data Collection**
- Dataset: Heart Disease Dataset (13 medical features)
- Features include: Age, Sex, Chest Pain Type, Blood Pressure, Cholesterol, Max Heart Rate, ST Depression, CA, Thalassemia, etc.
- Target: `1 = Disease`, `0 = No Disease`

---

### 2️⃣ **Data Preprocessing**
Performed in Python:
- Handling missing values  
- Converting categorical data  
- Scaling features  
- Splitting into train & test sets  

---

### 3️⃣ **Model Training**
Machine Learning Models:
- **Random Forest Classifier** (main model)
- Gradient Boosting
- Ensemble Voting Classifier

✔ Random Forest chosen because:
- High accuracy  
- Handles outliers  
- Prevents overfitting  
- Works well with small datasets  

The trained model is saved using **joblib**.

---

### 4️⃣ **Web Application (Flask Backend)**
Flask handles:
- Receiving user input from the HTML form  
- Loading the trained ML model  
- Making prediction in real-time  
- Sending the risk result + confidence back to frontend  
- Saving prediction history  

---

### 5️⃣ **Frontend (HTML + CSS + Bootstrap)**
User can enter:
- Age  
- Chest pain type  
- BP  
- Cholesterol  
- Max heart rate  
- ECG  
- Thalassemia  
- Major vessels  

Frontend displays:
- Risk level  
- Confidence  
- ML graphs  

---

### 6️⃣ **Visualization**
The system displays 7 ML visualizations:
- 🔥 **Correlation Heatmap**
- 📦 **Boxplots (Disease vs No Disease)**
- 📊 **Histograms / Distribution**
- ⚡ **Feature Importance**
- 🔵 **Scatterplot Matrix**
- ✔ **Confusion Matrix**
- 📈 **ROC Curve**

These help explain *why* the model predicted a certain risk.

---

## 🏗 Technologies Used

### 👩‍💻 Backend
- Python  
- Flask  
- Scikit-learn  
- Pandas  
- NumPy  
- Joblib  

### 🎨 Frontend
- HTML  
- CSS  
- Bootstrap  

### 🗃 Database
- SQLite (stores user prediction history)

---

## 📦 Project Structure
