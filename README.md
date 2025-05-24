# 🎓 Student Math Score Prediction

This project applies **Linear Regression** to predict students' math scores based on various demographic and educational features using the [Students Performance Dataset](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams).

---

## 📁 Dataset Features
The dataset includes:
- Gender
- Race/ethnicity
- Parental level of education
- Lunch type
- Test preparation course
- Reading score
- Writing score

---

## ⚙️ Machine Learning Model
- **Type**: Regression
- **Model**: `LinearRegression` from `sklearn.linear_model`
- **Target Variable**: `math score`
- **Preprocessing**:
  - Categorical features encoded using `pd.get_dummies(drop_first=True)`
  - Data split using `train_test_split`

---

## 🧪 How to Run This Project
1. Clone the repository.
2. Place the dataset in the root directory.
3. Run `math_score_prediction.py` in Spyder or Jupyter Notebook.
4. Make sure these Python libraries are installed:
   - `pandas`
   - `numpy`
   - `scikit-learn`
   - `matplotlib`
   - `seaborn`

---

## 📊 Evaluation Metrics
- **R² Score**: `0.85` ✅
- **MSE**, **MAE** (optional to include in next steps)
- Actual vs Predicted plot:
  
  ![Actual vs Predicted](b97588c1-fca2-44b8-8187-4f91bf6a9296.png)

---

## 🔍 Future Work
- Try advanced models: `Random Forest`, `XGBoost`, `SVR`
- Use cross-validation for more stable evaluation
- Visualize feature importances
- Add pipeline & hyperparameter tuning

---


## ✍️ Author
**Kotha** — Aspiring Machine Learning Engineer 💻  
