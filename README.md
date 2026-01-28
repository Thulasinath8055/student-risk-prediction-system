Student Performance Risk Prediction System
=========================================

PROJECT OVERVIEW
----------------
Real-time academic early-warning system built with scikit-learn pipeline and Streamlit to predict student 
failure risk using the UCI Student Performance dataset. Interview-ready | GATE DA aligned | Production-grade.

GATE DA Concepts Demonstrated:
- Probability: Failure likelihood via predicted probabilities
- Classification: Binary Pass (0) vs Fail/At-Risk (1)  
- Logistic Regression: Primary model with L1/L2 regularization
- Bias-Variance: LogReg vs Decision Tree comparison
- Evaluation Metrics: Precision, Recall, ROC-AUC, Confusion Matrix

MODEL PERFORMANCE
-----------------
Weighted Logistic Regression (Production Model):
Validation Accuracy:     71.0%
Fail-class Precision:    55.0%
Fail-class Recall:       65.0%  (Optimized for early warning)
F1-score (fail):         60.0%
ROC-AUC:                 0.76

Top Risk Factors: failures, absences, low studytime, high goout

QUICK START
-----------
1. Clone repository
   git clone https://github.com/Thulasinath8055/student-risk-prediction-system
   cd student-risk-prediction

2. Setup environment
   python -m venv venv
   source venv/bin/activate  (Linux/Mac)
   venv\Scripts\activate     (Windows)

3. Install dependencies
   pip install -r requirements.txt

4. Run web app
   streamlit run app.py
   Open: http://localhost:8501

PROJECT STRUCTURE
-----------------
student-risk-prediction/
├── models/
│   └── student_risk_pipeline.joblib     (Trained pipeline)
├── notebooks/
│   └── 01_student_risk_analysis.ipynb   (Full ML analysis)
├── app.py                               (Streamlit web app)
├── data/
│   └── student-mat.csv                  (UCI dataset)
├── requirements.txt
└── README.md

FEATURES
--------
- Production scikit-learn pipeline (preprocessing + Logistic Regression)
- Interactive Streamlit web interface for real-time predictions
- Explainable AI via model coefficients + feature importance
- Class imbalance handling (class_weight="balanced")
- Comprehensive evaluation metrics
- GATE DA aligned concepts throughout

PREDICTION FACTORS (22 indicators)
----------------------------------
Academic: past failures, studytime, absences
Behavioral: goout frequency, alcohol consumption (Dalc/Walc)
Support: schoolsup, famsup, activities
Demographics: age, sex, family size, parental status

INTERVIEW HIGHLIGHTS
--------------------
"Built complete ML lifecycle: EDA → feature engineering → model selection 
(LogReg vs Decision Tree) → class imbalance handling → production deployment. 
Achieved 65% recall on fail class (vs 38% baseline) using class weights. 
Deployed explainable predictions via Streamlit for educators."

COMPARATIVE RESULTS
-------------------
Model                    | Accuracy | Fail Recall | Fail Precision | ROC-AUC
-------------------------|----------|-------------|----------------|--------
LogReg (weighted)        | 71.0%    | 65.0%       | 55.0%          | 0.76
LogReg (baseline)        | 72.1%    | 38.4%       | 62.5%          | 0.77
Decision Tree            | 58.2%    | 50.0%       | 39.3%          | -

MODEL INSIGHTS
--------------
Risk-increasing (positive coefficients):
1. failures (+0.61) - Past academic failures
2. schoolsup_yes (+0.61) - Extra support (selection bias)
3. goout (+0.42) - Frequent socializing
4. age (+0.24) - Older students

Risk-reducing (negative coefficients):
1. higher_yes (-0.66) - Higher education ambition
2. sex_M (-0.30) - Male students
3. studytime (-0.09) - More study hours

REQUIREMENTS.TXT CONTENTS
-------------------------
streamlit==1.36.0
scikit-learn==1.5.1
pandas==2.2.2
numpy==1.26.4
matplotlib==3.9.2
seaborn==0.13.2
joblib==1.4.2

AUTHOR
------
Thulasinath Baindla
Data Science Enthusiast
thulasinath8055@gmail.com

LICENSE
-------
MIT License - Free to use, modify, deploy for academic/portfolio purposes.
