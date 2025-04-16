# Multi Model Based Disease Prediction using Machine Learning

** 🔍 Abstract** : 
In recent years, the integration of machine learning techniques into healthcare systems has significantly advanced the ability to predict and detect various diseases. This project primarily focuses on early disease detection by ensuring robust model performance through advanced evaluation methods. Our approach stands out from traditional disease prediction methods by employing tailored machine learning algorithms for multiple diseases.The foundation of the system is a comprehensive dataset encompassing a wide range of medical parameters essential for each disease category. These parameters include demographic information, lifestyle factors, clinical indicators, and various symptoms known to influence the onset and progression of diseases. To ensure the reliability and effectiveness of the predictive models, rigorous preprocessing techniques—such as data cleaning, normalization, and feature selection—are applied to enhance the quality of input data.While many existing machine learning models are limited to detecting a single disease, this project introduces a unified system capable of forecasting several diseases through a single user interface. The proposed model can predict conditions such as diabetes, heart disease, chronic kidney disease, cancer, and more. The main objective is to develop a web application that forecasts multiple diseases accurately. The web application is built using the Streamlit Python library for the frontend and communicates with backend machine learning models to predict disease probabilities.

### 👩‍💻💻👩‍💼Project Members
1. DISHA P U  
2. HITHASHREE 
3. NEHA DINESH SHETTIGAR 
4. K NIDHI PRABHU


### 🛠️ Deployment Steps
Please follow the below steps to run this project.
<br>
1. `pip install -r requirements.txt`<br>
2. `cd frontend`<br>
3. `streamlit run multiple_disease_prediction.py`<br><br>


###📁 Platform, Libraries and Frameworks used
1. [Streamlit](https://docs.streamlit.io/library/get-started)
2. [Python](https://www.python.org)
3. [Sklearn](https://scikit-learn.org/stable/index.html)

### 📁Dataset Used
1. [Diabetes disease dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data)
2. [Heart disease dataset](https://www.kaggle.com/datasets/rishidamarla/heart-disease-prediction/data)
3. [Liver disease dataset](https://www.kaggle.com/code/harisyammnv/liver-disease-prediction/data)
4. [chronic kidney](https://www.kaggle.com/datasets/mansoordaku/ckdisease)
5. [liver diseases](https://www.kaggle.com/datasets/uciml/indian-liver-patient-records)
6.[pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

<br></br>

## 📚💡🔍Research Paper: Multi Model Based Disease Prediction using Machine Learning
- **Description:** 
A study that developed a web application capable of predicting multiple diseases, including diabetes, heart disease, Parkinson's, liver disease,breast cancer,symptom based prediction,penumonia, jaundice, and hepatitis, using machine learning algorithms such as SVM, Decision Tree, and Random Forest. The system allows users to input data for a specific disease, and based on the trained model, the output is displayed. The paper also discusses the functional and non-functional requirements of the system, as well as the architecture design and implementation details.

- **📌Future Scope:**
1. Expansion of Symptoms Database
Integrating detailed symptom data from diverse medical sources, including rare or less frequent symptoms, can broaden the platform’s applicability to a wider range of health conditions. This expansion will enable the system to detect overlapping or subtle symptoms, making diagnostic suggestions more accurate and reliable.
2. Basic Patient Management System
Incorporating a basic patient management module would enhance the system's utility. This module could facilitate registration, history tracking, and appointment scheduling with healthcare providers. It could also store and retrieve a patient’s medical history, lab reports, and past predictions, ensuring proper follow-up and continuity of care.
3. Integration with Wearable Devices and IoT
Connectivity with wearable devices and IoT-enabled health trackers would allow real-time monitoring of vital signs. Predictions based on continuous data streams from sensors could provide early warnings about critical health events.
4. Personalized Medicine with AI
Leveraging AI to tailor healthcare recommendations based on individual health profiles, genetic predispositions, and lifestyle factors can significantly enhance personalized medicine. By analyzing user-specific data like medical history, lab results, and genetic information, the system can suggest targeted therapies, optimal dosages, and preventive measures, improving treatment effectiveness.
5. Cloud Scalability and High Availability for Streamlit Application
Deploying the Streamlit application on cloud platforms (e.g., AWS, Azure, Google Cloud) ensures scalability and responsiveness during high traffic. This is crucial for handling large user volumes, especially during health awareness campaigns or pandemics, while maintaining high availability and performance.

### 📰📊💻References
[1]	Khan, A., et al. (2018). "Multi-Disease Prediction Using Support Vector Machines, Random Forests, and Logistic Regression." Journal of Healthcare Informatics, 34(2), 112-123.
[2]	Zhang, L., et al. (2020). "Comparing Decision Trees, Random Forests, and Support Vector Machines for Multi-Disease Prediction." International Journal of Data Science, 45(3), 200-215.
[3]	Ahmed, S., et al. (2021). "Hybrid Models for Enhanced Multi-Disease Prediction Using Deep Learning and Conventional Machine Learning Approaches." Journal of Machine Learning in Medicine, 58(4), 321-334.
[4]	Chen, X., et al. (2017). "A Framework for Incorporating Multi-Disease Prediction Systems into Clinical Practice." Health Informatics Journal, 23(1), 45-58.
[5]	Sharma, R., et al. (2020). "Development of an Intuitive Disease Prediction Platform Using Streamlit." Healthcare Technology Letters, 7(5), 287-298.
[6]	Wang, Y., et al. (2020). "Enhancing Interpretability in Multi-Disease Prediction Models with Explainable AI Techniques." Journal of Artificial Intelligence in Healthcare, 12(2), 67-80.
[7]	Zhang, X., et al. (2019). "Transfer Learning for Multi-Disease Prediction: A Novel Approach with Limited Training Data." Computational Biology and Medicine, 112, 103398.
[8]	Sharma, P., & Guleria, A. (2023). "Deep Learning for Pneumonia Detection Using Chest X-Ray Images: A Review of CNN Models." Journal of Medical Imaging and Health Informatics, 13(2), 456-472.
[9]	Khan, A., et al. (2021). "Deep Learning for Pneumonia Detection Using Chest X-Rays: A Review of CNNs, Ensemble Methods, and Transfer Learning." International Journal of Medical Imaging, 29(1), 109-123.
[10]	 Kibria, H. B., et al. (2023). "Accurate and Interpretable Diabetes Prediction Using a Soft Voting Ensemble of Random Forest, AdaBoost, and Gradient Boosting Models." Journal of Machine Learning and Healthcare, 18(6), 89-98.



