Predicting the **Probability of Precipitation (PoP)** on an hourly basis presents an entirely different mathematical challenge due to "zero-inflation" (it does not rain most hours). The literature specifically shifts toward **classification models, logistic regressions, and probabilistic skill scores**. \[1\]

The following targeted reference document outlines academic literature, mathematical baselines, and open-source frameworks specifically focusing on predicting **rain probability and occurrences** from tabular weather station data:

---

## Literature & Open-Source Reference Document: Probability of Precipitation (PoP) \[2, 3\]

## 1\. Academic Literature (PoP & Hourly Probability)

## A. Postprocessing of NWP Precipitation Forecasts Using Deep Learning

* **Source:** *American Meteorological Society (AMS) / Weather and Forecasting (2023)* \[4, 5\]  
* **Summary:** This paper details the use of Artificial Neural Networks (ANNs) explicitly split into two components: one for predicting the **probability of precipitation** (a binary classification task) and another for predicting the amount. The authors evaluate their neural networks using the **Brier Skill Score (BSS)** to prove that machine learning models consistently out-calculate traditional logistic regressions by better mapping non-linear local variables. \[4\]  
* **Context for Your Project:** This provides the exact validation you need for your split architecture. It demonstrates that training an **ANN classifier** on localized station data provides highly calibrated probability curves (Brier Scores approaching near-perfection for short lead times), fitting perfectly into a **TFLite mobile deployment**. \[4, 6\]

## B. Efficacy of Machine Learning in Simulating Precipitation and Its Extremes \[7\]

* **Source:** *Research Square / Atmospheric Research (2024)* \[7\]  
* **Summary:** This study benchmarks machine learning classifiers (specifically Random Forest and Support Vector Classifiers) to predict precipitation occurrence across various meteorological terrains. The researchers highlight that **dew point temperature and relative humidity** possess a strong positive correlation with localized rain generation. \[7\]  
* **Context for Your Project:** It outlines the exact feature importance matrix for a tabular rain predictor. When writing your Kedro feature-engineering pipeline, this paper indicates that your model will derive its heaviest "rain-signal" weight from *changes in the dew point depression* (the gap between air temperature and dew point) and sudden drops in station pressure, rather than temperature alone. It achieved rain classification accuracies of **79% to 83%**. \[7\]

## C. Probabilistic Modeling of Hourly Precipitation Jointly with Gridded Datasets

* **Source:** *Springer / Mathematical Geosciences (2026)* \[1\]  
* **Summary:** This study specifically tackles the problem of **hourly precipitation**. To deal with the "zero-inflation challenge" (the fact that most hourly rows in a dataset show 0 mm of rain), the researchers utilize a framework that completely isolates *precipitation occurrence* (binary 0/1) from *precipitation intensity*. \[1\]  
* **Context for Your Project:** This justifies your exact goal: focusing strictly on whether it will rain or not. It confirms that trying to predict rain *amounts* degrades model skill, but isolating the hourly *occurrence* into a localized classification curve yields highly robust, operationally valuable probabilistic maps. \[1\]

---

## 2\. GitHub Frameworks & Rain Classifiers

## A. Rain Detection / Rainfall Prediction Classifier \[3\]

* **Source:** ShubhamRasal/Rainfall-Prediction-Using-Machine-Learning on GitHub  
* **Summary:** An open-source implementation built specifically to parse tabular historical weather station data (including pressure, humidity, and temperature) to output a rain probability. It directly targets the **class imbalance problem** where dry days dwarf rainy days.  
* **Context for Your Project:** This serves as a structural blueprint for the training phase of your ML pipeline. It provides clean Python code for evaluating a rain model using **ROC-AUC scores, Precision-Recall curves, and Confusion Matrices** rather than standard regression metrics, showing you how to tune your classifier to ensure it doesn't just guess "0% chance of rain" to manipulate its accuracy score.

## B. Weather Analytics & Rain Risk Pipelines

* **Source:** vitorbento/rain-prediction on GitHub  
* **Summary:** A developer pipeline showcasing how to transform a traditional tabular weather sequence into a binary classification problem. It emphasizes creating **lag variables** specifically for rain (e.g., "did it rain 1 hour ago? 2 hours ago?") to create a short-term probability loop.  
* **Context for Your Project:** This applies perfectly to your hourly target. By incorporating time-lagged variables of precipitation from nearby stations (like Picture Butte or Lethbridge Airport), your Kedro pipeline can feed the model structural indicators of a rain cell physically moving through the county toward your location.

---

## Key Takeaway for Your Framework

The literature indicates that your rain probability model must utilize a **Binary Cross-Entropy (Log Loss)** loss function. When deployed on-device via TFLite, the model will output a continuous value between `0.0` and `1.0` via a final **Sigmoid activation function**. This decimal value is interpreted natively by your Android app as the exact hourly **Probability of Precipitation (PoP)**.

Would you like to draft a **Kedro node code snippet** that handles the class imbalance problem (e.g., down-sampling dry hours or applying class weights) so your training data is balanced for rain prediction?

\[1\] [https://link.springer.com](https://link.springer.com/10.1007/s11004-026-10286-w)  
\[2\] [https://www.weather.gov](https://www.weather.gov/media/pah/WeatherEducation/pop.pdf)  
\[3\] [https://www.weather.gov](https://www.weather.gov/lmk/pops)  
\[4\] [https://journals.ametsoc.org](https://journals.ametsoc.org/view/journals/wefo/38/3/WAF-D-21-0207.1.xml)  
\[5\] [https://www.theweathernetwork.com](https://www.theweathernetwork.com/en/news/science/explainers/explainer-what-p-o-p-really-means-in-a-forecast-probability-of-precipitation)  
\[6\] [https://iwaponline.com](https://iwaponline.com/jwcc/article/15/4/1729/101038/Performance-evaluation-and-verification-of-post)  
\[7\] [https://www.researchsquare.com](https://www.researchsquare.com/article/rs-4339400/latest.pdf)  
