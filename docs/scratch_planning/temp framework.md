To achieve a zero-maintenance, modular, and completely free production loop, the ideal strategy is an **On-Device Inference Architecture paired with a Serverless Cloud Feature Pipeline**.

Running the model directly inside your Android app via **TensorFlow Lite (TFLite)** completely eliminates server maintenance, hosting costs, and cloud scaling issues. Meanwhile, a GitHub Actions workflow can process features and retrain the model entirely for free.

---

## 1\. High-Level Architecture Overview

\[ Data Sources \]       \[ Free Automation \]       \[ Free Cloud Storage \]       \[ Client App \]  
\+------------+                                                               

| Env Canada | \-----\\  \+-----------------+       \+--------------------+       \+-----------------+  
\+------------+       \-\>| GitHub Actions  | \----\> |   GitHub Release   | \----\> |   Android App   |  
\+------------+       \-\>| (Kedro / Poetry)|       | (yql\_model.tflite) |       |  (TFLite Run)   |

| Tempest/WU | \-----/  \+-----------------+       \+--------------------+       \+-----------------+  
                                                            ^                          |

                                                            | (Pulls Current           | (Inference Engine)  
                                                            |  Raw Features)           v  
                                                            \+----------------- \[ Live Station APIs \]

---

## 2\. The Modular Data & ML Pipeline (`Kedro`)

**Kedro** is an exceptional choice for this project. It enforces a strict separation between **Data Ingestion (Nodes)**, **Business Logic (Pipelines)**, and **Data Definitions (The Data Catalog)**.

## **Abstracting the Weather Sources**

To ensure your architecture can easily swap weather station sources, you must define an abstract Python interface for data fetching.

*\# src/weather\_project/pipelines/data\_ingestion/interface.py*  
from abc import ABC, abstractmethod  
import pandas as pd

class WeatherStationConnector(ABC):  
    @abstractmethod  
    def fetch\_hourly\_data(self, station\_id: str, days\_back: int) \-\> pd.DataFrame:  
        """Returns a standardized dataframe: \[timestamp, temp, humidity, pressure, rain\_binary\]"""  
        pass

You can write distinct implementations for `EnvironmentCanadaConnector`, `TempestAPIConnector`, and `WeatherUndergroundConnector`. Swapping stations is a simple matter of changing the instantiation key in your Kedro `parameters.yml` file.

## **The Modular Kedro Structure**

Your project pipeline will be split into three distinct, decoupled segments:

1. **`data_ingestion_pipeline`**: Loops through your active connectors, extracts raw JSON/CSV data, standardizes column headers, and saves them as a *Bronze layer* dataset.  
2. **`feature_engineering_pipeline`**: Takes the unified data, generates your cyclical sine/cosine time stamps, rolling averages, and lag features, and writes out a *Silver feature store* dataset.  
3. **`model_training_pipeline`**: Ingests features, trains your model, converts the final weights to `.tflite` format, and outputs the model binary.

---

## 3\. The 100% Free Automation Engine (`GitHub Actions`)

Because you do not want to maintain a local server, **GitHub Actions** can serve as your serverless orchestrator. GitHub provides up to 2,000 free runner minutes per month for public repositories, which is more than enough compute time to process a localized dataset.

* **Automated Inference Updates:** A cron-scheduled GitHub Action can run every hour to trigger your Kedro ingestion pipeline, fetch the latest station readings, and format them into a tiny, lightweight JSON file containing the raw input features.  
* **Automated Retraining:** A separate workflow can run once a month to pull a full year of historical records, retrain the model weights, evaluate metrics, and save them.  
* **The "No-Cost Storage" Hack:** Instead of paying for a GCP Cloud Storage Bucket, your GitHub Action workflow can programmatically upload the newly compiled `model.tflite` binary and the hourly current feature JSON file directly to a static **GitHub Release asset** or a `gh-pages` branch. Your Android app can simply ping that public URL for free to get updates.

---

## 4\. On-Device Android Deployment (`TFLite`)

Running inference directly on-device removes all API hosting costs and ensures the app functions flawlessly offline.

* **The Model Choice:** Use a **Multi-Layer Perceptron (MLP) Neural Network** built-in TensorFlow/Keras instead of XGBoost. While gradient-boosted trees are highly accurate on tabular data, compiling them to run natively on mobile devices is difficult. A simple neural network converts seamlessly into a `.tflite` file.  
* **Multi-Task Output Structure:** Design your model to have a single input layer (the weather station features) but **two distinct output layers**:  
  1. **Output 1 (Regression Line)**: A linear activation layer outputting a single continuous number representing the predicted hourly temperature.  
  2. **Output 2 (Classification Line)**: A sigmoid activation layer outputting a decimal value between `0.0` and `1.0` representing the exact hourly probability of rain.  
* **App Architecture:** When the user opens the Android app, it performs a lightweight background HTTP `GET` request to your static GitHub URL to fetch the current feature JSON and check for an updated model binary. The app then feeds those inputs into the local **TFLite runtime engine**, displaying instant hourly forecasts without querying an expensive cloud database.

---

## 5\. Summary of Key Advantages

| Requirement | How This Architecture Addresses It |
| :---- | :---- |
| **No Server Maintenance** | Handled completely by serverless **GitHub Actions** workflows. |
| **Completely Free** | Combines free GitHub runner minutes with on-device mobile compute power. |
| **Modular Framework** | **Kedro** separates your data source connectors from your core modeling logic. |
| **Dual Target Output** | A single **TFLite Multi-Task Model** delivers both temperature and rain probability simultaneously. |

## Next Steps to Proceed

To start building the foundation of this setup, let me know if you would like me to:

* Draft the exact **GitHub Actions `.yml` workflow file** to handle the automated cron execution.  
* Write a **Keras code snippet** demonstrating how to build a model with a split output for regression (temperature) and classification (rain probability).

