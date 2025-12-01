# 🩺 Radiology Agent — Capstone Project

**Project:** Automated Chest X-ray Analysis with AI Agents  
**Author:** Reihan Alinia Lat  
**Competition:** Kaggle — Agents Intensive Capstone Project  

---

## 🔍 Overview

This project implements a **modular AI pipeline** for automated chest X-ray analysis. The system leverages deep learning and agent-based architecture to classify X-rays, assign medical codes, and generate patient reports.

### Agents:

- **🖼 ImageAnalysisAgent** — Classifies X-ray images into `Normal`, `Pneumonia`, or `Other`.
- **📝 CodingAgent** — Automatically assigns ICD-10 and CPT codes.
- **📄 ReportGenerationAgent** — Generates patient reports with confidence scores.
- **💾 MemoryAgent** — Maintains patient history and previous analysis results.

### Highlights:

- Fine-tuned **ResNet18** backbone  
- Thresholding for low-confidence predictions (`Other`)  
- **Test-Time Augmentation (TTA)** for robust results  
- Fully **modular and reproducible** pipeline  

---
## 📂 Dataset / Images ![Dataset](https://img.shields.io/badge/Dataset-Figshare-blue)

**Training Dataset (for model fine-tuning):**  
The model was fine-tuned using the **Chest X-Ray Pneumonia dataset** from Kaggle:  
[Dataset link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

How to use the dataset:

Manual download: Download the dataset from Kaggle and upload it to your Colab or local machine.

Optional Kaggle API download: If you prefer, you can download directly using the Kaggle API (requires your own API key
Setting up Kaggle API Credentials:
To download datasets or submit to Kaggle using this notebook, you need to provide your Kaggle API credentials.

1.Go to your Kaggle account and create an API token. This will download a file named kaggle.json.

2.Place kaggle.json in your Colab environment at:
```
/root/.kaggle/kaggle.json
```
3.Example structure in the notebook:
```
kaggle_json = {
    "username": "<YOUR_KAGGLE_USERNAME>",
    "key": "<YOUR_KAGGLE_KEY>"
}

```


**Sample Images (for pipeline demonstration):**  
The chest‑X‑ray images used in this project are sourced from the **COVID‑19 Chest X‑Ray Image Repository**, a public dataset hosted on Figshare.  
[Dataset link](https://figshare.com/articles/dataset/COVID-19_Chest_X-Ray_Image_Repository/12580328)  
**License:** CC‑BY 4.0  

---
## 📦 Repository Structure

```
radiology-agent/
│
├── agents/
│   ├── image_analysis_agent.py
│   ├── coding_agent.py
│   ├── report_generation_agent.py
│   └── memory_agent.py
│
├── models/
│   └── chest_classifier.pt       # Fine-tuned ResNet18 weights
│
├── sample_images/                # Example X-ray images
│   └── README.md
│
├── utils/
│   ├── preprocessing.py
│   └── io_utils.py
│
├── memory/
│   └── patient_db.json
│
├── run_pipeline.py               # Main script to run full pipeline
├── requirements.txt              # Dependencies
└── README.md
```
### 📥 **Download Model Weights (Required)**


Because GitHub restricts files larger than 25MB, the model weights (chest_classifier.pt, ~43MB) are hosted on Google Drive.

You must download the weights before running the pipeline.

Download link:
👉👉 **[Download chest_classifier.pt](https://drive.google.com/file/d/1mDpUmGjR5OKXodd8DxFJVsR-iMsrPuIb/view?usp=drive_link)**

Or download via command line:
```
pip install gdown
mkdir -p models
gdown https://drive.google.com/uc?id=1mDpUmGjR5OKXodd8DxFJVsR-iMsrPuIb -O models/chest_classifier.pt


```
This will save the model to:
```
models/chest_classifier.pt
```
Make sure this path matches the one used in your code:
```
weights_path = "models/chest_classifier.pt"
```

---
## 🚀 Getting Started

1. **Clone the repository:**

```bash
!git clone https://github.com/your-username/radiology-agent-adk-capstone.git
%cd radiology-agent-adk-capstone
```
2.Install dependencies:
```
pip install -r requirements.txt
```
3.Run the pipeline on a sample image:
```
!PYTHONPATH=. python main.py --input sample_images --output submission.csv
```
## 🚀 Run Pipeline on Google Colab
The `run_colab.py` script allows you to **run the full chest X-ray analysis pipeline** on Google Colab with **pre-trained/fine-tuned weights** stored in Google Drive. It handles:
- Checking GPU availability
- Cloning the repository
- Installing dependencies
- Setting up Kaggle API credentials
- Downloading the Chest X-ray Pneumonia dataset
- Mounting Google Drive to load/save model weights
- Loading the fine-tuned ResNet18 model
- Running inference on sample images
- Generating a CSV submission file

### How to Use

1. Open the notebook or Colab session.
2. Ensure you have your Kaggle credentials set as environment variables:

```python
os.environ["KAGGLE_USERNAME"] = "<YOUR_KAGGLE_USERNAME>"
os.environ["KAGGLE_KEY"] = "<YOUR_KAGGLE_KEY>"
```
3.Run the script:
```
!python run_colab.py
```
The predictions CSV (submission.csv) will be generated and can be downloaded locally. The fine-tuned model weights will be loaded automatically from your Google Drive path:
/content/drive/MyDrive/radiology_agent/models/chest_classifier.pt
⚡ Notes

Make sure the model weights are uploaded to the specified Google Drive path.

The script uses the same image transformations as during training to ensure consistency.

Works with GPU if available.
---
📈 Results

Accurate classification of chest X-rays

Automatic ICD-10 & CPT coding

Clear and concise patient reports with confidence scores

⚡ Notes

Ensure you have a GPU for faster inference

The pipeline is modular, so agents can be updated independently

Test-Time Augmentation improves robustness on unseen images

---
📜 License

This project is licensed under the MIT License.

