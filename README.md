


# 🩺 Radiology Agent — Capstone Project

**Project:** Automated Chest X-ray Analysis with AI Agents  
**Author:** Reihan Alinia Lat  
**Competition:** *Kaggle — Agents Intensive Capstone Project*

---

## 🔍 Overview

This repository presents a multi-agent radiology analysis system designed for automated chest X-ray interpretation.

The pipeline integrates deep learning with modular, autonomous agents—each responsible for a specific clinical task.

Unlike traditional single-model workflows, this project highlights Agentic AI, the core theme of the Kaggle Capstone.



### Highlights:

- Fine-tuned **ResNet18** backbone  
- Thresholding for low-confidence predictions (`Other`)  
- **Test-Time Augmentation (TTA)** for robust results  
- Fully **modular and reproducible** pipeline
---
  ## 📦 Repository Structure

```
radiology-agent-adk-capstone/
│
├── agents/
│   ├── image_analysis_agent.py
│   ├── coding_agent.py
│   ├── report_generation_agent.py
    ├── memory_agent.py
│   └── PatientContextAgent.py
│
├── master_agent.py             ← Orchestrator
├── main.py                    ← CLI / script runner
├── run_colab.ipynb         ← Colab/ Notebook for full inference 
├── training/
│   └── train_finetune_colab.ipynb         ← optional: fine‑tuning from scratch
├── models/                    ← contains / expects pretrained weights
│   └── chest_classifier.pt
├── sample_images/             ← example X-ray inputs
├── memory/                    ← for patient-history JSON
│   └── patient_db.json
├── requirements.txt
└── README.md
```

---
## 📂 Dataset / Images ![Dataset](https://img.shields.io/badge/Dataset-Figshare-blue)

### **🧰 Training Dataset (for model fine-tuning)**
The model was fine-tuned using the **Chest X-Ray Pneumonia dataset** from Kaggle:  
[🔗 Dataset link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

You can download it either:  
- **Manually**, or  
- **Automatically** using the Kaggle API

**Kaggle API setup example:**

```
kaggle_json = {
    "username": "<YOUR_KAGGLE_USERNAME>",
    "key": "<YOUR_KAGGLE_KEY>"
}

```


### 🖼 Sample Images (for pipeline demonstration)

The chest X-ray images used in this project are included in the repository under the `sample_images/` folder.  

These images are sourced from the **COVID‑19 Chest X‑Ray Image Repository**, a public dataset hosted on Figshare:  
- **Dataset link:** [🔗 COVID‑19 Chest X‑Ray Image Repository](https://figshare.com/articles/dataset/COVID-19_Chest_X-Ray_Image_Repository/12580328)  
- **License:** CC‑BY 4.0

---

## 📈 Model

The project uses a **ResNet-18** model pretrained on **ImageNet**.  

### 🔹 Architecture
- Final fully-connected (FC) layer modified to predict:
  - **Normal**
  - **Pneumonia**
  - **Other**

### 🔹 Training Details
- Optimizer: **AdamW**  
- Input normalization: **ImageNet normalization**  
- Loss function: **Weighted cross-entropy** (used if needed)
  
---

 ### 🤖 Agent Architecture

- **🖼 ImageAnalysisAgent** — Classifies X-ray images into `Normal`, `Pneumonia`, or `Other`.
- **📝 CodingAgent** — Automatically assigns ICD-10 and CPT codes.
- **📄 ReportGenerationAgent** — Generates patient reports with confidence scores.
- **💾 MemoryAgent** — Maintains patient history and previous analysis results.
-  🧩 **PatientContextAgent** — Provides patient metadata (ID, name, age) for each request.
- 🎛 **MasterAgent** — Orchestrates the interaction between all agents for streamlined processing.
```
                     ┌─────────────────────────┐
                     │      MasterAgent        │
                     └──────────┬─────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
   ┌───────────────┐    ┌────────────────┐    ┌──────────────────────┐
   │ ImageAnalysis │    │   CodingAgent  │    │ ReportGenerationAgent│
   └───────────────┘    └────────────────┘    └──────────────────────┘
                                │
                                ▼
                        ┌───────────────┐
                        │ MemoryAgent   │
                        └───────────────┘
                                │
                                ▼
                        ┌────────────────────┐
                        │ PatientContextAgent│
                        └────────────────────┘
```
   
 ---
 

### 📥 Download Model Weights (Required)

GitHub restricts files larger than 25MB, so the model weights (**chest_classifier.pt**, ~43MB) are hosted on **Google Drive**.  
You **must download the weights** before running the pipeline.

#### 🔹 Options to download:

1. **Via browser:**  
   👉 [Download chest_classifier.pt](https://drive.google.com/file/d/1mDpUmGjR5OKXodd8DxFJVsR-iMsrPuIb/view?usp=sharing)

2. **Via command line:**

Via command line / Colab:
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

## 📈 Results

The pipeline provides:

- ✅ **Accurate classification of chest X-rays**  
- 📝 **Automatic ICD-10 & CPT coding**  
- 📄 **Clear patient reports** with confidence scores  
- 🧩 **Modular agent architecture** — each agent can be updated independently  
- ⚡ **Test-Time Augmentation (TTA)** improves robustness


⚡ Notes

Ensure you have a GPU for faster inference

Test-Time Augmentation improves robustness on unseen images

---

## 🧠 Why This Matters

This project goes beyond a standard classifier by demonstrating an **agentic workflow** inspired by real-world clinical radiology systems. Key aspects include:

- ✔ **Multi-agent collaboration** — agents work together seamlessly  
- ✔ **Task delegation** — each agent focuses on specialized tasks  
- ✔ **Memory-based state handling** — preserves context across the pipeline  
- ✔ **Modular diagnostic pipeline** — components can be updated or replaced independently  
- ✔ **Realistic radiology workflow** — closely resembles clinical systems

  This repository evaluates both:  
-*Technical ML execution*, and  
-*Agentic workflow understanding*

By combining these elements, the project showcases how **AI systems can be designed to handle complex diagnostic workflows** in a structured and scalable way.

---

## 🚀 Getting Started

### ☁️ Run on Google Colab

You can run the **full chest X-ray analysis pipeline** directly on Google Colab using the uploaded notebook (`run_colab.ipynb`):

[🔗 Open `run_colab.ipynb` in Colab](https://github.com/reyhan1994/radiology-agent-adk-capstone/blob/main/run_colab.ipynb)

### ✅ Features / Steps Handled
- ⚡ **Check GPU availability**  
- 📂 **Clone the repository**  
- 📦 **Install dependencies**  
- 🔑 **Set up Kaggle API credentials**  
- 🩺 **Download the Chest X-ray Pneumonia dataset**  
- 💾 **Mount Google Drive** to load/save model weights  
- 🏋️‍♂️ **Load the fine-tuned ResNet-18 model**  
- 🖼 **Run inference on sample images**  
- 📝 **Generate a CSV submission file**

> ✅ **Important:** You **only need to provide your own Kaggle username and API key** to access the datasets.

---
🏋️ Training / Fine-Tuning

If you want to retrain or fine-tune the model from scratch:

Open the notebook:
```
training/train_finetune_colab.ipynb
```

 

---
📜 License

This project is licensed under the MIT License.

