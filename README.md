🩺 Radiology Agent — Capstone Project
Project: Automated Chest X-ray Analysis with AI Agents
Author: Reihan Alinia Lat
Competition: Kaggle — Agents Intensive Capstone Project.
🔍 Project Overview

This project implements a modular AI pipeline for automated chest X-ray analysis:

🖼 ImageAnalysisAgent — classifies X-ray images into Normal, Pneumonia, or Other.

📝 CodingAgent — automatically assigns ICD-10 and CPT codes.

📄 ReportGenerationAgent — generates patient reports with confidence scores.

Highlights:

Fine-tuned ResNet18 backbone

Thresholding for low-confidence predictions (Other)

Test-Time Augmentation (TTA) for robust results

Fully modular and reproducible pipeline

📂 Repository Structure
'''
radiology-agent/
│
├── agents/
│   ├── image_analysis_agent.py
│   ├── coding_agent.py
│   ├── report_generation_agent.py
│   └── memory_agent.py
│
├── models/
│   └── chest_classifier.pt  # Fine-tuned ResNet18 weights
│
├── sample_images/           # Example X-ray images
├── utils/                   # Preprocessing & I/O helpers
├── memory/                  # Optional patient database
├── run_pipeline.py          # Run the full pipeline
├── requirements.txt         # Python dependencies
└── README.md
'''
