# **FactKG Implementation - Enhanced & Extended**
This repository provides additional scripts and extensions for running AI models and training pipelines based on the **FactKG** framework for fact verification.

## **🔗 Reference to the Original Repository**
This implementation builds on the official **FactKG** repository:
- 📌 **Original GitHub Repository**: [jiho283/FactKG](https://github.com/jiho283/FactKG)

To utilize this project effectively, **you must first download and set up the original repository** alongside the additional scripts provided here.

---

## **📌 Repository Overview**
### **Project Purpose**
This repository enhances the FactKG framework by providing additional code and configurations for claim verification using pre-trained transformer models (e.g., BERT, T5). It includes preprocessing scripts, training pipelines, and evaluation methods for claim verification.

### **Project Structure**
```
📂 FactKG
├── claim_only/        # Claim verification models (BERT, Flan-T5)
├── with_evidence/     # Retrieval-based verification models
├── data_set/          # Dataset files (Pickle format)
├── model/             # Trained model checkpoints
├── scripts/           # Helper scripts for preprocessing & evaluation
├── README.md          # Documentation
└── requirements.txt   # Dependencies
```

---

## **⚙️ Installation & Setup**
### **Step 1: Clone the Repositories**
Clone both the original FactKG repository and this extended implementation:
```bash
git clone https://github.com/jiho283/FactKG.git
git clone https://github.com/24001998/FactKG.git
```
Move into the main project directory:
```bash
cd FactKG
```

### **Step 2: Install Dependencies**
Ensure you have Python 3.8+ installed, then install all required dependencies:
```bash
pip install -r requirements.txt
```
Additionally, install necessary libraries if they are not already included:
```bash
pip install torch transformers tqdm pickle5 scikit-learn datasets pyyaml
```

---

## **🚀 Running the Models**
### **Step 1: Data Preprocessing**
Before training the models, process the dataset:
```bash
python data_preprocess.py --data_directory_path ./data_set/factkg_train.pickle --output_directory_path ./model/
```

### **Step 2: Train a BERT Model for Claim Classification**
Execute the following command to train the model:
```bash
python claim_only/bert_classification.py \
    --model_name bert-base-uncased \
    --exp_name bert_log \
    --train_data_path "./data_set/factkg_train.pickle" \
    --valid_data_path "./data_set/factkg_dev.pickle" \
    --scheduler linear \
    --batch_size 64 \
    --lr 3e-5 \
    --total_epoch 5
```

### **Step 3: Train the Evidence Retrieval Model**
Run the retrieval model for relation prediction:
```bash
python with_evidence/retrieve/model/relation_predict/main.py --mode train \
    --config with_evidence/retrieve/model/config/relation_predict_top3.yaml
```

### **Step 4: Running Inference for Fact Checking**
To predict the truthfulness of a claim using a trained model:
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.eval()

def predict_claim(claim):
    inputs = tokenizer(claim, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits).item()
    return "TRUE" if predicted_class == 1 else "FALSE"

print(predict_claim("Barack Obama was the 44th president of the United States."))
```

---

## **📊 Model Evaluation**
To evaluate model performance on the test set:
```bash
python with_evidence/classifier/baseline.py \
    --mode eval \
    --data_directory_path "./data_set" \
    --model_path "./with_evidence/classifier/trained_classifier.pth"
```

---

## **🛠 Notes & Recommendations**
- **Hardware Requirements:** It is recommended to run the training on a **GPU-powered machine** (e.g., NVIDIA CUDA).
- **Dataset Format:** Ensure all dataset files are in **Pickle (.pickle) format** before running the scripts.
- **Colab Support:** If running on **Google Colab**, update paths accordingly and mount Google Drive.
- **Logging & Debugging:** Use logging features in the scripts to track the training progress.

---

## **📜 License & Citation**
This project is based on **FactKG** and adheres to its original licensing terms. If you use this repository for research purposes, please cite the original authors accordingly.

📌 **FactKG Original Paper**: https://aclanthology.org/2023.acl-long.895.pdf

---

## **📢 Contributing**
Contributions are welcome! If you wish to improve the repository, submit a pull request with detailed changes.


---

## **📬 Contact & Support**
For inquiries, please reach out to 24001998@student.buid.ac.ae.  
If you encounter issues, open a **GitHub Issue** in this repository.


---


## **👥 Team Members**
Mohamed Alhashmi 24000162
Hamed Almarzooqi 24000630
Saleh Alharthi   24001998


**Happy Coding! 🚀**

