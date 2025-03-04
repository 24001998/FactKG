!python data_preprocess.py --data_directory_path /content/data_set/factkg_train.pickle --output_directory_path ../model/

from google.colab import drive
drive.mount('/content/drive')

!pip install torch transformers tqdm pickle5 scikit-learn

!pip install -r /content/drive/MyDrive/FactKG2/requirements.txt

!python /content/drive/MyDrive/FactKG2/claim_only/bert_classification.py --model_name bert-base-uncased --exp_name bert_log --train_data_path "/content/drive/MyDrive/FactKG2/data_set/factkg_train.pickle" --valid_data_path "/content/drive/MyDrive/FactKG2/data_set/factkg_dev.pickle" --scheduler linear --batch_size 64 --eval_batch_size 64 --total_epoch 5

!python /content/drive/MyDrive/FactKG2/claim_only/flan_xl_zeroshot.py --valid_data_path "/content/drive/MyDrive/FactKG2/data_set/factkg_test.pickle" --model_name google/flan-t5-xl

!python /content/drive/MyDrive/FactKG2/claim_only/bert_classification.py --model_name bert-base-uncased --exp_name bert_log --train_data_path "/content/drive/MyDrive/FactKG2/data_set/factkg_train.pickle" --valid_data_path "/content/drive/MyDrive/FactKG2/data_set/factkg_dev.pickle" --scheduler linear --batch_size 32 --lr 3e-5 --total_epoch 5

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the trained model and tokenizer
model_name = "bert-base-uncased"  # Change if using a different model
model_path = "/content/exp_bert_log/deberta-large_2025-03-02 10.52.30.log"  # Update this with your actual trained model path

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.eval()  # Set model to evaluation mode

# Function to predict claim truthfulness
def predict_claim(claim):
    inputs = tokenizer(claim, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits).item()
    confidence = torch.softmax(outputs.logits, dim=1)[0][predicted_class].item() * 100  # Confidence score

    if predicted_class == 1:
        print(f"✅ Claim is likely **TRUE** ({confidence:.2f}% confidence): {claim}")
    else:
        print(f"❌ Claim is likely **FALSE** ({confidence:.2f}% confidence): {claim}")

# Example claims to test
test_claims = [
    "Barack Obama was the 44th president of the United States.",  # True
    "The capital of France is Berlin.",  # False
    "Albert Einstein discovered gravity.",  # False
    "The Great Wall of China is visible from space."  # False
]

# Run predictions on test claims
for claim in test_claims:
    predict_claim(claim)

!python /content/drive/MyDrive/FactKG2/claim_only/bert_classification.py --model_name bert-base-uncased --exp_name bert_log --train_data_path "/content/drive/MyDrive/FactKG2/data_set/factkg_train.pickle" --valid_data_path "/content/drive/MyDrive/FactKG2/data_set/factkg_dev.pickle" --scheduler linear --batch_size 64 --lr 3e-5 --total_epoch 20

!pip install datasets transformers torch pyyaml

!python /content/drive/MyDrive/FactKG2/with_evidence/retrieve/data/data_preprocess.py --data_directory_path "/content/drive/MyDrive/FactKG2/data_set" --output_directory_path "/content/drive/MyDrive/FactKG2/with_evidence/test_model"

!python3 /content/drive/MyDrive/FactKG2/with_evidence/retrieve/model/relation_predict/main.py --mode train --config /content/drive/MyDrive/FactKG2/with_evidence/retrieve/model/config/relation_predict_top3.yaml

!python3 /content/drive/MyDrive/FactKG2/with_evidence/retrieve/model/relation_predict/main.py --mode eval --config /content/drive/MyDrive/FactKG2/with_evidence/retrieve/model/config/relation_predict_top3.yaml --model_path "/content/lightning_logs/version_0/checkpoints/epoch=9-step=17350.ckpt"

!python3 /content/drive/MyDrive/FactKG2/with_evidence/retrieve/model/hop_predict/main.py --mode train --config /content/drive/MyDrive/FactKG2/with_evidence/retrieve/model/config/hop_predict.yaml

!python3 /content/drive/MyDrive/FactKG2/with_evidence/retrieve/model/hop_predict/main.py \
    --mode eval \
    --config /content/drive/MyDrive/FactKG2/with_evidence/retrieve/model/config/hop_predict.yaml \
    --model_path /content/model.pth

!ls -lh /content/drive/MyDrive/FactKG2/with_evidence/retrieve/model/hop_predict

!python3 /content/drive/MyDrive/FactKG2/with_evidence/retrieve/model/relation_predict/main.py --mode train --config /content/drive/MyDrive/FactKG2/with_evidence/retrieve/model/config/relation_predict_top3.yaml

!python3 /content/drive/MyDrive/FactKG2/with_evidence/classifier/preprocess.py \
    --mode preprocess \
    --data_directory_path "/content/drive/MyDrive/FactKG2/data_set" \
    --kg_path "/content/drive/MyDrive/FactKG2/data_set/dbpedia_2015_undirected.pickle" \
    --output_directory_path "/content/drive/MyDrive/FactKG2/with_evidence/classifier/"

!python3 /content/drive/MyDrive/FactKG2/with_evidence/classifier/baseline.py \
    --mode preprocess \
    --data_directory_path "/content/drive/MyDrive/FactKG2/data_set" \
    --kg_path "/content/drive/MyDrive/FactKG2/data_set/dbpedia_2015_undirected.pickle" \
    --output_directory_path "/content/drive/MyDrive/FactKG2/with_evidence/classifier/"

!python3 /content/drive/MyDrive/FactKG2/with_evidence/classifier/baseline.py \
    --mode train \
    --data_directory_path "/content/drive/MyDrive/FactKG2/data_set" \
    --kg_path "/content/drive/MyDrive/FactKG2/data_set/dbpedia_2015_undirected.pickle" \
    --output_directory_path "/content/drive/MyDrive/FactKG2/with_evidence/classifier/"

!python3 /content/drive/MyDrive/FactKG2/with_evidence/classifier/baseline.py \
    --mode eval \
    --data_directory_path "/content/drive/MyDrive/FactKG2/data_set" \
    --kg_path "/content/drive/MyDrive/FactKG2/data_set/dbpedia_2015_undirected.pickle" \
    --output_directory_path "/content/drive/MyDrive/FactKG2/with_evidence/classifier/" \
    --model_path "/content/trained_model/model.pth"

!python3 /content/drive/MyDrive/FactKG2/with_evidence/classifier/baseline.py \
    --mode train \
    --data_directory_path "/content/drive/MyDrive/FactKG2/data_set" \
    --kg_path "/content/drive/MyDrive/FactKG2/data_set/dbpedia_2015_undirected.pickle" \
    --output_directory_path "/content/drive/MyDrive/FactKG2/with_evidence/classifier/"

import json
import pickle as pkl

# Load predictions
with open("/content/drive/MyDrive/FactKG2/with_evidence/classifier/predictions.json", "r") as f:
    predictions = json.load(f)

# Load ground truth labels
with open("/content/drive/MyDrive/FactKG2/data_set/factkg_test.pickle", "rb") as f:
    test_data = pkl.load(f)

# Extract true labels
correct = 0
total = 0
for claim, details in test_data.items():
    true_label = details["Label"]  # Assuming 'Label' is the correct field
    predicted_label = predictions.get(claim, -1)  # Default to -1 if not found

    if predicted_label == true_label:
        correct += 1
    total += 1

accuracy = correct / total * 100
print(f"✅ Model Accuracy: {accuracy:.2f}%")

import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the trained classifier model
model_path = "/content/drive/MyDrive/FactKG2/with_evidence/classifier/trained_classifier.pth"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.eval()

# Test claim
claim = "Barack Obama was born in Kenya."

# Tokenize
inputs = tokenizer(claim, return_tensors="pt", truncation=True, max_length=128, padding="max_length")

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()

# Interpret result
if prediction == 1:
    print("Claim is True.")
else:
    print("Claim is False.")

!pip install jq

cat /content/drive/MyDrive/FactKG2/with_evidence/classifier/predictions.json | jq

!python3 /content/drive/MyDrive/FactKG2/with_evidence/classifier/baseline.py \
    --data_directory_path "/content/drive/MyDrive/FactKG2/data_set" \
    --kg_path "/content/drive/MyDrive/FactKG2/data_set/dbpedia_2015_undirected.pickle" \
    --output_directory_path "/content/drive/MyDrive/FactKG2/with_evidence/classifier/" \
    --mode "train" \
    --model_path "/content/drive/MyDrive/FactKG2/with_evidence/classifier/model.pth" \
    --batch_size 8

!python3 /content/drive/MyDrive/FactKG2/with_evidence/classifier/baseline.py \
    --data_directory_path "/content/drive/MyDrive/FactKG2/data_set" \
    --kg_path "/content/drive/MyDrive/FactKG2/data_set/dbpedia_2015_undirected.pickle" \
    --output_directory_path "/content/drive/MyDrive/FactKG2/with_evidence/classifier/" \
    --mode "eval" \
    --model_path "/content/drive/MyDrive/FactKG2/with_evidence/classifier/model.pth" \
    --batch_size 8

cat /content/drive/MyDrive/FactKG2/with_evidence/classifier/predictions.json

import json
import pickle

# Load Predictions
predictions_path = "/content/drive/MyDrive/FactKG2/with_evidence/classifier/predictions.json"
with open(predictions_path, "r") as f:
    predictions = json.load(f)

# Load Ground Truth Labels
test_labels_path = "/content/drive/MyDrive/FactKG2/data_set/factkg_test.pickle"
with open(test_labels_path, "rb") as f:
    test_data = pickle.load(f)

# Convert test labels to a dictionary
ground_truth = {claim: data["Label"] for claim, data in test_data.items()}  # Adjust based on your dataset format

# Compare and calculate accuracy
correct = 0
total = len(predictions)

for claim, pred_label in predictions.items():
    true_label = ground_truth.get(claim, -1)  # Default to -1 if not found
    if true_label != -1 and pred_label == true_label:
        correct += 1

accuracy = (correct / total) * 100 if total > 0 else 0
print(f"✅ Model Accuracy: {accuracy:.2f}%")

!python3 /content/drive/MyDrive/FactKG2/with_evidence/classifier/baseline.py \
    --mode preprocess \
    --data_directory_path "/content/drive/MyDrive/FactKG2/data_set" \
    --kg_path "/content/drive/MyDrive/FactKG2/data_set/dbpedia_2015_undirected.pickle" \
    --output_directory_path "/content/drive/MyDrive/FactKG2/with_evidence/classifier"

!python3 /content/drive/MyDrive/FactKG2/with_evidence/classifier/baseline.py \
    --data_directory_path "/content/drive/MyDrive/FactKG2/data_set" \
    --kg_path "/content/drive/MyDrive/FactKG2/data_set/dbpedia_2015_undirected.pickle" \
    --output_directory_path "/content/drive/MyDrive/FactKG2/with_evidence/classifier/" \
    --mode "train" \
    --model_path "/content/drive/MyDrive/FactKG2/with_evidence/classifier/model.pth" \
    --batch_size 8

!python3 /content/drive/MyDrive/FactKG2/with_evidence/classifier/baseline.py \
    --data_directory_path "/content/drive/MyDrive/FactKG2/data_set" \
    --kg_path "/content/drive/MyDrive/FactKG2/data_set/dbpedia_2015_undirected.pickle" \
    --output_directory_path "/content/drive/MyDrive/FactKG2/with_evidence/classifier/" \
    --mode "eval" \
    --model_path "/content/drive/MyDrive/FactKG2/with_evidence/classifier/model.pth" \
    --batch_size 1

!python3 /content/drive/MyDrive/FactKG2/with_evidence/classifier/baseline.py \
    --mode eval \
    --data_directory_path "/content/drive/MyDrive/FactKG2/data_set" \
    --output_directory_path "/content/drive/MyDrive/FactKG2/with_evidence/classifier" \
    --model_path "/content/drive/MyDrive/FactKG2/with_evidence/classifier/model.pth"

import json
import pickle as pkl

# ✅ Define file paths
test_data_path = "/content/drive/MyDrive/FactKG2/data_set/factkg_test.pickle"
predictions_path = "/content/drive/MyDrive/FactKG2/with_evidence/classifier/predictions.json"

# ✅ Load ground truth labels
with open(test_data_path, "rb") as f:
    test_data = pkl.load(f)

ground_truth = {claim: bool(details.get("Label", [None])[0]) for claim, details in test_data.items()}

# ✅ Load model predictions
with open(predictions_path, "r") as f:
    predictions = json.load(f)

# ✅ Compute accuracy
correct = 0
total = len(ground_truth)

for claim, true_label in ground_truth.items():
    predicted_label = bool(predictions.get(claim, None))  # Convert prediction to Boolean

    if predicted_label == true_label:
        correct += 1

accuracy = (correct / total) * 100 if total > 0 else 0

# ✅ Display results
print(f"\n✅ Fixed Model Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")