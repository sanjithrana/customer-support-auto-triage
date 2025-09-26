# customer-support-auto-triage
# Customer Support Ticket Auto-Triage

An **AI/ML project** that classifies incoming customer support tickets into categories such as:

- Bug Report  
- Feature Request  
- Technical Issue  
- Billing Inquiry  
- Account Management  

The system uses **Natural Language Processing (NLP)** and a **machine learning classifier** (TF-IDF + Logistic Regression) and exposes a **REST API** for real-time predictions using FastAPI.

---

## 📂 Project Structure

customer-support-auto-triage/
├─ data/
│ └─ tickets.csv # Dataset (real or synthetic)
├─ models/
│ └─ ticket_model.joblib # Saved trained model
├─ src/
│ ├─ preprocess.py # Text cleaning & preprocessing
│ ├─ train.py # Train and save model
│ ├─ evaluate.py # Evaluate model performance
│ ├─ api.py # FastAPI server with /predict endpoint
│ └─ predict.py # Client script to test API
├─ requirements.txt # Dependencies
├─ README.md # Project documentation

yaml
Copy code

---

## ⚙️ Setup

1. **Clone repo & create virtual environment**
   ```bash
   git clone <your-repo-url>
   cd customer-support-auto-triage
   python -m venv my_env
   source my_env/bin/activate    # Linux/Mac
   my_env\Scripts\activate       # Windows
Install dependencies

bash
Copy code
pip install -r requirements.txt
Prepare dataset

Place your dataset in data/tickets.csv.

Required columns:

Ticket_ID

Subject

Description

Category (target label)

Priority

Timestamp

If you don’t have real data, use the included synthetic dataset.

🏋️ Training
Train the model and save it to models/ticket_model.joblib:

bash
Copy code
python src/train.py --data_path data/tickets.csv --label_col Category
📊 Evaluation
Evaluate performance and measure latency:

bash
Copy code
python src/evaluate.py --model_path models/ticket_model.joblib --data_path data/tickets.csv
Expected metrics:

Accuracy

Precision & Recall

F1-Score

Latency (ms/sample)

🌐 Run API
Start FastAPI server:

bash
Copy code
python src/api.py
#or 
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000

Server runs at:

Root → http://127.0.0.1:8000/

Docs → http://127.0.0.1:8000/docs (Swagger UI)

🤖 Test Predictions
Option 1: Swagger UI
Open http://127.0.0.1:8000/docs, expand /predict, click Try it out, and enter:

json
Copy code
{
  "Subject": "App crashes on login",
  "Description": "The app closes every time I try to log in"
}
Option 2: Curl
bash
Copy code
curl -X POST "http://127.0.0.1:8000/predict" \
 -H "Content-Type: application/json" \
 -d "{\"Subject\":\"App crashes on login\",\"Description\":\"The app closes every time I try to log in\"}"
Option 3: Predict Script
Run the included client script to test multiple tickets at once:

bash
Copy code
python src/predict.py
Example output:

json
Copy code
--- Test Case 1 ---
Input: {"Subject": "App crashes on login", "Description": "The app closes every time I try to log in"}
Output: {"category": "Bug Report", "confidence": 0.9231, "latency_ms": 3.74}
📦 Submission Guidelines
Push code to a Git repository with clear commit messages.

Include:

src/ folder with scripts

models/ folder with trained model (ticket_model.joblib)

README.md with setup & usage instructions

Dataset (or instructions to generate synthetic data)

Send the repo link & model files as per assignment instructions.

🚀 Future Improvements
Fine-tune transformer models (e.g., BERT, DistilBERT) for better accuracy.

Add Dockerfile for containerized deployment.

Integrate logging, monitoring, and error handling.

Implement batch predictions for bulk ticket processing.

✨ Author
Developed by sanjith chilupuri as part of AI/ML Customer Support Auto-Triage project.

yaml
Copy code

---

👉 Do you want me to also generate a **sample `requirements.txt`** alongside this README (so your repo is fully plug-and-play)?







Ask ChatGPT
