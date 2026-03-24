# Elder Connect - Fake Profile Detection System

## Overview
This project detects fake user profiles using two machine learning approaches:

1. BERT-based model for analyzing user messages/text
2. Traditional ML model trained on profile features

## Models
- **BERT Model**: NLP-based detection from messages
- **Profile Model**: Uses structured features (followers, activity, etc.)

## Tech Stack
- Python
- Scikit-learn
- Transformers (BERT)

## How to Run
```bash
pip install -r requirements.txt
python train_model.py
