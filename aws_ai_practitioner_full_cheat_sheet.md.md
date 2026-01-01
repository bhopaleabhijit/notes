Below is a compact, exam-oriented cheat sheet covering **all core topics** from the AWS Certified AI Practitioner syllabus. Each point has a **one-line explanation** for fast revision.

---

## Core AI & ML Concepts

- **Artificial Intelligence (AI)**: Systems that perform tasks requiring human-like intelligence.
- **Machine Learning (ML)**: AI that learns patterns from data without explicit programming.
- **Deep Learning (DL)**: ML using multi-layer neural networks for complex patterns.
- **Generative AI**: Models that create new content like text, images, audio, or code.
- **Large Language Model (LLM)**: A text-focused generative model trained on massive datasets.
- **Foundation Model (FM)**: A large pre-trained model adaptable to many tasks.

---

## ML Learning Types

- **Supervised Learning**: Trains on labeled data to predict known outputs.
- **Unsupervised Learning**: Finds patterns in unlabeled data.
- **Reinforcement Learning**: Learns by trial and error using rewards.

---

## Common Algorithms

- **Linear Regression**: Predicts continuous numeric values using a straight-line relationship.
- **Logistic Regression**: Predicts categories or probabilities (classification).
- **Decision Tree**: Uses if-else rules to make predictions.
- **Random Forest**: Ensemble of decision trees to reduce overfitting.
- **K-Means**: Groups data into clusters without labels.

---

## Model Training Concepts

- **Feature Engineering**: Selecting and transforming input variables.
- **Feature Store**: Central repository for managing ML features.
- **Training**: Learning patterns from historical data.
- **Inference**: Using a trained model to make predictions.
- **Overfitting**: Model performs well on training data but poorly on new data.
- **Underfitting**: Model is too simple to capture patterns.

---

## Generative AI Techniques

- **Prompt Engineering**: Designing prompts to guide model outputs.
- **Prompt Optimization**: Tuning parameters like temperature or top-p.
- **Fine-Tuning**: Customizing a model with domain-specific data.
- **RAG (Retrieval Augmented Generation)**: Combining LLMs with private data sources.
- **Embeddings**: Numerical representations of text for similarity search.

---

## Amazon Bedrock

- **Amazon Bedrock**: Fully managed service for accessing foundation models via API.
- **Model Providers**: Anthropic, Amazon Titan, Meta, Cohere, etc.
- **Guardrails**: Controls for safety, filtering, and responsible AI.
- **Model Evaluation**: Compare FM outputs for quality and relevance.
- **Inference Parameters**: Temperature, top-k, top-p control randomness.

---

## Amazon SageMaker

- **SageMaker**: End-to-end ML platform for building, training, and deploying models.
- **SageMaker Studio**: IDE for ML development.
- **SageMaker Canvas**: No-code ML tool for business users.
- **SageMaker JumpStart**: Pre-built models and solutions.
- **SageMaker Autopilot**: Automatically builds and tunes ML models.
- **SageMaker Clarify**: Detects bias and explains model predictions.
- **SageMaker Feature Store**: Stores and shares ML features.
- **SageMaker Wrangler**: Data preparation and feature engineering tool.

---

## Human-in-the-Loop Services

- **SageMaker Ground Truth**: Labels training data using humans and automation.
- **Amazon A2I**: Adds human review to ML predictions.
- **Amazon Mechanical Turk**: Marketplace for human workforce tasks.

---

## Search & NLP Services

- **Amazon Kendra**: Enterprise search powered by ML.
- **Amazon Comprehend**: NLP service for entities, sentiment, and key phrases.
- **Amazon Comprehend Medical**: NLP for healthcare text.

---

## Speech & Vision Services

- **Amazon Transcribe**: Converts speech to text.
- **Amazon Transcribe Medical**: Speech-to-text for medical audio.
- **Amazon Polly**: Text-to-speech service.
- **Amazon Rekognition**: Image and video analysis.

---

## Responsible AI

- **Bias**: Systematic unfairness in ML predictions.
- **Explainability**: Ability to understand why a model made a decision.
- **Transparency**: Clear understanding of model behavior and limits.
- **Privacy**: Protecting sensitive data used in ML.

---

## Exam Decision Shortcuts

- **No code / business user** → SageMaker Canvas
- **LLMs via API / least ops** → Amazon Bedrock
- **Private docs + LLM** → RAG
- **Bias or explain** → SageMaker Clarify
- **Label training data** → Ground Truth
- **Human review of predictions** → A2I
- **Enterprise search** → Kendra

---

## Final One-Line Exam Tip

Choose the option with **least operational effort**, **most managed service**, and **clear AWS alignment**.

