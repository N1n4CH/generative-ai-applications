# AI Career Recommendation System - Generative AI

This project implements a generative AI system that produces personalised career
recommendations for candidates entering or transitioning into the AI job market.
Given a resume or skill profile as input, the system matches the candidate against
five AI job market personas discovered in previous projects, then generates a
natural language recommendation explaining their fit - strong, partial or none -
and suggesting a development pathway where alignment is weak.

The generative model is GPT-2 (small, 117M parameters), fine-tuned on structured
prompt-response pairs constructed from 2,484 real resumes. This is the fifth
project in the AI Capstone series and directly continues the
analytical pipeline built across the full course - from job collection (Project 1),
salary analysis (Project 2), skill clustering (Project 3) and persona discovery
(Project 4) to generative career guidance (Project 5).

**Dataset:** Resume Dataset - Sneha Anbhawal, Kaggle (2,484 real resumes from
livecareer.com across 24 job categories)  
**Source:** https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

---

## How to Run the Project

**1. Clone the repository**

    git clone https://github.com/n1n4ch/generative-ai-applications.git
    cd generative-ai

**2. Install dependencies**

    pip install -r requirements.txt

**3. Download the dataset**

Download `Resume.csv` from Kaggle and place it in the project folder:
https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

**4. Open and run the notebook**

    jupyter notebook generative_model.ipynb

Run all cells in order via **Kernel → Restart & Run All**

> Model weights are not included in the repository. Running Task 3 in the
> notebook fine-tunes GPT-2 and generates the `gpt2_career_advisor/` directory
> locally. An internet connection is required on first run to download the
> pre-trained GPT-2 weights from HuggingFace (~500MB).

---

## Project Structure

| File | Description |
|------|-------------|
| `generative_model.ipynb` | Main notebook - data loading, preprocessing, model training, generation |
| `Resume.csv` | Input dataset - download from Kaggle (not included in repo) |
| `gpt2_career_advisor/` | Fine-tuned model weights - generated locally by running the notebook |
| `Generative_AI_Analysis_Report.pdf` | Written report with citations |
| `requirements.txt` | Python dependencies (generated via `pip freeze`) |

---

## Approach

**Data preparation** - All 2,484 resumes are retained without category
pre-filtering. Raw text is cleaned to remove URLs, HTML tags, contact
information and special characters. Skills are extracted using the 50-skill
taxonomy from Project 3. Fit level is determined by cosine similarity between
each resume's skill vector and the five archetype centroids from Project 4 -
strong (≥0.4), partial (≥0.2) or none (<0.2).

**Model** — GPT-2 (small) is fine-tuned for 3 epochs on structured
prompt-response pairs using cross-entropy loss, AdamW optimiser (lr=5e-5)
and a linear warmup schedule. Each training example encodes the candidate's
skills, best-matching archetype and fit level as a prompt, with a templated
recommendation paragraph as the target response.

**Inference** — At generation time, a structured prompt is built from the
candidate's extracted skills and similarity scores. GPT-2 generates a
personalised recommendation using nucleus sampling (top-p=0.9, temperature=0.7).
Post-processing strips any hallucinated tags from the output.

---

## Fit Scenarios

The system produces three distinct output types:

| Fit Level | Similarity Score | Output |
|-----------|-----------------|--------|
| **Strong** | ≥ 0.4 | Archetype match with skill confirmation and focus advice |
| **Partial** | 0.2 – 0.4 | Closest archetype with missing skills identified |
| **No fit** | < 0.2 | Development pathway toward foundational AI skills |

---

## The Five AI Job Market Archetypes

Derived from k-means clustering of DACH + European job postings in Projects 3 and 4:

| # | Archetype | Defining Skills |
|---|-----------|-----------------|
| 0 | Data & Analytics Generalist | python, sql, machine learning, agile, tableau |
| 1 | Cloud ML Engineer | python, aws, azure, gcp, docker, kubernetes |
| 2 | MLOps & GenAI Engineer | python, mlops, generative ai, kubernetes, llm |
| 3 | AI Automation & Integration | python, aws, rpa, generative ai, spark |
| 4 | Deep Learning & AI Research | python, deep learning, computer vision, pytorch, nlp |

---

## Data Bias and Responsible Use

This system is designed as a candidate-facing career orientation tool and must
not be used for hiring screening or applicant ranking. Fit scores are based on
a narrow 50-skill keyword taxonomy and cannot capture the full range of a
candidate's competencies. Archetypes were derived from UK and DACH job postings
and reflect those markets' hiring norms - candidates from different educational
backgrounds or non-English-speaking countries may be disadvantaged by terminology
mismatches. Under the EU AI Act, any deployment in a recruitment context requires
classification as a high-risk AI system and corresponding compliance measures.

---

## Requirements

Regenerate with:

    pip freeze > requirements.txt

