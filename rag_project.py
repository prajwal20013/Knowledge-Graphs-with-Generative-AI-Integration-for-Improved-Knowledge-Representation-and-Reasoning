import torch
from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizer,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    RagTokenizer, RagRetriever, RagSequenceForGeneration
)
from datasets import Dataset
import pandas as pd
import numpy as np
import requests
from io import StringIO
import os
import networkx as nx
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import defaultdict

def universal_load_general_text_data(source, is_url=False, row_to_text_func=None, column_names=None, header_option=None, sep=None):
    if is_url:
        response = requests.get(source)
        if not response.ok:
            print("Download error:", response.status_code, response.text[:200])
            return [], []
        data = StringIO(response.text)
        df = pd.read_csv(data, names=column_names, header=header_option, sep=sep, skipinitialspace=True) if column_names else pd.read_csv(data, header=header_option, sep=sep, skipinitialspace=True)
    else:
        df = pd.read_csv(source, names=column_names, header=header_option, sep=sep, skipinitialspace=True) if column_names else pd.read_csv(source, header=header_option, sep=sep, skipinitialspace=True)
    df = df.replace("?", pd.NA)
    print("Dataframe columns:", df.columns.tolist())
    if 'text' in df.columns:
        passages = df['text'].dropna().astype(str).tolist()
        titles = df['title'].astype(str).tolist() if 'title' in df.columns else [f"doc_{i}" for i in range(len(passages))]
    else:
        if row_to_text_func:
            passages = []
            for _, row in df.iterrows():
                passage = row_to_text_func(row)
                if passage is not None and isinstance(passage, str) and passage.strip() != "":
                    passages.append(passage)
        else:
            passages = [" ".join([str(x) for x in row.values]) for _, row in df.iterrows()]
        titles = [f"doc_{i}" for i in range(len(passages))]
    return titles, passages

def build_simple_knowledge_graph(passages, titles):
    G = nx.Graph()
    key_phrases = [
        "diabetes", "glucose", "blood pressure", "bmi", "pedigree", "pregnancies", "malignant", "benign",
        "education", "workclass", "income", "occupation", "hours per week", "capital gain", "capital loss",
        "heart disease", "angina", "cholesterol", "resting bp", "fasting sugar", "max heart rate", "st depression",
        "cp", "trestbps", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "sex"
    ]
    for title, passage in zip(titles, passages):
        passage_lower = passage.lower()
        for phrase in key_phrases:
            if phrase in passage_lower:
                G.add_node(phrase)
                G.add_edge(title, phrase)
        words = re.findall(r'\w+', passage_lower)
        entities = [w for w in words if w not in ENGLISH_STOP_WORDS]
        for entity in set(entities):
            G.add_node(entity)
            G.add_edge(title, entity)
    return G

def prepare_dataset(titles, passages, dataset_path, index_path):
    if os.path.exists(dataset_path) and os.path.exists(index_path):
        print("Using cached dataset and FAISS index.")
        return

    dataset = Dataset.from_dict({"title": titles, "text": passages})
    tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    encoder.eval()

    def encode(batch):
        inputs = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256, return_tensors="pt")
        with torch.no_grad():
            embeddings = encoder(**inputs).pooler_output
        return {"embeddings": [e.numpy() for e in embeddings.cpu()]}

    print("Encoding passages and building FAISS index...")
    dataset = dataset.map(encode, batched=True, batch_size=16)
    dataset.save_to_disk(dataset_path)
    dataset = Dataset.load_from_disk(dataset_path)
    dataset.add_faiss_index(column="embeddings")
    dataset.get_index("embeddings").save(index_path)
    print("Dataset and index saved.")

def get_question_hidden_states(query, question_encoder, question_encoder_tokenizer):
    inputs = question_encoder_tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        question_hidden_states = question_encoder(**inputs).pooler_output
    return question_hidden_states.cpu().numpy()

def safe_index(idx):
    try:
        return int(idx.flatten()[0])
    except Exception:
        return int(idx)

# --- utilities (put near your imports) ---
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))


def _pick_ids_from_get_top_docs(out1, out2):
    
    # HF RAG's index.get_top_docs(q, k) will return
    #   (scores, ids) or  (ids, vectors)  according to index backend.
    # We robustly pick IDs by preferring the integer-typed output.
    
    a, b = np.asarray(out1), np.asarray(out2)
    # prefer the integer array as ids
    if np.issubdtype(a.dtype, np.integer):
        ids = a
    elif np.issubdtype(b.dtype, np.integer):
        ids = b
    else:
        # fallback: choose the one with smaller absolute values unlikely to be embeddings
        ids = a if a.ndim >= b.ndim else b
    # squeeze possible leading batch dim
    if ids.ndim > 1:
        ids = ids[0]
    return ids


def standard_retrieve(query, retriever, question_encoder, question_encoder_tokenizer, dataset, n_docs=5):
    
    # Returns a list of dicts: [{'title','text','score'}, ...]
    # score = cosine(question_vector, doc['embeddings'])
    
    # 1) encode query to DPR vector (1, D)
    inputs = question_encoder_tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        q_hidden = question_encoder(**inputs).pooler_output.cpu().numpy()  # (1, D)
    q_vec = q_hidden[0]

    # 2) retrieve top doc ids
    out1, out2 = retriever.index.get_top_docs(q_hidden, n_docs)
    ids = _pick_ids_from_get_top_docs(out1, out2)
    ids = list(map(int, list(ids)))[:n_docs]

    # 3) build results with cosine against stored embeddings
    results = []
    for idx in ids:
        row = dataset[int(idx)]
        emb = np.array(row.get("embeddings", []), dtype=np.float32)
        score = _cosine(q_vec, emb) if emb.size else 0.0
        results.append({
            "title": row.get("title", f"doc_{int(idx)}"),
            "text": row.get("text", ""),
            "score": float(score),
        })

    # 4) sort & return
    results.sort(key=lambda d: d["score"], reverse=True)
    return results[:n_docs]

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")

def _tokenize(s: str):
    return [t.lower() for t in _TOKEN_RE.findall(str(s or ""))]

def _graph_overlap_score(G, title: str, query_tokens: set) -> float:
    # Proportion of the doc node's neighbors that appear in the query.
    try:
        if G is None or title not in G:
            return 0.0
        neigh = list(G.neighbors(title))
        if not neigh:
            return 0.0
        n_hit = sum(1 for n in neigh if str(n).lower() in query_tokens)
        return n_hit / max(1, len(neigh))
    except Exception:
        return 0.0


def graph_retrieve(query, G, dataset, question_encoder, question_encoder_tokenizer, retriever, n_docs=5, alpha=0.7):
    
    # GraphRAG retrieval that blends:
    #    final_score = alpha * cosine(query, doc_embedding) + (1 - alpha) * graph_overlap
    # Returns [{'title','text','score'}, ...] sorted by score desc.
    
    # 1) query vector
    inputs = question_encoder_tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        q_hidden = question_encoder(**inputs).pooler_output.cpu().numpy()
    q_vec = q_hidden[0]

    # 2) retrieve a larger candidate pool
    pool_k = max(n_docs * 4, 20)
    out1, out2 = retriever.index.get_top_docs(q_hidden, pool_k)
    ids = _pick_ids_from_get_top_docs(out1, out2)
    ids = list(map(int, list(ids)))

    # 3) precompute query tokens
    q_tokens = set(_tokenize(query))

    # 4) score candidates
    scored = []
    for idx in ids:
        row = dataset[int(idx)]
        title = row.get("title", f"doc_{int(idx)}")
        text = row.get("text", "")
        emb = np.array(row.get("embeddings", []), dtype=np.float32)

        cos = _cosine(q_vec, emb) if emb.size else 0.0
        g = _graph_overlap_score(G, title, q_tokens)
        final = alpha * cos + (1.0 - alpha) * g

        scored.append({"title": title, "text": text, "score": float(final)})

    # 5) sort & truncate
    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:n_docs]

def diabetes_row_to_text(row):
    try:
        outcome = "has diabetes" if int(row["Outcome"]) == 1 else "does not have diabetes"
        return (f"Patient is {row['Age']} years old, {row['Pregnancies']} pregnancies, "
                f"glucose {row['Glucose']}, blood pressure {row['BloodPressure']}, "
                f"BMI {row['BMI']}, pedigree function {row['DiabetesPedigreeFunction']}, and {outcome}.")
    except Exception as e:
        print("Row error (diabetes):", e)
        return None

def adult_row_to_text(row):
    try:
        result = "high income" if str(row["income"]).strip() == ">50K" else "low income"
        return (f"Person is {row['age']} years old, works as {row['occupation']}, "
                f"education {row['education']} ({row['education_num']} years), marital status {row['marital_status']}, "
                f"relationship {row['relationship']}, race {row['race']}, sex {row['sex']}, "
                f"hours per week {row['hours_per_week']}, native country {row['native_country']}, and has {result}.")
    except Exception as e:
        print("Row error (adult):", e)
        return None

def heart_row_to_text(row):
    try:
        cp_types = {"1.0": "typical angina", "2.0": "atypical angina", "3.0": "non-anginal pain", "4.0": "asymptomatic"}
        slope_types = {"1.0": "upsloping", "2.0": "flat", "3.0": "downsloping"}
        thal_types = {"3.0": "normal", "6.0": "fixed defect", "7.0": "reversable defect"}
        sex = "male" if str(row["sex"]) == "1.0" else "female"
        cp = cp_types.get(str(row["cp"]), "unknown chest pain")
        slope = slope_types.get(str(row["slope"]), "unknown slope")
        thal = thal_types.get(str(row["thal"]), "unknown thalassemia")
        outcome = "has heart disease" if str(row["target"]) != "0.0" else "no heart disease"
        return (f"Patient is {row['age']} years old {sex} with {cp}, resting BP {row['trestbps']} mmHg, "
                f"cholesterol {row['chol']} mg/dl, fasting sugar {row['fbs']}, ECG {row['restecg']}, "
                f"max HR {row['thalach']}, exercise angina {row['exang']}, ST depression {row['oldpeak']}, "
                f"slope {slope}, vessels colored {row['ca']}, thalassemia {thal}, "
                f"and {outcome}.")
    except Exception as e:
        print("Row error (heart):", e)
        return None

def main():
    print("Choose dataset:")
    print("1. Diabetes (Medical)")
    print("2. Adult Income (Financial)")
    print("3. Heart Disease (Medical, UCI Cleveland)")
    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "1":
        source = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
        is_url = True
        columns = None
        header_option = 0  # First row is header
        sep = ","
        row_to_text_func = diabetes_row_to_text
    elif choice == "2":
        source = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        is_url = True
        columns = [
            "age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
            "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
        ]
        header_option = None  # No header row
        sep = ","
        row_to_text_func = adult_row_to_text
    elif choice == "3":
        source = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        is_url = True
        columns = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal", "target"
        ]
        header_option = None  # No header in file
        sep = ","
        row_to_text_func = heart_row_to_text
    else:
        print("Invalid choice.")
        return

    titles, passages = universal_load_general_text_data(
        source, is_url=is_url,
        row_to_text_func=row_to_text_func,
        column_names=columns,
        header_option=header_option,
        sep=sep
    )

    print(f"Loaded {len(passages)} passages.")
    if len(passages) == 0:
        print("No passages loaded! Please check the dataset and mapping function.")
        return

    dataset_path = "./my_dataset"
    index_path = "./my_dataset/faiss_index"
    n_docs = 3

    prepare_dataset(titles, passages, dataset_path, index_path)
    dataset = Dataset.load_from_disk(dataset_path)
    G = build_simple_knowledge_graph(passages, titles)

    rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    retriever = RagRetriever.from_pretrained(
        "facebook/rag-sequence-nq",
        index_name="custom",
        passages_path=dataset_path,
        index_path=index_path,
    )
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)
    model.eval()
    question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    question_encoder.eval()

    query = input("\nEnter your query: ").strip()
    if not query:
        if choice == "1":
            query = "What are the risk factors for diabetes?"
        elif choice == "2":
            query = "What are the characteristics of high income individuals?"
        elif choice == "3":
            query = "What are the common symptoms of heart disease?"

    # Standard RAG retrieval
    standard_retrieve(query, retriever, question_encoder, question_encoder_tokenizer, dataset, n_docs)

    # GraphRAG retrieval
    graph_retrieve(query, G, dataset, question_encoder, question_encoder_tokenizer, retriever, n_docs)

    # Generate answer using the RAG model
    inputs = rag_tokenizer(query, return_tensors="pt")
    inputs.pop("token_type_ids", None)
    with torch.no_grad():
        generated = model.generate(**inputs, num_beams=4, min_length=10, max_length=50)
    answer = rag_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    print("\nGenerated Answer:", answer)

if __name__ == "__main__":
    main()