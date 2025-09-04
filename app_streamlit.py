import os
import re
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import (
    RagTokenizer, RagRetriever, RagSequenceForGeneration,
    DPRQuestionEncoderTokenizer, DPRQuestionEncoder
)

# Import existing project functions 

from rag_project import (
    universal_load_general_text_data,
    prepare_dataset,
    standard_retrieve,
    graph_retrieve,
    heart_row_to_text,
    diabetes_row_to_text,
    adult_row_to_text,
    build_simple_knowledge_graph,   # <-- used to build G for GraphRAG
)

st.set_page_config(page_title="GraphRAG UI", layout="wide")
st.title("Knowledge Graphs with Generative AI — RAG vs GraphRAG")
st.caption("Per-dataset comparison: Standard RAG vs GraphRAG, with dataset-specific visualizations. Uses your existing Python functions.")

# -------------------- Dataset config --------------------
DATASETS = {
    "PIMA Diabetes (medical)": {
        "source": "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
        "is_url": True,
        "columns": None,
        "header_option": 0,
        "sep": ",",
        "row_to_text_func": diabetes_row_to_text,
        "default_query": "What are the risk factors for diabetes?"
    },
    "Adult Income (financial)": {
        "source": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "is_url": True,
        "columns": [
            "age","workclass","fnlwgt","education","education_num","marital_status",
            "occupation","relationship","race","sex","capital_gain","capital_loss",
            "hours_per_week","native_country","income"
        ],
        "header_option": None,
        "sep": ",",
        "row_to_text_func": adult_row_to_text,
        "default_query": "What are the characteristics of high income individuals?"
    },
    "UCI Heart Disease (Cleveland, medical)": {
        "source": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "is_url": True,
        "columns": [
            "age","sex","cp","trestbps","chol","fbs","restecg","thalach",
            "exang","oldpeak","slope","ca","thal","target"
        ],
        "header_option": None,
        "sep": ",",
        "row_to_text_func": heart_row_to_text,
        "default_query": "What are common symptoms of heart disease?"
    }
}

# Helpers 
def normalize_docs(docs, n_limit=None):
    """Normalize various outputs to list of {'title','text','score'(opt)}."""
    out = []

    def maybe_limit(seq):
        return seq if n_limit is None else seq[:n_limit]

    if isinstance(docs, list):
        for i, d in enumerate(maybe_limit(docs)):
            if isinstance(d, dict):
                out.append({
                    "title": d.get("title", f"doc_{i}"),
                    "text": d.get("text", str(d)),
                    "score": d.get("score", None)
                })
            else:
                out.append({"title": f"doc_{i}", "text": str(d), "score": None})
        return out

    try:
        if hasattr(docs, "to_dict"):
            dct = docs.to_dict()
            key = "text" if "text" in dct else next(iter(dct.keys()))
            N = len(dct[key])
            N = N if n_limit is None else min(N, n_limit)
            titles = dct.get("title", [f"doc_{i}" for i in range(N)])
            texts = dct.get("text", [""] * N)
            scores = dct.get("score", [None] * N)
            for i in range(N):
                out.append({"title": titles[i], "text": texts[i], "score": scores[i]})
            return out
    except Exception:
        pass

    try:
        for i, d in enumerate(maybe_limit(list(docs))):
            if isinstance(d, dict):
                out.append({
                    "title": d.get("title", f"doc_{i}"),
                    "text": d.get("text", str(d)),
                    "score": d.get("score", None)
                })
            else:
                out.append({"title": f"doc_{i}", "text": str(d), "score": None})
        return out
    except Exception:
        return [{"title": "doc_0", "text": str(docs), "score": None}]

def safe_count(series):
    return series.value_counts(dropna=False)

def clean_adult_df(df):
    # Strip whitespace from strings
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
    # coerce numbers
    numeric_cols = ["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_raw_dataframe(cfg):
    src, is_url, sep, header_option, columns = cfg["source"], cfg["is_url"], cfg["sep"], cfg["header_option"], cfg["columns"]
    if is_url:
        if header_option is None and columns is not None:
            df = pd.read_csv(src, header=None, names=columns, sep=sep, na_values=["?"," NA","NA","? "])
        else:
            df = pd.read_csv(src, header=header_option, sep=sep, na_values=["?"," NA","NA","? "])
    else:
        if header_option is None and columns is not None:
            df = pd.read_csv(src, header=None, names=columns, sep=sep, na_values=["?"," NA","NA","? "])
        else:
            df = pd.read_csv(src, header=header_option, sep=sep, na_values=["?"," NA","NA","? "])

    # dataset specific cleanup
    if "income" in df.columns:
        df = clean_adult_df(df)
    return df

# Visualizations per dataset
def viz_diabetes(df):
    figs = []

    # 1) Outcome distribution
    fig1, ax = plt.subplots(figsize=(4.2,3.2))
    safe_count(df["Outcome"]).plot(kind="bar", ax=ax, color=["#7cd1f9","#f9a37c"])
    ax.set_title("Outcome distribution"); ax.set_xlabel("Outcome"); ax.set_ylabel("Count")
    figs.append(fig1)

    # 2) Glucose histogram
    fig2, ax = plt.subplots(figsize=(4.2,3.2))
    df["Glucose"].dropna().plot(kind="hist", bins=25, ax=ax)
    ax.set_title("Glucose (mg/dL)"); figs.append(fig2)

    # 3) BMI histogram
    fig3, ax = plt.subplots(figsize=(4.2,3.2))
    df["BMI"].dropna().plot(kind="hist", bins=25, ax=ax)
    ax.set_title("BMI"); figs.append(fig3)

    # 4) Age histogram
    fig4, ax = plt.subplots(figsize=(4.2,3.2))
    df["Age"].dropna().plot(kind="hist", bins=25, ax=ax)
    ax.set_title("Age"); figs.append(fig4)

    # 5) Correlation heatmap (numeric)
    fig5, ax = plt.subplots(figsize=(4.6,3.6))
    numeric = df.select_dtypes(include=[np.number]).corr()
    im = ax.imshow(numeric, cmap="Blues")
    ax.set_title("Correlation (numeric)"); ax.set_xticks(range(len(numeric.columns)))
    ax.set_yticks(range(len(numeric.columns)))
    ax.set_xticklabels(numeric.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(numeric.columns, fontsize=7)
    fig5.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    figs.append(fig5)

    return figs

def viz_heart(df):
    figs = []

    # 1) Target distribution
    fig1, ax = plt.subplots(figsize=(4.2,3.2))
    safe_count(df["target"]).plot(kind="bar", ax=ax, color=["#7cd1f9","#f9a37c","#fcd34d","#cbd5e1","#86efac"])
    ax.set_title("Diagnosis (0=healthy, higher=severity)"); ax.set_xlabel("target"); ax.set_ylabel("Count")
    figs.append(fig1)

    # 2) Age histogram
    fig2, ax = plt.subplots(figsize=(4.2,3.2))
    df["age"].dropna().plot(kind="hist", bins=25, ax=ax)
    ax.set_title("Age"); figs.append(fig2)

    # 3) Cholesterol histogram
    fig3, ax = plt.subplots(figsize=(4.2,3.2))
    df["chol"].dropna().plot(kind="hist", bins=25, ax=ax)
    ax.set_title("Cholesterol (mg/dl)"); figs.append(fig3)

    # 4) Max heart rate (thalach)
    fig4, ax = plt.subplots(figsize=(4.2,3.2))
    df["thalach"].dropna().plot(kind="hist", bins=25, ax=ax)
    ax.set_title("Max Heart Rate"); figs.append(fig4)

    # 5) ST depression vs target (box)
    if "oldpeak" in df.columns:
        fig5, ax = plt.subplots(figsize=(4.2,3.2))
        df.boxplot(column="oldpeak", by="target", ax=ax)
        ax.set_title("ST Depression by target"); ax.set_ylabel("oldpeak"); ax.figure.suptitle("")
        figs.append(fig5)

    return figs

def viz_adult(df):
    figs = []

    # 1) Income distribution
    fig1, ax = plt.subplots(figsize=(4.2,3.2))
    safe_count(df["income"]).plot(kind="bar", ax=ax, color=["#7cd1f9","#f9a37c"])
    ax.set_title("Income distribution"); ax.set_xlabel("income"); ax.set_ylabel("Count")
    figs.append(fig1)

    # 2) Age histogram
    fig2, ax = plt.subplots(figsize=(4.2,3.2))
    df["age"].dropna().plot(kind="hist", bins=25, ax=ax)
    ax.set_title("Age"); figs.append(fig2)

    # 3) Hours/week histogram
    fig3, ax = plt.subplots(figsize=(4.2,3.2))
    df["hours_per_week"].dropna().plot(kind="hist", bins=25, ax=ax)
    ax.set_title("Hours per week"); figs.append(fig3)

    # 4) Workclass top categories
    fig4, ax = plt.subplots(figsize=(4.2,3.2))
    safe_count(df["workclass"]).head(10).plot(kind="bar", ax=ax)
    ax.set_title("Top Workclass categories"); figs.append(fig4)

    # 5) Education_num distribution
    fig5, ax = plt.subplots(figsize=(4.2,3.2))
    df["education_num"].dropna().plot(kind="hist", bins=25, ax=ax)
    ax.set_title("Education level (num)"); figs.append(fig5)

    return figs

def render_visualizations(df, ds_key):
    st.subheader("📊 Dataset Visualizations")
    if "Diabetes" in ds_key:
        figs = viz_diabetes(df)
    elif "Heart" in ds_key:
        figs = viz_heart(df)
    else:
        figs = viz_adult(df)

    # Layout as 3-column grid (at least 5 charts)
    cols = st.columns(3)
    for i, fig in enumerate(figs):
        with cols[i % 3]:
            st.pyplot(fig)

# Sidebar 
with st.sidebar:
    st.header("Settings")
    ds_name = st.selectbox("Dataset", list(DATASETS.keys()), index=0)
    cfg = DATASETS[ds_name]
    n_docs = st.slider("Top-k documents", 1, 10, 3)
    run_generation = st.checkbox("Generate answer (RAG model)", value=False,
                                 help="Loads ~2GB weights on first run.")

#                Cache heavy components 
@st.cache_resource(show_spinner=True)
def load_models():
    #Arg-free: safe to cache.

    rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    qe_tok = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    qe = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").eval()
    return rag_tokenizer, qe_tok, qe

@st.cache_resource(show_spinner=True)
def build_index_and_retriever(_titles, _passages, _dataset_path, _index_path):
    
    # Underscored args tell Streamlit not to hash large/unhashable objects.
    # Builds dataset + FAISS index with your function and returns retriever.
    
    prepare_dataset(_titles, _passages, _dataset_path, _index_path)
    dataset = Dataset.load_from_disk(_dataset_path)
    retriever = RagRetriever.from_pretrained(
        "facebook/rag-sequence-nq",
        index_name="custom",
        passages_path=_dataset_path,
        index_path=_index_path,
    )
    return dataset, retriever

@st.cache_resource(show_spinner=True)
def load_generation_model(_retriever):
    # Underscored retriever avoids UnhashableParamError.
    model = RagSequenceForGeneration.from_pretrained(
        "facebook/rag-sequence-nq",
        retriever=_retriever
    ).eval()
    return model

# Load raw dataframe for visualizations 
with st.spinner("Loading raw dataset for visualizations…"):
    df_raw = load_raw_dataframe(cfg)
st.write("### Dataset preview")
st.dataframe(df_raw.head(), use_container_width=True)
render_visualizations(df_raw, ds_name)

# Load passages for retrieval 
with st.spinner("Converting dataset to passages for retrieval…"):
    titles, passages = universal_load_general_text_data(
        cfg["source"],
        is_url=cfg["is_url"],
        row_to_text_func=cfg["row_to_text_func"],
        column_names=cfg["columns"],
        header_option=cfg["header_option"],
        sep=cfg["sep"]
    )

st.write(f"**Passages loaded for retrieval:** {len(passages)}")
if len(passages) == 0:
    st.error("No passages were created from this dataset. Check mapping function and data source.")
    st.stop()

with st.expander("Preview a few passages", expanded=False):
    for i, (t, p) in enumerate(zip(titles[:5], passages[:5]), 1):
        st.markdown(f"**{t}** — {p}")

# Build dataset index + retriever (cached)
dataset_path = os.path.abspath("./st_cache_dataset")
index_path = os.path.join(dataset_path, "faiss_index")
with st.spinner("Building FAISS index & retriever (first run may take a bit)…"):
    dataset, retriever = build_index_and_retriever(titles, passages, dataset_path, index_path)

# Load encoders / (optional) generator
rag_tokenizer, question_encoder_tokenizer, question_encoder = load_models()
model = None
if run_generation:
    with st.spinner("Loading generation model…"):
        model = load_generation_model(retriever)

# Build a knowledge graph for GraphRAG (internal only; not displayed)
try:
    G = build_simple_knowledge_graph(passages, titles)
except Exception:
    G = None

# Query UI
default_q = cfg["default_query"]
query = st.text_input("Enter your query", value=default_q, help="Try something specific to the dataset.")
go = st.button("Run RAG vs GraphRAG comparison for this dataset")

if go:
    #  Standard RAG 
    st.subheader("🔹 Standard RAG — Top documents")
    rag_raw = standard_retrieve(
        query,
        retriever,
        question_encoder,
        question_encoder_tokenizer,
        dataset,
        n_docs
    )
    rag_docs = normalize_docs(rag_raw, n_limit=n_docs)
    if not rag_docs:
        st.info("No RAG docs found.")
    else:
        for i, d in enumerate(rag_docs, 1):
            st.markdown(f"**{i}.** {d['text'][:500]}{'...' if len(d['text'])>500 else ''}")

    #  GraphRAG 
    st.subheader("🔸 GraphRAG — Top documents")
    # Ensure G exists; if not, try to rebuild once
    if G is None:
        try:
            G = build_simple_knowledge_graph(passages, titles)
        except Exception:
            st.warning("Could not build knowledge graph; GraphRAG may be degraded.")
    gr_raw = graph_retrieve(
        query,
        G,                          # pass the graph
        dataset,
        question_encoder,
        question_encoder_tokenizer,
        retriever,
        n_docs
    )
    graphrag_docs = normalize_docs(gr_raw, n_limit=n_docs)
    if not graphrag_docs:
        st.info("No GraphRAG docs found.")
    else:
        for i, d in enumerate(graphrag_docs, 1):
            st.markdown(f"**{i}.** {d['text'][:500]}{'...' if len(d['text'])>500 else ''}")

    #  Per-dataset comparison chart 
    rag_avg = np.mean([d.get("score", 0) or 0 for d in rag_docs]) if rag_docs else 0.0
    gr_avg  = np.mean([d.get("score", 0) or 0 for d in graphrag_docs]) if graphrag_docs else 0.0

    st.subheader("📊 RAG vs GraphRAG — average top-k score (this dataset)")
    fig, ax = plt.subplots(figsize=(5.2,3.6))
    ax.bar(["RAG", "GraphRAG"], [rag_avg, gr_avg], color=["#60a5fa","#f97316"])
    ax.set_ylabel("Average retrieval score"); ax.set_ylim(0, max(0.01, rag_avg, gr_avg) * 1.25)
    for i, v in enumerate([rag_avg, gr_avg]):
        ax.text(i, v + (ax.get_ylim()[1]*0.02), f"{v:.3f}", ha="center", fontsize=10)
    st.pyplot(fig)

    #  Optional: Generated answer using RAG models
    if run_generation and model is not None:
        st.subheader("🧠 Generated Answer (RAG)")
        try:
            inputs = rag_tokenizer(query, return_tensors="pt")
            inputs.pop("token_type_ids", None)  # RAG models don't use token_type_ids
            with st.spinner("Generating…"):
                out = model.generate(**inputs, num_beams=4, min_length=10, max_length=80)
            answer = rag_tokenizer.batch_decode(out, skip_special_tokens=True)[0]
            st.success(answer)
        except Exception as e:
            st.error(f"Generation error: {e}")
else:
    st.info("Enter a query and click **Run RAG vs GraphRAG comparison for this dataset**.")
