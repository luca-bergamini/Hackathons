"""RAG module - Sales Advisor usando Bedrock e vector store per insights FarmaVita.

Dovete implementare:
1. Preparazione documenti (riassunti per negozio, trend, pattern)
2. Creazione vector store (FAISS, ChromaDB, o altro)
3. Embedding con Bedrock Titan o altro modello
4. Chain RAG per rispondere a domande sulle vendite
5. Integrazione con l'endpoint /explain dell'API

Modelli Bedrock suggeriti (eu-west-1):
    - LLM: anthropic.claude-3-sonnet-20240229-v1:0
    - Embeddings: amazon.titan-embed-text-v1
"""

import logging
import os
import re
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

VECTOR_STORE_DIR = Path("data/faiss_index")

def prepare_store_documents(df: pd.DataFrame) -> list[Document]:
    """Crea un documento testuale per ogni store con statistiche strutturate e leggibili."""
    logger.info("Preparazione documenti per il RAG...")

    if "is_train" in df.columns:
        train_df = df[df["is_train"] == 1].copy()
    else:
        train_df = df.copy()

    if "store_id" not in train_df.columns:
        logger.warning("Colonna store_id non trovata.")
        return []

    # Aggiunge nome del giorno della settimana se non presente
    if "date" in train_df.columns and "day_of_week" not in train_df.columns:
        train_df["date"] = pd.to_datetime(train_df["date"], errors="coerce")
        train_df["day_of_week_name"] = train_df["date"].dt.day_name()
    elif "day_of_week" in train_df.columns:
        day_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
                   4: "Friday", 5: "Saturday", 6: "Sunday"}
        train_df["day_of_week_name"] = train_df["day_of_week"].map(day_map)

    # Solo giorni aperti per le statistiche di vendita
    open_df = train_df[train_df["is_open"] == 1] if "is_open" in train_df.columns else train_df

    docs = []
    for store_id, store_df in train_df.groupby("store_id"):
        store_id = int(store_id)
        open_store = open_df[open_df["store_id"] == store_id]
        lines = [f"FarmaVita Store ID {store_id}."]

        # --- Info statiche ---
        for col, label in [
            ("description_type", "Store Type"),
            ("description_assortment", "Assortment"),
            ("state", "State"),
        ]:
            if col in store_df.columns:
                val = store_df[col].mode()
                if len(val) > 0 and pd.notna(val.iloc[0]):
                    lines.append(f"{label}: {val.iloc[0]}.")

        if "distance_meters" in store_df.columns:
            dist = store_df["distance_meters"].median()
            if pd.notna(dist):
                lines.append(f"Competitor distance: {dist:.0f} meters.")

        if "open_since_year" in store_df.columns:
            yr = store_df["open_since_year"].mode()
            if len(yr) > 0 and pd.notna(yr.iloc[0]):
                lines.append(f"Open since: {int(yr.iloc[0])}.")

        # --- Statistiche vendite ---
        if "sales" in open_store.columns and len(open_store) > 0:
            avg_sales = open_store["sales"].mean()
            lines.append(f"Average daily sales (open days): {avg_sales:.0f}.")

            # Vendite per giorno della settimana
            if "day_of_week_name" in open_store.columns:
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                dow = open_store.groupby("day_of_week_name")["sales"].mean().round(0)
                dow_parts = [f"{d}: {dow[d]:.0f}" for d in day_order if d in dow.index]
                if dow_parts:
                    lines.append(f"Average sales by day of week — {', '.join(dow_parts)}.")
                    min_day = min([(d, dow[d]) for d in day_order if d in dow.index], key=lambda x: x[1])
                    max_day = max([(d, dow[d]) for d in day_order if d in dow.index], key=lambda x: x[1])
                    lines.append(f"Lowest sales day: {min_day[0]} ({min_day[1]:.0f}). Highest: {max_day[0]} ({max_day[1]:.0f}).")

            # Vendite per mese
            if "month" in open_store.columns:
                month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
                mo = open_store.groupby("month")["sales"].mean().round(0)
                mo_parts = [f"{month_names[int(m)]}: {v:.0f}" for m, v in mo.items() if pd.notna(v)]
                if mo_parts:
                    lines.append(f"Average sales by month — {', '.join(mo_parts)}.")

        # --- Promozioni ---
        if "promo" in store_df.columns:
            promo_rate = store_df["promo"].mean()
            if pd.notna(promo_rate):
                lines.append(f"Promotion active {promo_rate*100:.0f}% of days.")
        if "has_promo2" in store_df.columns:
            p2 = store_df["has_promo2"].mean()
            if pd.notna(p2):
                lines.append(f"Has Promo2 scheme: {'yes' if p2 > 0.5 else 'no'}.")

        # --- Meteo ---
        for col, label in [("temperature_avg", "Avg temperature"), ("precipitation_mm", "Avg precipitation mm")]:
            if col in store_df.columns:
                val = store_df[col].mean()
                if pd.notna(val):
                    lines.append(f"{label}: {val:.1f}.")

        if "weather_event" in store_df.columns:
            weather = store_df["weather_event"].mode()
            if len(weather) > 0 and pd.notna(weather.iloc[0]):
                lines.append(f"Most common weather: {weather.iloc[0]}.")

        # --- Macro indicatori ---
        for col, label in [("gdp_index", "GDP index"), ("unemployment_rate", "Unemployment rate"),
                            ("trend_index", "Google Trend index")]:
            if col in store_df.columns:
                val = store_df[col].mean()
                if pd.notna(val):
                    lines.append(f"{label}: {val:.2f}.")

        if "has_local_event" in store_df.columns:
            ev = store_df["has_local_event"].mean()
            if pd.notna(ev):
                lines.append(f"Local events on {ev*100:.0f}% of days.")

        doc = Document(
            page_content=" ".join(lines),
            metadata={"store_id": store_id}
        )
        docs.append(doc)

    logger.info(f"Creati {len(docs)} documenti.")
    return docs



def build_vector_store(docs: list[Document]):
    """Embedda i documenti e crea un index FAISS locale."""
    logger.info("Creazione vector store in corso...")
    
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name=os.environ.get("AWS_REGION", "eu-west-1")
    )
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTOR_STORE_DIR))
    logger.info(f"Vector store salvato in {VECTOR_STORE_DIR}")
    return vectorstore

def get_vector_store():
    """Carica il vector store salvato localmente."""
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name=os.environ.get("AWS_REGION", "eu-west-1")
    )
    if VECTOR_STORE_DIR.exists() and (VECTOR_STORE_DIR / "index.faiss").exists():
        return FAISS.load_local(str(VECTOR_STORE_DIR), embeddings, allow_dangerous_deserialization=True)
    else:
        logger.warning("Vector store non trovato. Verrà inizializzato come vuoto.")
        return None

def query_sales_advisor(question: str) -> dict:
    """Risponde a domande sulle vendite usando ricerca semantica sul vector store."""
    vectorstore = get_vector_store()
    if vectorstore is None:
        return {"answer": "Il Vector Store non è stato ancora inizializzato.", "sources": []}

    # Ricerca semantica pura — il FAISS troverà i documenti più rilevanti
    
    # Estrai store_id dalla domanda se presente (es. "store 42", "negozio 42", "id 42")
    mentioned_store_id = None
    match = re.search(r'\b(?:store|negozio|id|farmacia)\s*(?:id\s*)?#?(\d+)\b', question, re.IGNORECASE)
    if match:
        mentioned_store_id = int(match.group(1))
        logger.info(f"Store ID estratto dalla domanda: {mentioned_store_id}")
    
    # Recupera documenti semanticamente rilevanti
    docs = vectorstore.similarity_search(question, k=5)
    
    # Se abbiamo uno store ID specifico, assicuriamoci che il suo documento sia incluso
    if mentioned_store_id is not None:
        already_included = any(d.metadata.get("store_id") == mentioned_store_id for d in docs)
        if not already_included:
            # Cerca direttamente nel docstore
            docstore = vectorstore.docstore
            index_to_docstore_id = vectorstore.index_to_docstore_id
            for doc_id in index_to_docstore_id.values():
                doc = docstore.search(doc_id)
                if doc and doc.metadata.get("store_id") == mentioned_store_id:
                    docs = [doc] + docs[:4]  # metti lo store richiesto in testa
                    logger.info(f"Documento store {mentioned_store_id} aggiunto direttamente dal docstore.")
                    break
    
    logger.info(f"Documenti recuperati: {len(docs)}")
    for d in docs:
        logger.info(f"  store_id={d.metadata.get('store_id')}: {d.page_content[:80]}")
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    system_prompt = (
        "Sei un Sales Advisor esperto per FarmaVita. Usa il seguente contesto "
        "relativo ai negozi per rispondere alla domanda dell'utente.\n"
        "Se non conosci la risposta in base al contesto, dì semplicemente che non lo sai.\n\n"
        f"Context:\n{context}"
    )
    
    llm = ChatBedrock(
        model_id="eu.amazon.nova-pro-v1:0",
        region_name=os.environ.get("AWS_REGION", "eu-west-1"),
        model_kwargs={"temperature": 0.0}
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    chain = prompt | llm
    response = chain.invoke({"input": question})
    
    return {
        "question": question,
        "answer": response.content if hasattr(response, "content") else str(response),
        "sources": [doc.page_content for doc in docs]
    }
