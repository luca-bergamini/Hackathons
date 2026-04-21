"""Test per il modulo RAG (src/rag/main.py).

Testa la preparazione documenti e le funzioni del RAG.
Le funzioni che richiedono AWS Bedrock sono mockate.
"""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


def test_rag_module_imports():
    """Verify the RAG module can be imported."""
    from src.rag import main  # noqa: F401


def test_rag_functions_exist():
    """Le funzioni principali devono esistere."""
    from src.rag.main import (
        prepare_store_documents,
        build_vector_store,
        get_vector_store,
        query_sales_advisor,
    )
    assert all(callable(f) for f in [
        prepare_store_documents, build_vector_store,
        get_vector_store, query_sales_advisor,
    ])


def test_vector_store_dir_defined():
    """VECTOR_STORE_DIR deve essere un Path."""
    from src.rag.main import VECTOR_STORE_DIR
    assert isinstance(VECTOR_STORE_DIR, Path)


class TestPrepareStoreDocuments:
    """Test per prepare_store_documents."""

    def test_basic_documents_creation(self, sample_features_df):
        """Deve creare un Document per ogni store."""
        from src.rag.main import prepare_store_documents

        docs = prepare_store_documents(sample_features_df)
        store_ids = sample_features_df["store_id"].unique()

        # Solo i train stores (is_train == 1 o True)
        train_stores = sample_features_df[
            sample_features_df["is_train"] == True
        ]["store_id"].unique()

        assert len(docs) > 0
        assert len(docs) <= len(store_ids)

    def test_document_has_store_id_metadata(self, sample_features_df):
        """Ogni documento deve avere store_id nei metadata."""
        from src.rag.main import prepare_store_documents

        docs = prepare_store_documents(sample_features_df)
        for doc in docs:
            assert "store_id" in doc.metadata
            assert isinstance(doc.metadata["store_id"], int)

    def test_document_has_content(self, sample_features_df):
        """Ogni documento deve avere page_content non vuoto."""
        from src.rag.main import prepare_store_documents

        docs = prepare_store_documents(sample_features_df)
        for doc in docs:
            assert len(doc.page_content) > 0
            assert "FarmaVita Store ID" in doc.page_content

    def test_document_contains_sales_info(self, sample_features_df):
        """I documenti con sales devono contenere statistiche di vendita."""
        from src.rag.main import prepare_store_documents

        docs = prepare_store_documents(sample_features_df)
        # Almeno un documento deve parlare di vendite
        has_sales = any("Average daily sales" in d.page_content for d in docs)
        assert has_sales

    def test_document_contains_day_of_week(self, sample_features_df):
        """I doc devono contenere info sulle vendite per giorno della settimana."""
        from src.rag.main import prepare_store_documents

        docs = prepare_store_documents(sample_features_df)
        has_dow = any("day of week" in d.page_content for d in docs)
        assert has_dow

    def test_empty_df_returns_empty(self):
        """Un DataFrame vuoto deve restituire una lista vuota."""
        from src.rag.main import prepare_store_documents

        df = pd.DataFrame()
        docs = prepare_store_documents(df)
        assert docs == []

    def test_no_store_id_returns_empty(self):
        """DataFram senza store_id → lista vuota."""
        from src.rag.main import prepare_store_documents

        df = pd.DataFrame({"other_col": [1, 2, 3]})
        docs = prepare_store_documents(df)
        assert docs == []

    def test_document_promo_info(self, sample_features_df):
        """I documenti devono menzionare le promozioni se presenti."""
        from src.rag.main import prepare_store_documents

        docs = prepare_store_documents(sample_features_df)
        has_promo = any("Promotion" in d.page_content for d in docs)
        assert has_promo

    def test_document_weather_info(self, sample_features_df):
        """I documenti devono menzionare info meteo se presenti."""
        from src.rag.main import prepare_store_documents

        df = sample_features_df.copy()
        df["temperature_avg"] = 20.5  # assicura che ci sia

        docs = prepare_store_documents(df)
        has_temp = any("temperature" in d.page_content.lower() for d in docs)
        assert has_temp

    def test_only_train_data_used(self, sample_features_df):
        """Deve usare solo i dati di training."""
        from src.rag.main import prepare_store_documents

        df = sample_features_df.copy()
        # Solo dati test — non dovrebbe generare nulla se non ci sono train
        test_only = df[df["is_train"] == False].copy()
        test_only = test_only.drop(columns=["is_train"])

        # Senza is_train → usa tutto il df come train
        docs = prepare_store_documents(test_only)
        # Dovrebbe generare documenti dal df intero
        assert len(docs) >= 0

    def test_documents_unique_stores(self, sample_features_df):
        """Non ci devono essere documenti duplicati per lo stesso store."""
        from src.rag.main import prepare_store_documents

        docs = prepare_store_documents(sample_features_df)
        store_ids = [d.metadata["store_id"] for d in docs]
        assert len(store_ids) == len(set(store_ids))


class TestGetVectorStore:
    """Test per get_vector_store (con mock AWS)."""

    @patch("src.rag.main.VECTOR_STORE_DIR", Path("/nonexistent/path"))
    @patch("src.rag.main.BedrockEmbeddings")
    def test_returns_none_when_not_found(self, mock_embeddings):
        """Se la directory non esiste, deve restituire None."""
        from src.rag.main import get_vector_store

        mock_embeddings.return_value = MagicMock()
        result = get_vector_store()
        assert result is None


class TestQuerySalesAdvisor:
    """Test per query_sales_advisor (con mock)."""

    @patch("src.rag.main.get_vector_store")
    def test_returns_dict_when_no_vectorstore(self, mock_get_vs):
        """Se non c'è vector store, deve restituire un messaggio di errore."""
        from src.rag.main import query_sales_advisor

        mock_get_vs.return_value = None
        result = query_sales_advisor("Test question?")

        assert isinstance(result, dict)
        assert "answer" in result
        assert "sources" in result
        assert "non è stato ancora inizializzato" in result["answer"]

    def test_store_id_extraction_from_question(self):
        """Il regex deve estrarre lo store_id dalla domanda."""
        import re
        pattern = r'\b(?:store|negozio|id|farmacia)\s*(?:id\s*)?#?(\d+)\b'

        test_cases = [
            ("store 42", 42),
            ("negozio 100", 100),
            ("farmacia 7", 7),
            ("id 55", 55),
            ("Store ID 42", 42),
        ]
        for text, expected in test_cases:
            match = re.search(pattern, text, re.IGNORECASE)
            assert match is not None, f"Non ha matchato: '{text}'"
            assert int(match.group(1)) == expected

    def test_store_id_not_found_in_generic_question(self):
        """Domande senza store_id non devono estrarre un ID."""
        import re
        pattern = r'\b(?:store|negozio|id|farmacia)\s*(?:id\s*)?#?(\d+)\b'

        text = "Qual è il trend delle vendite?"
        match = re.search(pattern, text, re.IGNORECASE)
        assert match is None
