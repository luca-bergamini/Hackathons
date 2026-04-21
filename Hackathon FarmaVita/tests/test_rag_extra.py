"""Test aggiuntivi per RAG — copertura build_vector_store, get_vector_store, query_sales_advisor."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pandas as pd
import pytest


@patch("src.rag.main.BedrockEmbeddings")
@patch("src.rag.main.FAISS")
def test_build_vector_store(mock_faiss, mock_embeddings, sample_features_df, tmp_path):
    """build_vector_store deve creare la directory e salvarvi l'index."""
    from src.rag.main import prepare_store_documents, build_vector_store, VECTOR_STORE_DIR

    docs = prepare_store_documents(sample_features_df)
    mock_embeddings.return_value = MagicMock()

    mock_vs = MagicMock()
    mock_faiss.from_documents.return_value = mock_vs

    with patch("src.rag.main.VECTOR_STORE_DIR", tmp_path / "faiss_test"):
        result = build_vector_store(docs)

    mock_faiss.from_documents.assert_called_once()
    mock_vs.save_local.assert_called_once()


@patch("src.rag.main.BedrockEmbeddings")
@patch("src.rag.main.FAISS")
def test_get_vector_store_loads_existing(mock_faiss, mock_embeddings, tmp_path):
    """get_vector_store deve caricare l'index se esiste."""
    from src.rag.main import get_vector_store

    mock_embeddings.return_value = MagicMock()

    # Crea i file finti
    faiss_dir = tmp_path / "faiss_test"
    faiss_dir.mkdir()
    (faiss_dir / "index.faiss").touch()

    mock_faiss.load_local.return_value = MagicMock()

    with patch("src.rag.main.VECTOR_STORE_DIR", faiss_dir):
        result = get_vector_store()

    mock_faiss.load_local.assert_called_once()
    assert result is not None


@patch("src.rag.main.ChatBedrock")
@patch("src.rag.main.get_vector_store")
def test_query_sales_advisor_with_store(mock_get_vs, mock_chat):
    """query_sales_advisor deve restituire risposta e sources."""
    from src.rag.main import query_sales_advisor
    from langchain_core.documents import Document

    # Mock vector store
    mock_vs = MagicMock()
    mock_docs = [
        Document(page_content="FarmaVita Store ID 42. Average daily sales: 5000.", metadata={"store_id": 42}),
        Document(page_content="FarmaVita Store ID 43. Average daily sales: 3000.", metadata={"store_id": 43}),
    ]
    mock_vs.similarity_search.return_value = mock_docs
    mock_vs.docstore = MagicMock()
    mock_vs.index_to_docstore_id = {}
    mock_get_vs.return_value = mock_vs

    # Mock LLM response
    mock_response = MagicMock()
    mock_response.content = "Il negozio 42 ha vendite medie di 5000 euro."
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_response

    with patch("src.rag.main.ChatPromptTemplate") as mock_prompt:
        mock_prompt_instance = MagicMock()
        mock_prompt.from_messages.return_value = mock_prompt_instance
        mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)

        result = query_sales_advisor("Come va lo store 42?")

    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result
    assert len(result["sources"]) > 0
    assert "5000" in result["answer"]


@patch("src.rag.main.ChatBedrock")
@patch("src.rag.main.get_vector_store")
def test_query_sales_advisor_injects_missing_store(mock_get_vs, mock_chat):
    """Se lo store menzionato non è nei risultati, deve essere iniettato dal docstore."""
    from src.rag.main import query_sales_advisor
    from langchain_core.documents import Document

    target_doc = Document(
        page_content="FarmaVita Store ID 99. Average daily sales: 8000.",
        metadata={"store_id": 99},
    )

    mock_vs = MagicMock()
    # similarity_search NON restituisce store 99
    mock_vs.similarity_search.return_value = [
        Document(page_content="Store 1", metadata={"store_id": 1}),
        Document(page_content="Store 2", metadata={"store_id": 2}),
    ]
    # Ma il docstore lo contiene
    mock_vs.docstore.search.return_value = target_doc
    mock_vs.index_to_docstore_id = {"0": "doc_99"}
    mock_get_vs.return_value = mock_vs

    mock_response = MagicMock()
    mock_response.content = "Lo store 99 vende bene."
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_response

    with patch("src.rag.main.ChatPromptTemplate") as mock_prompt:
        mock_prompt_instance = MagicMock()
        mock_prompt.from_messages.return_value = mock_prompt_instance
        mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)

        result = query_sales_advisor("Riguardo allo store 99: come va?")

    assert isinstance(result, dict)
    assert "answer" in result


@patch("src.rag.main.ChatBedrock")
@patch("src.rag.main.get_vector_store")
def test_query_no_store_mentioned(mock_get_vs, mock_chat):
    """Domanda senza store ID — deve comunque funzionare."""
    from src.rag.main import query_sales_advisor
    from langchain_core.documents import Document

    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = [
        Document(page_content="Info generali", metadata={"store_id": 1}),
    ]
    mock_vs.docstore = MagicMock()
    mock_vs.index_to_docstore_id = {}
    mock_get_vs.return_value = mock_vs

    mock_response = MagicMock()
    mock_response.content = "Le vendite vanno bene in media."
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_response

    with patch("src.rag.main.ChatPromptTemplate") as mock_prompt:
        mock_prompt_instance = MagicMock()
        mock_prompt.from_messages.return_value = mock_prompt_instance
        mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)

        result = query_sales_advisor("Quali sono i trend generali?")

    assert isinstance(result, dict)
    assert "answer" in result
