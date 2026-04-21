"""Test aggiuntivi per ingestion — copertura delle linee di main()."""

from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pandas as pd
import pytest


@patch("src.ingestion.main.Path")
@patch("src.ingestion.main.create_engine")
@patch("src.ingestion.main.pd.read_sql_table")
def test_main_extracts_all_tables(mock_read_sql, mock_engine, mock_path):
    """main() deve tentare di leggere tutte le tabelle dal DB."""
    from src.ingestion.main import main

    # Simula che nessun file esista già
    mock_dir = MagicMock()
    mock_path.return_value = mock_dir
    mock_file = MagicMock()
    mock_file.exists.return_value = False
    mock_dir.__truediv__ = MagicMock(return_value=mock_file)

    mock_read_sql.return_value = pd.DataFrame({"col": [1, 2]})
    mock_engine.return_value = MagicMock()

    main()

    # Deve aver chiamato read_sql_table almeno 15 volte (tutte le tabelle)
    assert mock_read_sql.call_count >= 15


@patch("src.ingestion.main.Path")
@patch("src.ingestion.main.create_engine")
@patch("src.ingestion.main.pd.read_sql_table")
def test_main_skips_existing(mock_read_sql, mock_engine, mock_path):
    """main() deve saltare le tabelle già scaricate."""
    from src.ingestion.main import main

    mock_dir = MagicMock()
    mock_path.return_value = mock_dir
    mock_file = MagicMock()
    # Simula che tutti i file esistano già
    mock_file.exists.return_value = True
    mock_dir.__truediv__ = MagicMock(return_value=mock_file)

    mock_engine.return_value = MagicMock()

    main()

    # Non deve aver letto nulla dal DB
    mock_read_sql.assert_not_called()


@patch("src.ingestion.main.Path")
@patch("src.ingestion.main.create_engine")
@patch("src.ingestion.main.pd.read_sql_table")
def test_main_handles_read_error(mock_read_sql, mock_engine, mock_path):
    """main() deve gestire errori di lettura senza crashare."""
    from src.ingestion.main import main

    mock_dir = MagicMock()
    mock_path.return_value = mock_dir
    mock_file = MagicMock()
    mock_file.exists.return_value = False
    mock_dir.__truediv__ = MagicMock(return_value=mock_file)

    mock_read_sql.side_effect = Exception("Connection timeout")
    mock_engine.return_value = MagicMock()

    # Non deve crashare
    main()
