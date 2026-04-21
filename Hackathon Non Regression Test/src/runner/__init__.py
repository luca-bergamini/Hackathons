"""
Runner — Step 3 della pipeline NRT.

Responsabilità:
- Esecuzione parallela delle inferenze per tutte le coppie (task, modello Candidate)
- Gestione retry, timeout e cattura errori per ogni record
- Produzione di output strutturato per record e metriche aggregate per coppia (task, modello)

Entry point: python -m src.runner.main
"""
