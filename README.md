# Hackathons
A repository with all my Hackathons projects

---

## Hackathon FarmaVita

**FarmaVita Store Sales Predictor** — una pipeline end-to-end per la previsione delle vendite giornaliere di una catena di farmacie tedesche.

Il progetto legge dati da un database PostgreSQL (RDS, sola lettura) con ~960k record di vendite su 1.115 negozi, e costruisce una data platform su AWS. La pipeline copre:

- **Ingestion & Processing** — join di 10 tabelle normalizzate + dati di augmentation (meteo, Google Trends, eventi locali, indicatori macroeconomici) con gestione dei dati sporchi
- **Feature Engineering** — costruzione del dataset di training a partire dallo schema relazionale
- **ML Model** — training e valutazione del modello predittivo per le vendite
- **RAG Sales Advisor** — componente GenAI basato su Amazon Bedrock per suggerimenti sulle vendite
- **API & Frontend** — servizio FastAPI + Lambda e dashboard interattiva

Stack: Python 3.11, AWS (S3, Lambda, Bedrock), FastAPI, FAISS, PostgreSQL.

---

## Hackathon Non Regression Test

**NRT Pipeline** — un sistema di Non-Regression Testing per la valutazione e il confronto di modelli LLM candidati rispetto a una baseline legacy.

La pipeline recupera un dataset da S3 e lo processa attraverso 5 step sequenziali:

1. **Data Processing** — caricamento da S3, validazione dello schema, split per `agent_id` e identificazione del task (euristica + fallback LLM)
2. **Model Selection** — lettura della configurazione dei modelli candidati e del judge da `configs/models.yaml`
3. **Runner** — inferenza parallela sui modelli candidati tramite `ThreadPoolExecutor` (10 worker), con retry e timeout
4. **Evaluation** — valutazione LLM-as-a-Judge (Claude Haiku) + metriche deterministiche per task, parallelizzata su 50 worker
5. **Reporting** — generazione di report Excel (4 fogli) e JSON, con upload opzionale su S3

Stack: Python 3.11, AWS (S3, Bedrock), React + Vite (frontend).
