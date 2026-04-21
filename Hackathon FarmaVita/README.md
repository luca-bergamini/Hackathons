# FarmaVita Store Sales Predictor — Team [NOME_TEAM]

**AI Coding Tool assegnato:** [NOME_TOOL]

## Indice

- [Architettura](#architettura)
- [Setup & Installazione](#setup--installazione)
- [Sorgente Dati (Database RDS)](#sorgente-dati-database-rds)
- [Struttura del Progetto](#struttura-del-progetto)
- [Pipeline](#pipeline)
- [Data Augmentation](#data-augmentation)
- [Modello ML](#modello-ml)
- [RAG — Sales Advisor](#rag--sales-advisor)
- [Frontend](#frontend)
- [API](#api)
- [Come eseguire i test](#come-eseguire-i-test)
- [Scelte tecniche e motivazioni](#scelte-tecniche-e-motivazioni)
- [Limiti noti](#limiti-noti)

---

## Architettura

> Inserire qui un diagramma dell'architettura (ASCII, Mermaid, o immagine).

```
[Sorgente esterna]
  PostgreSQL RDS (tabelle normalizzate + tabelle augmentation sporche)
    |
    v
[Vostra sottoscrizione AWS]
  Ingestion -> Processing -> Feature Engineering -> ML Training -> API + RAG
```

**Nota:** Il database RDS e' una **sorgente dati esterna di sola lettura**. Tutta la vostra data platform (storage, processing, ML, serving, GenAI) va costruita sulla **vostra sottoscrizione AWS**.

## Setup & Installazione

### Prerequisiti
- Python 3.11+
- AWS CLI configurato con le credenziali della **vostra sottoscrizione AWS** (`aws configure`)
- Accesso in lettura al database PostgreSQL sorgente (credenziali fornite)

### Installazione

```bash
# 1. Clonare il repository (dopo il fork)
git clone <URL_DEL_VOSTRO_FORK>
cd hackaton_genaitools_1

# 2. Creare virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# 3. Installare le dipendenze
pip install -r requirements.txt

# 4. Configurare le variabili d'ambiente
cp .env.example .env
# Editare .env (le credenziali DB sono gia preconfigurate)

# 5. Verificare che tutto funzioni
python scripts/validate_setup.py
```

### Variabili d'ambiente

Creare un file `.env` nella root del progetto (vedi `.env.example`):

```bash
# Database PostgreSQL (sorgente dati - SOLA LETTURA)
DB_HOST=hackathon-farmavita-db.cjcn7vyqigdy.eu-west-1.rds.amazonaws.com
DB_PORT=5432
DB_NAME=farmavita
DB_USER=hackathon_reader
DB_PASSWORD=ReadOnly_FarmaVita2026

# AWS (la VOSTRA sottoscrizione)
AWS_REGION=eu-west-1
```

## Sorgente Dati (Database RDS)

Il database PostgreSQL e' una **sorgente dati esterna di sola lettura** che contiene tutti i dati pre-caricati per la challenge.

**Connessione:**
- **Host:** `hackathon-farmavita-db.cjcn7vyqigdy.eu-west-1.rds.amazonaws.com`
- **Porta:** `5432`
- **Database:** `farmavita`
- **User:** `hackathon_reader`
- **Password:** `ReadOnly_FarmaVita2026`
- **Accesso:** sola lettura

**Schema `raw` — 10 tabelle normalizzate:**

I dati sono **normalizzati** su piu' tabelle. Dovete fare JOIN per ricostruire il dataset di lavoro.

| Tabella | Righe | Descrizione |
|---------|-------|-------------|
| `raw.daily_sales` | 963.689 | Vendite giornaliere (SOLO store_id, date, sales, customers, is_open) |
| `raw.stores` | 1.115 | Anagrafica negozi (FK a store_types e assortment_levels) |
| `raw.store_types` | 4 | Tipi negozio (a/b/c/d) |
| `raw.assortment_levels` | 3 | Livelli assortimento (a/b/c) |
| `raw.competitions` | 1.112 | Distanza e data apertura competitor |
| `raw.promo_daily` | 404.344 | Giorni con promo attiva (store_id, date) |
| `raw.promo_continuous` | 571 | Config Promo2 (da calcolare se attiva!) |
| `raw.state_holidays` | 604 | Festivita' per stato (serve store_states per JOIN) |
| `raw.school_holidays` | 7.836 | Vacanze scolastiche per stato |
| `raw.prediction_requests` | 53.520 | Test set (solo id, store_id, date, is_open) |

**Schema `augmentation` — dati sporchi:**

| Tabella | Descrizione |
|---------|-------------|
| `augmentation.store_states` | **Chiave di JOIN** Store → State tedesco |
| `augmentation.weather` | Dati meteo (sporchi!) |
| `augmentation.google_trends` | Trend di ricerca (sporchi!) |
| `augmentation.macroeconomic` | Indicatori economici (sporchi!) |
| `augmentation.local_events` | Eventi locali (sporchi!) |

Le tabelle sono **normalizzate**: dovete esplorare lo schema e implementare i JOIN necessari per ricostruire il dataset di lavoro.

> Consultare `docs/DATABASE_SCHEMA.md` per la documentazione completa dello schema e le relazioni tra tabelle.

## Struttura del Progetto

```
├── src/
│   ├── ingestion/       # Lettura dati dal database RDS (sorgente esterna)
│   ├── processing/      # Pulizia e trasformazione dati
│   ├── features/        # Feature engineering
│   ├── models/          # Training e valutazione modelli ML
│   ├── serving/         # FastAPI + Lambda handler per API
│   ├── rag/             # RAG con Bedrock - Sales Advisor
│   ├── frontend/        # Dashboard interattiva
│   └── utils/           # Utility condivise
├── tests/               # Unit test
├── notebooks/           # Notebook esplorativi
├── configs/             # File di configurazione (database.yml)
├── docs/                # Documentazione (schema DB, ecc.)
├── scripts/             # Script utility (validazione setup, ecc.)
├── infrastructure/      # IaC (CloudFormation/CDK/Terraform)
├── requirements.txt
├── Makefile
└── .gitlab-ci.yml       # CI/CD: Ruff + pytest + SonarQube
```

## Pipeline

### 1. Ingestion
> Lettura dati dal database RDS (sorgente esterna). Esplorare lo schema, capire le relazioni tra le tabelle e ricostruire il dataset di lavoro. Salvataggio su S3 o storage locale nel vostro account AWS.

### 2. Data Augmentation
> Descrivere la strategia di arricchimento dati: quali dataset di augmentation usati, come sono stati puliti, metodo di join, impatto sulle metriche.

### 3. Processing
> Trasformazioni applicate, gestione missing values, encoding variabili categoriche.

### 4. Feature Engineering
> Elencare le feature create con il razionale per ciascuna.

| # | Feature | Descrizione | Razionale |
|---|---------|-------------|-----------|
| 1 | [nome]  | [desc]      | [perche]  |

## Data Augmentation

> I dati di augmentation sono **intenzionalmente sporchi**. Documentare qui la strategia di pulizia e integrazione.

| Fonte | Tabella DB | Problemi trovati | Strategia di pulizia | Impatto su RMSPE |
|-------|-----------|------------------|---------------------|------------------|
| [fonte] | [tabella] | [problemi] | [strategia] | [prima -> dopo] |

## Modello ML

**Task unico:** Predire le vendite giornaliere (`sales`) per ogni riga di `raw.prediction_requests` (53.520 predizioni).

- **Target:** Sales (intero, fatturato giornaliero del negozio)
- **Metrica:** RMSPE (Root Mean Square Percentage Error)
- **Scelta del modello:** Libera
- **Test set:** `raw.prediction_requests` contiene solo `id`, `store_id`, `date`, `is_open` — le feature vanno ricostruite con gli stessi JOIN usati per il training

| Modello | RMSPE | MAE | R2 | Note |
|---------|-------|-----|-----|------|
| [nome]  | [val] | [val] | [val] | [note] |

## RAG — Sales Advisor

> Architettura RAG: vector store, chunking strategy, modello Bedrock, prompt template.
> Esempi di domande/risposte supportate.

## Frontend

> Dashboard interattiva (Streamlit, Gradio, o altro) che integra:
> - Visualizzazione predizioni di vendita per negozio/data
> - Chat con il Sales Advisor (RAG) per domande sui pattern di vendita
> - Grafici e analisi esplorative dei dati
>
> La dashboard deve chiamare la vostra API (`/predict`, `/explain`, `/health`).
>
> Avvio: `streamlit run src/frontend/main.py`

## API

La vostra API deve esporre i seguenti endpoint:

### Endpoint di predizione (singolo)

```
POST /predict
Request:  {"store_id": 42, "date": "2015-09-15", "is_open": 1}
Response: {"store_id": 42, "date": "2015-09-15", "predicted_sales": 5432.0}
```

### Endpoint di predizione (batch)

Usato dalla valutazione automatica per generare le 53.520 predizioni del test set.

```
POST /predict/batch
Request:  {"requests": [{"id": 1, "store_id": 42, "date": "2015-09-15", "is_open": 1}, ...]}
Response: {"predictions": [{"id": 1, "predicted_sales": 5432.0}, ...]}
```

### Endpoint RAG

```
POST /explain
Request:  {"store_id": 42, "question": "Perche il negozio 42 ha vendite basse il lunedi?"}
Response: {"store_id": 42, "question": "...", "answer": "...", "sources": [...]}
```

### Health check

```
GET /health
Response: {"status": "healthy"}
```

## Submission

### 1. Deployare la vostra API

Deployate la vostra API su AWS (Lambda + API Gateway, ECS, EC2, ecc.) e verificate che sia raggiungibile.

### 2. Registrare l'URL dell'API

Scrivete l'URL della vostra API nel file **`api_url.txt`** nella root del repo (una sola riga):

```
https://xxx.execute-api.eu-west-1.amazonaws.com/prod
```

Commit e push su GitLab:
```bash
git add api_url.txt
git commit -m "Add API URL for evaluation"
git push
```

### 3. Self-test (opzionale)

Potete verificare che la vostra API funzioni correttamente con:
```bash
python scripts/generate_submission.py --api-url http://localhost:8000
```

Lo script chiama `/predict/batch` con tutte le 53.520 prediction_requests e salva `submission.csv`.

> **Nota:** la valutazione ufficiale viene fatta dagli organizzatori chiamando direttamente la vostra API. Non dovete consegnare nessun file CSV.

## Questionario finale

A fine hackathon, compilate il questionario interattivo per darci il vostro feedback sullo strumento AI utilizzato:

```bash
python scripts/fill_questionnaire.py
```

Le risposte vengono salvate in `questionnaire_responses.json`. Fate commit e push:

```bash
git add questionnaire_responses.json
git commit -m "Add questionnaire responses"
git push
```

## Come eseguire i test

```bash
# Tutti i test con coverage
make test

# Solo linting
make lint

# Formattare il codice
make format
```

La pipeline CI/CD su GitLab esegue automaticamente a ogni commit:
1. **Lint** (Ruff) - controlla lo stile del codice
2. **Test** (pytest) - esegue i test con coverage
3. **Quality** (SonarQube) - analisi statica su https://sonar.datareply.eu

## Scelte tecniche e motivazioni

> Documentare le scelte architetturali e tecniche piu rilevanti.

## Limiti noti

- [Limite 1]
- [Limite 2]

## Team

| Nome | Ruolo |
|------|-------|
| [nome] | [ruolo] |
