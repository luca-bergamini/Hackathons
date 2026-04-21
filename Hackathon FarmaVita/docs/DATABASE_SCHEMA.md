# Schema Database — Hackathon FarmaVita

**Host:** `hackathon-farmavita-db.cjcn7vyqigdy.eu-west-1.rds.amazonaws.com`
**Porta:** `5432` | **Database:** `farmavita`
**User:** `hackathon_reader` | **Password:** `ReadOnly_FarmaVita2026`
**Accesso:** sola lettura

> **Nota:** Il database RDS e' una sorgente dati esterna di sola lettura.
> I dati sono distribuiti su piu' tabelle normalizzate: dovrete fare JOIN per ricostruire il dataset di lavoro.

---

## Schema `raw` — Dati FarmaVita (normalizzati)

### Tabelle dimensionali

#### raw.store_types (4 righe)
| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| id | INTEGER | PK |
| type_code | VARCHAR | Codice tipo (a, b, c, d) |
| description | VARCHAR | Descrizione del tipo di negozio |

#### raw.assortment_levels (3 righe)
| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| id | INTEGER | PK |
| level_code | VARCHAR | Codice assortimento (a, b, c) |
| description | VARCHAR | Descrizione del livello |

#### raw.stores (1.115 righe)
| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| store_id | INTEGER | PK - ID del negozio (1-1115) |
| store_type_id | INTEGER | FK -> store_types.id |
| assortment_id | INTEGER | FK -> assortment_levels.id |

#### raw.competitions (1.112 righe)
| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| store_id | INTEGER | ID del negozio |
| distance_meters | FLOAT | Distanza competitor piu' vicino (metri) |
| open_since_month | FLOAT | Mese apertura competitor (NaN se sconosciuto) |
| open_since_year | FLOAT | Anno apertura competitor (NaN se sconosciuto) |

### Tabella dei fatti

#### raw.daily_sales (963.689 righe)
Vendite giornaliere per store. **Nota: questa tabella contiene SOLO il dato transazionale puro.**

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| store_id | INTEGER | ID del negozio |
| date | VARCHAR | Data (YYYY-MM-DD) |
| sales | INTEGER | Fatturato del giorno |
| customers | INTEGER | Numero di clienti |
| is_open | INTEGER | 1=aperto, 0=chiuso |

### Tabelle promozioni

#### raw.promo_daily (404.344 righe)
Giorni in cui una promozione giornaliera era attiva per uno store.

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| store_id | INTEGER | ID del negozio |
| date | VARCHAR | Data (YYYY-MM-DD) |

Se una coppia (store_id, date) e' presente in questa tabella → la promo era attiva quel giorno.

#### raw.promo_continuous (571 righe)
Configurazione della promozione continuativa (Promo2). **I team devono calcolare se era attiva in una data specifica** usando since_week, since_year e active_months.

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| store_id | INTEGER | ID del negozio |
| since_week | FLOAT | Settimana di inizio Promo2 |
| since_year | FLOAT | Anno di inizio Promo2 |
| active_months | VARCHAR | Mesi in cui Promo2 e' attiva (es. "Jan,Apr,Jul,Oct") |

### Tabelle festività

#### raw.state_holidays (604 righe)
Festività statali per stato tedesco. **Per usare questa tabella serve il JOIN con `augmentation.store_states`** per sapere in quale stato si trova ogni negozio.

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| state | VARCHAR | Stato tedesco |
| date | VARCHAR | Data (YYYY-MM-DD) |
| holiday_type | VARCHAR | a=pubblico, b=Pasqua, c=Natale |

#### raw.school_holidays (7.836 righe)
Vacanze scolastiche per stato tedesco. Stessa logica di JOIN con store_states.

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| state | VARCHAR | Stato tedesco |
| date | VARCHAR | Data (YYYY-MM-DD) |

### Test set

#### raw.prediction_requests (53.520 righe)
Le righe per cui dovete produrre le predizioni di vendita. **Contiene solo id, store, data e is_open — la colonna Sales NON c'e'!**
Generare le predizioni con `python scripts/generate_submission.py`.

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| id | INTEGER | PK - ID della predizione |
| store_id | INTEGER | ID del negozio |
| date | VARCHAR | Data (YYYY-MM-DD) |
| is_open | INTEGER | 1=aperto, 0=chiuso |

---

## Schema `augmentation` — Dati di arricchimento (sporchi)

Invariato. I dati di arricchimento sono intenzionalmente sporchi.

#### augmentation.store_states (1.115 righe)
**CHIAVE DI JOIN FONDAMENTALE** — mappa ogni Store al suo State tedesco. Necessaria per collegare daily_sales alle tabelle holidays e ai dataset di arricchimento.

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| Store | INTEGER | ID del negozio |
| State | VARCHAR | Stato tedesco |

#### augmentation.weather / google_trends / macroeconomic / local_events
Vedi documentazione precedente. Tutti intenzionalmente sporchi.

---

## Relazioni tra tabelle (ERD)

```
raw.stores ──FK──> raw.store_types
    │  └──FK──> raw.assortment_levels
    │
    ├── raw.daily_sales (store_id, date)
    │       Per ricostruire il dataset completo, JOIN con:
    │       ├── raw.promo_daily (store_id, date)
    │       ├── raw.promo_continuous (store_id) + calcolo logico
    │       ├── raw.competitions (store_id)
    │       ├── raw.stores → store_types / assortment_levels
    │       └── augmentation.store_states → raw.state_holidays
    │                                     → raw.school_holidays
    │
    └── raw.prediction_requests (store_id, date)
            Stessi JOIN necessari per creare le feature di predizione
```

