# Aggiornamento Regole Bonus

Sono stati introdotti requisiti aggiuntivi per ogni componente bonus. I livelli precedenti restano validi â€” i nuovi requisiti consentono di ottenere punteggi piÃ¹ alti su ogni bonus.

---

## BN_EXTENDED_TASKS â€” Extended Task Coverage

Oltre ai valutatori base per i 3 task opzionali, per raggiungere il punteggio massimo:

- **SQL Generation**: usa una libreria di parsing dedicata (es. `sqlparse`) invece di regex o controlli manuali
- **PII Redaction**: calcola precision, recall **e F1** a livello di singola entitÃ  (non solo presenza/assenza)
- **Intent Detection**: implementa una valutazione con **precision@k** o ranking degli intent, non solo exact match

---

## BN_ADVANCED_EVAL â€” Advanced Evaluation

Oltre alle tre tecniche base, per raggiungere il punteggio massimo:

- **Gestione senza ground truth**: implementa **self-consistency cross-model** aggregando i segnali da almeno 2 modelli diversi (non solo un singolo LLM-as-Judge)
- **Validazione code generation**: oltre all'esecuzione con cattura stdout/stderr/exit code, genera **test unitari automaticamente** tramite LLM per verificare la correttezza funzionale del codice prodotto
- **Metriche composite**: i pesi devono essere **documentati con rationale esplicito** nel codice, specifici per tipo di task

---

## BN_INSIGHT_AGENT â€” Insight Agent

Oltre agli insight interpretativi base, per raggiungere il punteggio massimo:

- I pattern di errore devono **citare i test case specifici** (test_id) che li hanno generati
- Ogni pattern deve avere un **severity level** strutturato: `high`, `medium`, `low`
- L'agent deve produrre **raccomandazioni operative** (es. "usare modello X per task Y", "aumentare max_tokens per summarization") â€” non solo osservazioni descrittive

---

## BN_S3_PERSISTENCE â€” Persistenza S3

Oltre al salvataggio degli output su S3, per raggiungere il punteggio massimo:

- Le chiavi S3 devono essere **deterministiche e idempotenti**: lo stesso `job_id` deve sempre produrre le stesse chiavi (rieseguire un job non crea duplicati)
- Produrre un file **`manifest.json`** per ogni job che elenca tutti gli artefatti con path S3 e dimensione
- Implementare il **listing dei job passati** leggendo i manifest da S3 (es. per popolare la lista job nella UI)

---

## BN_ADVANCED_FRONTEND â€” Visualizzazione avanzata

Oltre alle quattro funzionalitÃ  avanzate, per raggiungere il punteggio massimo:

- I dati del job in corso devono **aggiornarsi automaticamente** con polling â‰¤ 5 secondi, senza che l'utente debba ricaricare la pagina manualmente
- Il comportamento di polling deve essere verificabile nel codice

---

## BN_SYNTHETIC_DATASET â€” Synthetic Dataset Generation Agent

Oltre alla generazione base con variabilitÃ  controllata, per raggiungere il punteggio massimo:

- I record sintetici devono essere **auto-validati** passandoli attraverso l'Evaluator prima di aggiungerli al dataset â€” i record che non superano la validazione vengono scartati o corretti
- L'agente deve **mirare specificamente agli edge case emersi dai fallimenti** della pipeline: analizza i test case con score basso e genera nuovi record simili per aumentare la copertura di quelle aree critiche

---

## BN_PROMPT_OPTIMIZATION â€” Prompt Optimization Agent

Oltre al loop iterativo base, per raggiungere il punteggio massimo:

- **Ottimizzazione multi-obiettivo**: i pesi su qualitÃ , costo e latenza devono essere configurabili (non ottimizzare solo la qualitÃ )
- **Test di significativitÃ  statistica**: dimostra che il prompt ottimale Ã¨ statisticamente migliore della baseline (es. t-test, bootstrap)
- **Report di ablation**: mostra il contributo di ogni singola modifica al prompt â€” quale parte del cambiamento ha portato il miglioramento
