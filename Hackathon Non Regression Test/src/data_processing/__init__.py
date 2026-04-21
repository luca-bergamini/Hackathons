"""
Data Processing Agent — Step 1 della pipeline NRT.

Responsabilità:
- Connessione ad AWS S3 tramite boto3 e lettura del dataset JSONL
- Validazione dello schema: verifica presenza dei campi obbligatori e opzionali per ogni record
  (test_id, input_messages, ground_truth, expected_output_type, metadata, agent_id, output)
- Skip automatico dei record malformati con logging dedicato
- Suddivisione del dataset per agent_id
- Identificazione del task type per ogni agent_id tramite analisi di input_messages
  (classi: classification, context_qa, metadata_extraction, code_generation,
   translation, summarization, rephrasing, sql_generation, pii_redaction, intent_detection)
- Produzione del file CSV di profiling con colonne:
  agent_id, task, numero_record, percentuale_ground_truth, numero_record_malformati

Entry point: python -m src.data_processing.main
"""
