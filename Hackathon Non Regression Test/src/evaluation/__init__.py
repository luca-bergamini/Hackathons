"""
Evaluator Agent — Step 4 della pipeline NRT.

Responsabilità:
- Un valutatore dedicato per ogni task type supportato
- Ogni valutatore combina:
    * metriche deterministiche (exact match, accuracy, validazione strutturale JSON,
      compilazione/esecuzione codice, ...)
    * valutazione qualitativa tramite LLM-as-Judge
      (il Judge DEVE essere un modello diverso da quello valutato)
- Produzione di score strutturati per ogni record e aggregati per modello
- Copertura minima: 5 task type obbligatori

Task type supportati:
  Obbligatori: classification, context_qa, metadata_extraction,
               code_generation, text_refinement (translation/summarization/rephrasing)
  Bonus:       sql_generation, pii_redaction, intent_detection

Entry point: python -m src.evaluation.main
"""
