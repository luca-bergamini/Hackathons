#!/usr/bin/env python3
"""
Questionario post-hackathon interattivo.

Compilatelo a fine hackathon per darci il vostro feedback sullo strumento AI utilizzato.
Le risposte vengono salvate in questionnaire_responses.json nella root del progetto.

Uso:
    python scripts/fill_questionnaire.py

Dopo aver completato il questionario:
    git add questionnaire_responses.json
    git commit -m "Add questionnaire responses"
    git push
"""

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(BASE_DIR, "questionnaire_responses.json")

# Colori terminale
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header(section, title):
    print(f"\n{CYAN}{'=' * 60}{RESET}")
    print(f"{CYAN}  {section}. {title}{RESET}")
    print(f"{CYAN}{'=' * 60}{RESET}\n")


def ask_choice(question, options, allow_multiple=False):
    """Domanda a scelta singola o multipla."""
    print(f"{BOLD}{question}{RESET}\n")
    for i, opt in enumerate(options, 1):
        print(f"  {YELLOW}{i}{RESET}) {opt}")

    if allow_multiple:
        print(f"\n{DIM}  (separa i numeri con virgola, es: 1,3,5){RESET}")
        while True:
            answer = input(f"\n  {GREEN}> {RESET}").strip()
            try:
                indices = [int(x.strip()) for x in answer.split(",")]
                if all(1 <= idx <= len(options) for idx in indices):
                    return [options[idx - 1] for idx in indices]
            except ValueError:
                pass
            print(f"  {YELLOW}Inserisci numeri validi (1-{len(options)}){RESET}")
    else:
        while True:
            answer = input(f"\n  {GREEN}> {RESET}").strip()
            try:
                idx = int(answer)
                if 1 <= idx <= len(options):
                    return options[idx - 1]
            except ValueError:
                pass
            print(f"  {YELLOW}Inserisci un numero tra 1 e {len(options)}{RESET}")


def ask_number(question, min_val=1, max_val=10):
    """Domanda con input numerico senza scala di giudizio."""
    print(f"{BOLD}{question}{RESET}")
    print(f"{DIM}  ({min_val}-{max_val}){RESET}")
    while True:
        answer = input(f"\n  {GREEN}> {RESET}").strip()
        try:
            val = int(answer)
            if min_val <= val <= max_val:
                return val
        except ValueError:
            pass
        print(f"  {YELLOW}Inserisci un numero tra {min_val} e {max_val}{RESET}")


def ask_rating(question, min_val=1, max_val=5):
    """Domanda con valutazione numerica."""
    print(f"{BOLD}{question}{RESET}")
    print(f"{DIM}  ({min_val} = pessimo, {max_val} = eccellente){RESET}")
    while True:
        answer = input(f"\n  {GREEN}> {RESET}").strip()
        try:
            val = int(answer)
            if min_val <= val <= max_val:
                return val
        except ValueError:
            pass
        print(f"  {YELLOW}Inserisci un numero tra {min_val} e {max_val}{RESET}")


def ask_text(question, required=True, multiline=False):
    """Domanda a testo libero."""
    print(f"{BOLD}{question}{RESET}")
    if multiline:
        print(f"{DIM}  (scrivi su piu' righe, riga vuota per terminare){RESET}")
        lines = []
        while True:
            line = input(f"  {GREEN}> {RESET}")
            if not line and lines:
                break
            if line:
                lines.append(line)
        return "\n".join(lines)
    else:
        while True:
            answer = input(f"\n  {GREEN}> {RESET}").strip()
            if answer or not required:
                return answer
            print(f"  {YELLOW}Risposta obbligatoria{RESET}")


def ask_yes_no(question):
    """Domanda si/no."""
    print(f"{BOLD}{question}{RESET}")
    print(f"{DIM}  (s/n){RESET}")
    while True:
        answer = input(f"\n  {GREEN}> {RESET}").strip().lower()
        if answer in ("s", "si", "y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print(f"  {YELLOW}Rispondi s o n{RESET}")


def ask_task_rating(task_name):
    """Valuta l'uso dello strumento per un task specifico."""
    used = ask_yes_no(f"Avete usato lo strumento AI per: {task_name}?")
    if not used:
        return {"used": False, "rating": None, "comment": ""}
    rating = ask_rating(f"Quanto e' stato utile per {task_name}?")
    comment = ask_text("Commento breve (opzionale):", required=False)
    return {"used": True, "rating": rating, "comment": comment}


def main():
    clear_screen()
    print(f"""
{CYAN}╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     QUESTIONARIO POST-HACKATHON FARMAVITA                ║
║     Valutazione dello strumento AI Coding                ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝{RESET}

{DIM}Le risposte verranno salvate in questionnaire_responses.json
Commit e push su GitLab al termine.{RESET}
""")

    responses = {}

    # --- Informazioni team ---
    print_header(0, "Informazioni team")
    responses["team_name"] = ask_text("Nome del team:")
    responses["ai_tool"] = ask_choice(
        "Strumento AI assegnato:",
        ["Claude Code", "Opencode", "Codex", "Antigravity", "Cursor", "Copilot", "Kiro", "Cline", "Altro"],
    )
    if responses["ai_tool"] == "Altro":
        responses["ai_tool"] = ask_text("Quale strumento?")
    responses["team_size"] = ask_number("Numero di membri del team:", min_val=1, max_val=10)

    # --- 1. Esperienza generale ---
    print_header(1, "Esperienza generale")

    responses["overall_experience"] = ask_choice(
        "Come valuteresti la tua esperienza complessiva con lo strumento AI?",
        ["Eccellente", "Buona", "Sufficiente", "Insufficiente", "Pessima"],
    )

    responses["speed_impact"] = ask_choice(
        "Lo strumento ha accelerato o rallentato il lavoro rispetto allo sviluppo tradizionale?",
        [
            "Accelerato significativamente (>50% piu' veloce)",
            "Accelerato moderatamente (20-50% piu' veloce)",
            "Velocita' simile",
            "Rallentato moderatamente",
            "Rallentato significativamente",
        ],
    )

    # --- 2. Capacita' tecniche ---
    print_header(2, "Capacita' tecniche")
    print(f"{DIM}Valuta da 1 (pessimo) a 5 (eccellente) le capacita' dello strumento:{RESET}\n")

    skill_areas = [
        "Comprensione dei requisiti",
        "Generazione codice Python",
        "Data engineering (pandas, SQL, JOIN)",
        "Machine Learning (scikit-learn, XGBoost)",
        "AWS (boto3, Lambda, Bedrock)",
        "Testing (pytest)",
        "Debugging e fix errori",
        "Refactoring e qualita' codice",
        "Documentazione (README, commenti)",
        "GenAI / RAG (Bedrock, vector store)",
    ]

    responses["skill_ratings"] = {}
    for area in skill_areas:
        responses["skill_ratings"][area] = ask_rating(f"  {area}:")

    # --- 3. Punti di forza ---
    print_header(3, "Punti di forza")

    responses["strengths"] = []
    for i in range(1, 4):
        s = ask_text(f"Punto di forza #{i}:")
        responses["strengths"].append(s)

    responses["best_moment"] = ask_text(
        "Qual e' stato il momento in cui lo strumento ti ha impressionato di piu'?"
    )

    # --- 4. Limitazioni ---
    print_header(4, "Limitazioni")

    responses["weaknesses"] = []
    for i in range(1, 4):
        w = ask_text(f"Limitazione/frustrazione #{i}:")
        responses["weaknesses"].append(w)

    responses["wrong_output"] = ask_text(
        "C'e' stato un momento in cui lo strumento ha prodotto risultati sbagliati o fuorvianti?",
        required=False,
    )

    responses["manual_fallback"] = ask_text(
        "Per quali task avete dovuto 'abbandonare' lo strumento e scrivere codice manualmente?",
        required=False,
    )

    # --- 5. Uso per task ---
    print_header(5, "Uso specifico per task")
    print(f"{DIM}Per ogni task, indica se lo strumento e' stato usato e quanto utile:{RESET}\n")

    tasks = [
        "Ingestion da PostgreSQL (JOIN tabelle)",
        "Pulizia dati augmentation (sporchi)",
        "Feature engineering",
        "Training modello ML",
        "Ottimizzazione RMSPE",
        "API FastAPI / Lambda",
        "RAG con Bedrock",
        "Scrittura test (pytest)",
        "CI/CD configuration",
        "Documentazione / README",
        "Debugging",
        "Presentazione finale",
    ]

    responses["task_usage"] = {}
    for task in tasks:
        print(f"\n{CYAN}--- {task} ---{RESET}")
        responses["task_usage"][task] = ask_task_rating(task)

    # --- 6. Confronto e raccomandazioni ---
    print_header(6, "Confronto e raccomandazioni")

    responses["would_recommend"] = ask_choice(
        "Consiglieresti questo strumento per il lavoro quotidiano?",
        [
            "Si, per tutti i task di sviluppo",
            "Si, ma solo per alcuni task specifici",
            "Non sono sicuro",
            "No, non lo consiglierei",
        ],
    )

    responses["best_use_case"] = ask_text(
        "Per quale tipo di progetto pensi che lo strumento sia piu' adatto?",
        required=False,
    )

    responses["preferred_tool"] = ask_text(
        "Se potessi scegliere qualsiasi strumento AI per il prossimo hackathon, quale sceglieresti e perche'?",
        required=False,
    )

    # --- 7. Metriche quantitative ---
    print_header(7, "Metriche quantitative (stima)")

    responses["metrics"] = {}
    responses["metrics"]["num_prompts"] = ask_text(
        "Numero approssimativo di prompt/richieste allo strumento:", required=False
    )
    responses["metrics"]["ai_code_percentage"] = ask_text(
        "% di codice generato dall'AI vs scritto manualmente (es: 70%):", required=False
    )
    responses["metrics"]["manual_corrections"] = ask_text(
        "Quante volte avete dovuto correggere l'output dell'AI?", required=False
    )
    responses["metrics"]["total_commits"] = ask_text(
        "Numero di commit totali:", required=False
    )
    responses["metrics"]["final_rmspe"] = ask_text(
        "RMSPE finale ottenuto:", required=False
    )

    # --- 8. Commenti liberi ---
    print_header(8, "Commenti liberi")

    responses["free_comments"] = ask_text(
        "Qualsiasi altro commento, suggerimento, o feedback:",
        required=False,
        multiline=True,
    )

    # --- Salvataggio ---
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)

    print(f"""
{GREEN}{'=' * 60}
  Questionario completato!
  Risposte salvate in: questionnaire_responses.json
{'=' * 60}{RESET}

  Ora fate commit e push:

    {YELLOW}git add questionnaire_responses.json
    git commit -m "Add questionnaire responses"
    git push{RESET}

{DIM}  Grazie per il feedback!{RESET}
""")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Questionario interrotto. Le risposte NON sono state salvate.{RESET}")
        sys.exit(1)
