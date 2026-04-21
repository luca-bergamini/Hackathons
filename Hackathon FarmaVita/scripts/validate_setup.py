"""
Script di validazione setup per i team dell'hackathon.
Verifica che l'ambiente sia configurato correttamente.

Uso: python scripts/validate_setup.py
"""

import os
import sys


def check_mark(ok):
    return "OK" if ok else "ERRORE"


def check_python_version():
    """Verifica la versione di Python."""
    version = sys.version_info
    ok = version.major == 3 and version.minor >= 11
    print(f"  [{check_mark(ok)}] Python {version.major}.{version.minor}.{version.micro} (richiesto: 3.11+)")
    return ok


def check_dependencies():
    """Verifica che le dipendenze principali siano installate."""
    deps = {
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "psycopg2": "psycopg2-binary",
        "sqlalchemy": "sqlalchemy",
        "fastapi": "fastapi",
        "boto3": "boto3",
        "langchain": "langchain",
        "faiss": "faiss-cpu",
        "xgboost": "xgboost",
    }

    all_ok = True
    for module, package in deps.items():
        try:
            __import__(module)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [ERRORE] {package} non installato -> pip install {package}")
            all_ok = False
    return all_ok


def check_env_variables():
    """Verifica che le variabili d'ambiente siano configurate."""
    required = ["DB_HOST", "DB_USER", "DB_PASSWORD"]
    optional = ["DB_PORT", "DB_NAME", "AWS_REGION"]

    all_ok = True
    for var in required:
        val = os.environ.get(var)
        if val:
            display = val[:5] + "..." if "PASSWORD" in var else val
            print(f"  [OK] {var} = {display}")
        else:
            print(f"  [ERRORE] {var} non configurata")
            all_ok = False

    for var in optional:
        val = os.environ.get(var)
        if val:
            print(f"  [OK] {var} = {val}")
        else:
            print(f"  [INFO] {var} non configurata (usa default)")

    return all_ok


def check_database_connection():
    """Verifica la connessione in sola lettura al database PostgreSQL sorgente."""
    try:
        import psycopg2

        host = os.environ.get("DB_HOST", "hackathon-farmavita-db.cjcn7vyqigdy.eu-west-1.rds.amazonaws.com")
        port = os.environ.get("DB_PORT", "5432")
        dbname = os.environ.get("DB_NAME", "farmavita")
        user = os.environ.get("DB_USER", "")
        password = os.environ.get("DB_PASSWORD", "")

        if not user or not password:
            print("  [SKIP] DB_USER o DB_PASSWORD non configurate")
            return False

        conn = psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password, connect_timeout=10)
        cur = conn.cursor()

        # Verifica accesso in lettura schema raw (normalizzato)
        raw_tables = [
            "store_types", "assortment_levels", "stores", "competitions",
            "daily_sales", "promo_daily", "promo_continuous",
            "state_holidays", "school_holidays", "prediction_requests",
        ]
        for table in raw_tables:
            cur.execute(f"SELECT COUNT(*) FROM raw.{table}")
            count = cur.fetchone()[0]
            print(f"  [OK] raw.{table}: {count:,} righe")

        # Verifica accesso in lettura schema augmentation
        for table in ["store_states", "weather", "google_trends", "macroeconomic", "local_events"]:
            cur.execute(f'SELECT COUNT(*) FROM augmentation."{table}"')
            count = cur.fetchone()[0]
            print(f"  [OK] augmentation.{table}: {count:,} righe")

        conn.close()
        print("  [OK] Accesso in sola lettura verificato")
        return True

    except psycopg2.OperationalError as e:
        print(f"  [ERRORE] Connessione fallita: {e}")
        return False
    except Exception as e:
        print(f"  [ERRORE] {e}")
        return False


def check_aws():
    """Verifica la configurazione AWS (sottoscrizione del team)."""
    try:
        import boto3

        sts = boto3.client("sts", region_name=os.environ.get("AWS_REGION", "eu-west-1"))
        identity = sts.get_caller_identity()
        print(f"  [OK] AWS Account: {identity['Account']}")
        print(f"  [OK] AWS User: {identity['Arn']}")
        return True
    except Exception as e:
        print(f"  [ERRORE] AWS non configurato: {e}")
        return False


def main():
    print("=" * 60)
    print("  VALIDAZIONE SETUP - Hackathon FarmaVita")
    print("=" * 60)

    results = {}

    print("\n1. Python")
    results["python"] = check_python_version()

    print("\n2. Dipendenze")
    results["deps"] = check_dependencies()

    print("\n3. Variabili d'ambiente")
    results["env"] = check_env_variables()

    print("\n4. Connessione Database sorgente (sola lettura)")
    results["db"] = check_database_connection()

    print("\n5. AWS (sottoscrizione del team)")
    results["aws"] = check_aws()

    # Riepilogo
    print("\n" + "=" * 60)
    print("  RIEPILOGO")
    print("=" * 60)
    all_ok = all(results.values())
    for name, ok in results.items():
        print(f"  {check_mark(ok):6s} {name}")

    if all_ok:
        print("\n  Tutto OK! Sei pronto per l'hackathon.")
    else:
        print("\n  Alcuni controlli sono falliti. Risolvi i problemi sopra e riprova.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
