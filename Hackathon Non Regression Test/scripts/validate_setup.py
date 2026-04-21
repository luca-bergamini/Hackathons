"""Setup validator for the Model Migration team template.

Usage:
    python scripts/validate_setup.py
"""

import os
import sys
from dotenv import load_dotenv


def check_mark(ok: bool) -> str:
    return "OK" if ok else "ERROR"


def check_python_version() -> bool:
    version = sys.version_info
    ok = version.major == 3 and version.minor >= 11
    print(f"  [{check_mark(ok)}] Python {version.major}.{version.minor}.{version.micro} (required: 3.11+)")
    return ok


def check_dependencies() -> bool:
    deps = {
        "boto3": "boto3",
        "pandas": "pandas",
        "openpyxl": "openpyxl",
        "streamlit": "streamlit",
        "yaml": "pyyaml",
        "dotenv": "python-dotenv",
        "ipykernel": "ipykernel",
    }
    dev_deps = {
        "pytest": "pytest",
        "pytest_cov": "pytest-cov",
        "ruff": "ruff",
    }

    all_ok = True
    for module, package in deps.items():
        try:
            __import__(module)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [ERROR] {package} not installed — run: make setup")
            all_ok = False

    for module, package in dev_deps.items():
        try:
            __import__(module)
            print(f"  [OK] {package} (dev)")
        except ImportError:
            print(f"  [WARN] {package} not installed — run: make setup")
            # dev deps non bloccanti

    return all_ok


def check_env_variables() -> bool:
    required = ["AWS_REGION", "S3_BUCKET"]

    all_ok = True
    for var in required:
        val = os.environ.get(var)
        if val:
            print(f"  [OK] {var} = {val}")
        else:
            print(f"  [ERROR] {var} not set — controlla il tuo .env")
            all_ok = False

    return all_ok


def check_aws_access() -> bool:
    try:
        import boto3
        sts = boto3.client("sts", region_name=os.environ.get("AWS_REGION", "eu-west-1"))
        identity = sts.get_caller_identity()
        print(f"  [OK] AWS account: {identity['Account']} ({identity['Arn']})")
    except Exception as e:
        print(f"  [ERROR] Credenziali AWS non valide o non configurate: {e}")
        return False

    bucket = os.environ.get("S3_BUCKET")
    if bucket:
        try:
            s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "eu-west-1"))
            _acct = os.environ.get("AWS_ACCOUNT_ID", "")
            _owner_kw = {"ExpectedBucketOwner": _acct} if _acct else {}
            s3.head_bucket(Bucket=bucket, **_owner_kw)
            print(f"  [OK] S3 bucket '{bucket}' raggiungibile")
        except Exception as e:
            print(f"  [ERROR] S3 bucket '{bucket}' non raggiungibile: {e}")
            return False
    else:
        print("  [INFO] S3_BUCKET non impostata — skip verifica bucket")

    return True


def check_bedrock_access() -> bool:
    try:
        import boto3
        region = os.environ.get("AWS_REGION", "eu-west-1")
        bedrock = boto3.client("bedrock", region_name=region)
        bedrock.list_foundation_models()
        print(f"  [OK] Bedrock raggiungibile in {region}")
        return True
    except Exception as e:
        print(f"  [ERROR] Bedrock non raggiungibile: {e}")
        return False


def main() -> int:
    print("=" * 60)
    print("  SETUP VALIDATION - NRT Pipeline Team Template")
    print("=" * 60)

    load_dotenv()

    results = {}

    print("\n1. Python")
    results["python"] = check_python_version()

    print("\n2. Dipendenze")
    results["deps"] = check_dependencies()

    print("\n3. Variabili d'ambiente")
    results["env"] = check_env_variables()

    print("\n4. Accesso AWS e S3")
    results["aws"] = check_aws_access()

    print("\n5. Accesso Bedrock")
    results["bedrock"] = check_bedrock_access()

    print("\n" + "=" * 60)
    print("  RIEPILOGO")
    print("=" * 60)
    all_ok = all(results.values())
    for name, ok in results.items():
        print(f"  {check_mark(ok):6s} {name}")

    if all_ok:
        print("\n  Tutto ok. Siete pronti per l'hackathon!")
    else:
        print("\n  Alcuni check falliti. Correggete i problemi sopra e riprovate.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
