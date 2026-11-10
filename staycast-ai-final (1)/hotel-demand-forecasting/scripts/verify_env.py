"""Phase 1 environment check: python version + required packages.

Run:  python scripts/verify_env.py
"""
import importlib.metadata as md
import sys

REQUIRED = [
    "pandas", "numpy", "scikit-learn", "joblib", "holidays", "pydantic",
    "fastapi", "uvicorn", "httpx", "matplotlib", "seaborn", "plotly", "pytest",
]

ok = sys.version_info >= (3, 10)
print(f"{'OK ' if ok else 'BAD'} python {sys.version.split()[0]} (need >= 3.10)")
for pkg in REQUIRED:
    try:
        print(f"OK  {pkg}=={md.version(pkg)}")
    except md.PackageNotFoundError:
        ok = False
        print(f"BAD {pkg} not installed  ->  pip install -r requirements.txt")
sys.exit(0 if ok else 1)
