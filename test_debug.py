import json, sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
from tests.test_phase12b_6_freeze import setup_valid_files
manifest_p, preflight_p, cohort_p, sap_p, ledger_p, cm, preflight, manifest = setup_valid_files(Path("/tmp"))
print("preflight manifest:", preflight.manifest_canonical_fingerprint)
print("manifest manifest:", manifest.manifest_fingerprint)
