"""Licensing guards enforced as tests rather than as documentation.

docs/LICENSING.md records that the project must not take on an AGPL dependency. MiVOLO's
own repository depends on Ultralytics YOLOv8 (AGPL-3.0) for its detector, which would be
network-viral for a deployed service. We vendor only MiVOLO's model definition and use our
own SCRFD detector. That is easy to undo by accident, so it is asserted here.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_ultralytics_is_not_installed_or_imported():
    """Importing the MiVOLO adapter must not pull in AGPL-licensed ultralytics."""
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import facet.third_party.mivolo;"
        "import importlib.util as u;"
        "assert 'ultralytics' not in sys.modules, 'ultralytics was imported';"
        "print('ok')" % str(SRC)
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


def test_yolo_detector_is_not_vendored():
    """The AGPL-dependent file must never appear in our tree."""
    vendored = list((SRC / "facet" / "third_party" / "mivolo").glob("*.py"))
    names = {p.name for p in vendored}
    assert "yolo_detector.py" not in names
    assert names, "vendored MiVOLO model files are missing"


def test_no_source_file_imports_ultralytics():
    """Detect real imports, not prose.

    The word "ultralytics" appears legitimately in comments explaining why we avoid it, so
    this parses the AST and looks at import statements only.
    """
    import ast

    hits = []
    for p in SRC.rglob("*.py"):
        tree = ast.parse(p.read_text(), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n.split(".")[0] == "ultralytics" for n in names):
                hits.append(f"{p.relative_to(ROOT)}:{node.lineno}")
    assert not hits, f"ultralytics imported in: {hits}"


def test_model_registry_declares_license_for_every_entry():
    """Every model in the registry must carry license + commercial_use, so a
    --commercial-safe run can mechanically exclude non-compliant components."""
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "models" / "registry.yaml").read_text())
    missing = []
    for section, entries in cfg.items():
        if not isinstance(entries, dict):
            continue
        for name, spec in entries.items():
            if not isinstance(spec, dict):
                continue
            if "license" not in spec or "commercial_use" not in spec:
                missing.append(f"{section}.{name}")
    assert not missing, f"registry entries missing license metadata: {missing}"
