"""Rasterize the native Word PDF through the packaged DOCX renderer."""

import importlib.util
import os
from pathlib import Path
from typing import Any

from pypdf import PdfReader

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
RUNTIME = Path("C:/Users/Ello/.cache/codex-runtimes/codex-primary-runtime/dependencies")
SKILL = Path(
    "C:/Users/Ello/.codex/plugins/cache/openai-primary-runtime/documents/26.1013.11914/skills/documents"
)
os.environ["PATH"] = (
    str(RUNTIME / "native/poppler/Library/bin") + os.pathsep + os.environ["PATH"]
)
spec = importlib.util.spec_from_file_location(
    "packaged_docx_renderer", SKILL / "render_docx.py"
)
if spec is None or spec.loader is None:
    raise ImportError("Packaged document renderer unavailable")
renderer: Any = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def word_pdf(doc_path, user_profile, convert_tmp_dir, stem, verbose=False):
    source = WORK / "word_render.pdf"
    if not source.is_file() or not source.stat().st_size:
        raise FileNotFoundError(source)
    return str(
        source
    ), "Native Word PDF export, read-only document, hidden isolated instance."


renderer.convert_to_pdf = word_pdf
images = renderer.rasterize(
    str(ROOT / "models/reports/Laporan_Eksperimen_Batch_02_sampai_05.docx"),
    str(WORK / "rendered"),
    dpi=125,
    verbose=False,
    emit_pdf=True,
)
reader = PdfReader(WORK / "word_render.pdf")
if len(images) != len(reader.pages):
    raise ValueError("Missing page image")
for index, page in enumerate(reader.pages, 1):
    text = page.extract_text() or ""
    print(f"PAGE {index}: {len(text)} chars | {text[:115]!r} | end {text[-85:]!r}")
print("Rendered", len(images), "pages")
