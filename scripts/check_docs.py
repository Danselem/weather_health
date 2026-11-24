"""Fail if Markdown references a local image or document that does not exist."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
FILES = [ROOT / "README.md", *sorted(
    path for path in (ROOT / "docs").rglob("*.md") if "archive" not in path.parts
)]
PATTERNS = [
    re.compile(r'src="([^"]+)"'),
    re.compile(r"!\[[^\]]*\]\(([^)]+)\)"),
    re.compile(r"\[[^\]]+\]\(([^)]+\.md(?:#[^)]*)?)\)"),
]

missing = []
for path in FILES:
    text = path.read_text(encoding="utf-8")
    for pattern in PATTERNS:
        for target in pattern.findall(text):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            clean = target.split("#", 1)[0]
            resolved = (path.parent / clean).resolve()
            if not resolved.exists():
                missing.append(f"{path.relative_to(ROOT)} -> {target}")

if missing:
    print("Broken local documentation references:")
    print("\n".join(missing))
    sys.exit(1)
print(f"Checked {len(FILES)} Markdown files; all local references exist.")
