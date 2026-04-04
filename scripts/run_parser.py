from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.parser import RAGAnythingParser

SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".txt", ".md", ".png", ".jpg", ".jpeg"}


def parse_single(parser: RAGAnythingParser, path: Path, root: Path | None = None, limit: int = 5) -> dict:
    source_uri = str(path.relative_to(root)).replace("\\", "/") if root else path.name
    elements = parser.parse_file(source_uri=source_uri, path=path)
    preview = [
        {
            "fragment_id_hint": f"{source_uri}:{e.element_index}",
            "element_index": e.element_index,
            "type": e.type,
            "page": e.page,
            "content_preview": e.content[:300],
            "meta": e.meta,
        }
        for e in elements[:limit]
    ]
    return {
        "source_uri": source_uri,
        "path": str(path),
        "elements_count": len(elements),
        "preview": preview,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run RAG-Anything parser without FastAPI")
    ap.add_argument("--input", required=True, help="Path to file or directory")
    ap.add_argument("--preview-limit", type=int, default=5, help="How many parsed elements to print per file")
    ap.add_argument("--json", action="store_true", help="Print JSON output")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input path does not exist: {input_path}")

    parser = RAGAnythingParser()

    if input_path.is_file():
        result = parse_single(parser, input_path, limit=args.preview_limit)
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result)
        return

    files = [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    results = [parse_single(parser, p, root=input_path, limit=args.preview_limit) for p in files]
    output = {"input": str(input_path), "files_count": len(files), "results": results}
    print(json.dumps(output, ensure_ascii=False, indent=2) if args.json else output)


if __name__ == "__main__":
    main()
