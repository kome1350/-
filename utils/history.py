"""
生成した画像・動画をディスクに保存し、履歴として一覧・取得するためのユーティリティ。

outputs/ 配下に実体ファイルと、同名 + .json のメタデータファイルを保存する。
"""

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class GenerationRecord:
    kind: str  # "image" または "video"
    prompt: str
    provider: str
    model: str
    timestamp: str
    file: str

    @property
    def path(self) -> Path:
        return OUTPUT_DIR / self.file


def save_generation(
    kind: str,
    data: bytes,
    prompt: str,
    provider: str,
    model: str,
    ext: str,
) -> GenerationRecord:
    """生成物を保存し、メタデータとともに GenerationRecord を返す。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    filename = f"{kind}_{ts}_{uid}.{ext}"
    file_path = OUTPUT_DIR / filename
    file_path.write_bytes(data)

    record = GenerationRecord(
        kind=kind,
        prompt=prompt,
        provider=provider,
        model=model,
        timestamp=ts,
        file=filename,
    )
    meta_path = OUTPUT_DIR / f"{filename}.json"
    meta_path.write_text(
        json.dumps(asdict(record), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return record


def list_generations(kind: Optional[str] = None) -> List[GenerationRecord]:
    """保存済みの生成履歴を新しい順に返す。"""
    records: List[GenerationRecord] = []
    for meta_file in sorted(OUTPUT_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            record = GenerationRecord(**data)
        except Exception:
            continue
        if not record.path.exists():
            continue
        if kind and record.kind != kind:
            continue
        records.append(record)
    return records


def delete_generation(record: GenerationRecord) -> None:
    """生成物とメタデータを削除する。"""
    meta_path = OUTPUT_DIR / f"{record.file}.json"
    record.path.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)
