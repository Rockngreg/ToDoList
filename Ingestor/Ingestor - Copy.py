import os
import hashlib
from typing import List, Dict, Optional, Union
import json

import pandas as pd

import docx  # python-docx
import pdfplumber
from PIL import Image
import pytesseract

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dataclasses import dataclass, asdict


@dataclass
class Record:
    id: str
    source_type: str  # 'docx' | 'pdf' | 'sql'
    source_name: str
    chunk_id: Union[int, str]
    text: str
    metadata: Dict


class Ingestor:
    def __init__(self, sql_engine: Union[Engine, str, None] = None, 
                 max_tokens: int = 500, overlap: int = 50):
        """
        :param sql_engine: SQLAlchemy engine or engine URL string. Can be None if not using SQL.
        :param max_tokens: approximate chunk size in words (naive).
        :param overlap: number of overlapping words between chunks.
        """
        if isinstance(sql_engine, str):
            self.engine = create_engine(sql_engine)
        else:
            self.engine = sql_engine  # can be None
        self.max_tokens = max_tokens
        self.overlap = overlap

    # ---- utility ----
    def _make_id(self, *parts) -> str:
        m = hashlib.sha256()
        for p in parts:
            if isinstance(p, str):
                m.update(p.encode("utf-8"))
            else:
                m.update(str(p).encode("utf-8"))
        return m.hexdigest()[:16]

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.max_tokens, len(words))
            chunk_words = words[start:end]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
            if end == len(words):
                break
            start = end - self.overlap
            if start < 0:
                start = 0
        return chunks

    # ---- DOCX ----
    def ingest_docx(self, path: str) -> List[Record]:
        doc = docx.Document(path)
        full_text = []
        for para in doc.paragraphs:
            if para.text and para.text.strip():
                full_text.append(para.text.strip())
        combined = "\n".join(full_text)
        chunks = self._chunk_text(combined)
        records: List[Record] = []
        for idx, chunk in enumerate(chunks):
            rec = Record(
                id=self._make_id(path, idx),
                source_type="docx",
                source_name=os.path.basename(path),
                chunk_id=idx,
                text=chunk,
                metadata={
                    "file_path": path,
                    "chunk_index": idx,
                    "original_length_words": len(chunk.split()),
                },
            )
            records.append(rec)
        return records

    # ---- PDF ----
    def _extract_text_pdf_native(self, path: str) -> List[str]:
        pages_text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                try:
                    txt = page.extract_text() or ""
                except Exception:
                    txt = ""
                pages_text.append(txt)
        return pages_text

    def _is_mostly_empty(self, pages_text: List[str], threshold_chars: int = 50) -> bool:
        non_empty = [p for p in pages_text if len(p.strip()) > threshold_chars]
        ratio = len(non_empty) / max(1, len(pages_text))
        return ratio < 0.3

    def _ocr_pdf_page(self, page_image: Image.Image, ocr_lang: str) -> str:
        return pytesseract.image_to_string(page_image, lang=ocr_lang)

    def ingest_pdf(self, path: str, ocr_if_needed: bool = True, ocr_lang: str = "eng") -> List[Record]:
        records: List[Record] = []
        pages_native = self._extract_text_pdf_native(path)
        need_ocr = ocr_if_needed and self._is_mostly_empty(pages_native)
        for page_idx, native_text in enumerate(pages_native):
            text_to_use = native_text or ""
            used_ocr = False
            if need_ocr or (not native_text.strip() and ocr_if_needed):
                try:
                    with pdfplumber.open(path) as pdf:
                        page = pdf.pages[page_idx]
                        pil_image = page.to_image(resolution=300).original
                        ocr_text = self._ocr_pdf_page(pil_image, ocr_lang)
                        if ocr_text.strip():
                            text_to_use = ocr_text
                            used_ocr = True
                except Exception:
                    pass  # fall back to native if OCR fails
            if not text_to_use.strip():
                continue
            chunks = self._chunk_text(text_to_use)
            for chunk_idx, chunk in enumerate(chunks):
                rec = Record(
                    id=self._make_id(path, page_idx, chunk_idx),
                    source_type="pdf",
                    source_name=os.path.basename(path),
                    chunk_id=f"{page_idx}_{chunk_idx}",
                    text=chunk,
                    metadata={
                        "file_path": path,
                        "page_number": page_idx + 1,
                        "chunk_index": chunk_idx,
                        "used_ocr": used_ocr,
                        "original_native_text_length": len(native_text.split()),
                    },
                )
                records.append(rec)
        return records

    # ---- SQL ----
    def fetch_sql_table(
        self,
        table_name: Optional[str] = None,
        query: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        if self.engine is None:
            raise RuntimeError("SQL engine not configured.")
        with self.engine.connect() as conn:
            if query:
                q = text(query)
            elif table_name:
                q = text(f"SELECT * FROM {table_name}")
            else:
                raise ValueError("Either table_name or query must be provided.")
            if limit:
                q = text(str(q).rstrip(";") + f" LIMIT {limit}")
            df = pd.read_sql(q, conn)
        return df

    def records_from_dataframe(
        self,
        df: pd.DataFrame,
        source_name: str,
        key_columns: Optional[List[str]] = None,
        text_columns: Optional[List[str]] = None
    ) -> List[Record]:
        if text_columns is None:
            text_columns = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
        if key_columns is None:
            key_columns = df.columns.tolist()
        records: List[Record] = []
        for idx, row in df.iterrows():
            parts = []
            for col in text_columns:
                val = row.get(col, "")
                if pd.isna(val):
                    continue
                parts.append(str(val))
            combined = " | ".join(parts).strip()
            if not combined:
                continue
            chunks = self._chunk_text(combined)
            key_vals = [str(row.get(k, "")) for k in key_columns]
            base_id = self._make_id(source_name, *key_vals)
            for chunk_idx, chunk in enumerate(chunks):
                rec = Record(
                    id=self._make_id(base_id, chunk_idx),
                    source_type="sql",
                    source_name=source_name,
                    chunk_id=f"{idx}_{chunk_idx}",
                    text=chunk,
                    metadata={
                        "row_index": int(idx),
                        "chunk_index": chunk_idx,
                        "key_values": {k: row.get(k) for k in key_columns},
                        "text_columns_used": text_columns,
                    },
                )
                records.append(rec)
        return records

    def ingest_sql_sources(
        self,
        tables: Optional[List[str]] = None,
        queries: Optional[List[str]] = None,
        key_columns_override: Optional[Dict[str, List[str]]] = None,
        text_columns_override: Optional[Dict[str, List[str]]] = None,
        limit_per_source: Optional[int] = None
    ) -> List[Record]:
        all_records: List[Record] = []
        if tables:
            for tbl in tables:
                df = self.fetch_sql_table(table_name=tbl, limit=limit_per_source)
                key_cols = key_columns_override.get(tbl) if key_columns_override else None
                text_cols = text_columns_override.get(tbl) if text_columns_override else None
                recs = self.records_from_dataframe(df, source_name=tbl, key_columns=key_cols, text_columns=text_cols)
                all_records.extend(recs)
        if queries:
            for i, qry in enumerate(queries):
                alias = f"query_{i}"
                df = self.fetch_sql_table(query=qry, limit=limit_per_source)
                key_cols = key_columns_override.get(alias) if key_columns_override else None
                text_cols = text_columns_override.get(alias) if text_columns_override else None
                recs = self.records_from_dataframe(df, source_name=alias, key_columns=key_cols, text_columns=text_cols)
                all_records.extend(recs)
        return all_records

    # ---- export helpers ----
    @staticmethod
    def to_jsonl(records: List[Record], output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(asdict(r), default=str) + "\n")

    @staticmethod
    def to_dataframe(records: List[Record]) -> pd.DataFrame:
        rows = []
        for r in records:
            flat = {
                "id": r.id,
                "source_type": r.source_type,
                "source_name": r.source_name,
                "chunk_id": r.chunk_id,
                "text": r.text,
            }
            for k, v in r.metadata.items():
                flat[f"meta_{k}"] = v
            rows.append(flat)
        return pd.DataFrame(rows)


# Initialize with SQL connection (or omit if only docs)
ingestor = Ingestor(sql_engine=None)

# Ingest documents
docx_recs = ingestor.ingest_docx("C:/Users/gleblanc/OneDrive - Corporate Services, LLC/Documents/Python/Ingestor/specs.docx")
pdf_recs = ingestor.ingest_pdf("C:/Users/gleblanc/OneDrive - Corporate Services, LLC/Documents/Python/Ingestor/nimbustrack_release_notes.pdf")

# Ingest SQL


# Combine and export
all_recs = docx_recs + pdf_recs
df = Ingestor.to_dataframe(all_recs)
ingestor.to_jsonl(all_recs, "C:/Users/gleblanc/OneDrive - Corporate Services, LLC/Documents/Python/Ingestor/ingested.jsonl")
