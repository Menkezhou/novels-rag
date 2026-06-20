#!/usr/bin/env python3
"""
导出 ChromaDB 向量数据为 JSON（用于备份或迁移）
只会导出文本+元数据，向量本身二进制暂不导出（太大）
"""

import json, os, sys
import chromadb
from chromadb.config import Settings

CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export.json")
COLLECTION = "novel_chunks_v4"
BATCH = 1000


def export():
    if not os.path.exists(CHROMA_DIR):
        print(f"错误: 找不到 {CHROMA_DIR}，请先运行 rag_novel_v4.py 构建索引")
        sys.exit(1)

    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
    col = client.get_collection(COLLECTION)
    total = col.count()
    print(f"数据库中 {total} 个块，开始导出...")

    all_data = []
    for offset in range(0, total, BATCH):
        limit = min(BATCH, total - offset)
        result = col.get(limit=limit, offset=offset, include=["documents", "metadatas"])
        for doc_id, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
            all_data.append({"id": doc_id, "text": doc, "meta": meta})
        print(f"\r  {min(offset + BATCH, total)}/{total}", end="")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
    print(f"\n导出完成: {OUTPUT} ({size_mb:.1f}MB, {len(all_data)} 条)")


if __name__ == "__main__":
    export()
