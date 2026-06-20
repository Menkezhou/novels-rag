#!/usr/bin/env python3
"""
RAG 小说问答系统 v4
- 父子检索：小块(200-300字)嵌入检索 + 邻块扩展为父上下文生成
- 双路召回：向量检索(ChromaDB) + TF-IDF关键词检索 → RRF融合
- 扩大召回：TOP-20 → LLM重排序 → TOP-5
- CPU限速：psutil实时监控 → 自动调节休眠 → 维持85-90%占用
- 增量导入：检测已索引文件 → 断点续跑 → 支持中断恢复
"""

import os
import re
import math
import time
import json
import requests
import chromadb
import psutil
from chromadb.config import Settings
from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict

# ============================================================
# 配置
# ============================================================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:1111")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOVEL_DIR = os.path.join(BASE_DIR, "novels")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
BM25_CACHE = os.path.join(BASE_DIR, "bm25_index.json")

COLLECTION_NAME = "novel_chunks_v4"

# ---- 分块参数（父子检索）----
MIN_CHUNK_SIZE = 100      # 小块最小字数（低于此值合并）
MAX_CHUNK_SIZE = 300      # 小块最大字数（用于精确检索）
PARENT_RANGE = 1          # 父上下文：命中块前后各取 PARENT_RANGE 个邻块
                          # 缩减到±1，减少噪声，约600-900字上下文

# ---- 检索参数 ----
TOP_K_VECTOR = 20         # 向量检索召回数
TOP_K_KEYWORD = 20        # 关键词检索召回数
RRF_K = 60                # RRF融合常数
FINAL_K = 5               # LLM重排后保留用于生成的块数

# ---- CPU限速 ----
CPU_TARGET = 85            # 目标CPU占用上限(%)
CPU_CHECK_INTERVAL = 10    # 每N批检查一次CPU
CPU_SLEEP_MIN = 0.05       # 最小休眠(秒)
CPU_SLEEP_MAX = 3.0        # 最大休眠(秒)

# ---- 嵌入批处理 ----
BATCH_SIZE = 5             # 每批嵌入块数（降低以减少单次负载）

# ---- 场景转折词（保留v3逻辑）----
SCENE_BREAK_WORDS = [
    "与此同时", "另一边", "镜头一转", "画面一转", "再说",
    "此时", "此刻", "不一会儿", "不久之后", "片刻之后",
    "与此同时", "另一方面", "转眼", "霎时间", "忽然",
    "突然", "蓦地", "就在这时",
]


# ============================================================
# CPU 监控与限速
# ============================================================
class CPUMonitor:
    """实时监控CPU占用，自动调节休眠时间"""

    def __init__(self, target: int = 85):
        self.target = target
        self.sleep_time = 0.0
        self.check_count = 0

    def check(self) -> float:
        """返回当前CPU占用率"""
        return psutil.cpu_percent(interval=0.3)

    def regulate(self):
        """根据CPU占用自动调节休眠时间"""
        self.check_count += 1
        if self.check_count % CPU_CHECK_INTERVAL != 0:
            return

        cpu = self.check()
        if cpu > self.target + 5:
            # CPU太高，增加休眠
            self.sleep_time = min(CPU_SLEEP_MAX, self.sleep_time + 0.2)
        elif cpu > self.target:
            self.sleep_time = min(CPU_SLEEP_MAX, self.sleep_time + 0.1)
        elif cpu < self.target - 10:
            # CPU有余量，减少休眠
            self.sleep_time = max(CPU_SLEEP_MIN, self.sleep_time - 0.15)
        elif cpu < self.target - 5:
            self.sleep_time = max(CPU_SLEEP_MIN, self.sleep_time - 0.05)

        if self.sleep_time > 0:
            time.sleep(self.sleep_time)

    def status(self) -> str:
        cpu = self.check()
        return f"CPU: {cpu:.0f}% | 休眠: {self.sleep_time:.1f}s"


# ============================================================
# API 封装
# ============================================================
def ollama_embed(text: str) -> List[float]:
    """调用Ollama生成嵌入向量，带重试"""
    if len(text) > 600:
        text = text[:600]  # nomic-embed-text 上下文限制安全截断

    for attempt in range(3):
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=120,
            )
            if resp.status_code == 200:
                return resp.json()["embedding"]
            if resp.status_code >= 500:
                time.sleep(3)
                continue
            resp.raise_for_status()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3)
    raise RuntimeError("Ollama embed failed after 3 retries")


def deepseek_chat(prompt: str, system: str = "", max_tokens: int = 1024) -> str:
    """调用DeepSeek API进行文本生成"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(
        f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek API error: {resp.status_code} {resp.text[:200]}")
    return resp.json()["choices"][0]["message"]["content"]


# ============================================================
# 结构分块（v4 小块版）
# ============================================================
def extract_chapter_title(lines: List[str]) -> str:
    """从文件开头提取章节名"""
    candidates = []
    for line in lines[:10]:
        line = line.strip()
        if not line:
            continue
        if re.match(r"^第[一二三四五六七八九十百千\d]+章", line):
            if line not in candidates:
                candidates.append(line)
    return candidates[0] if candidates else "未知"


def is_scene_break(line: str) -> bool:
    """判断是否为场景转折点"""
    for word in SCENE_BREAK_WORDS:
        if line.strip().startswith(word):
            return True
    return False


def split_text_structural_small(text: str, filename: str) -> List[dict]:
    """
    结构分块 v4 —— 小块版（200-300字）
    用于精确检索，后续通过父上下文扩展补全
    """
    raw_lines = text.split("\n")
    chapter_title = extract_chapter_title(raw_lines)

    # 去重章节名
    seen_title = False
    clean_lines = []
    for i, line in enumerate(raw_lines):
        stripped = line.strip()
        if i < 10 and stripped == chapter_title:
            if seen_title:
                continue
            seen_title = True
        clean_lines.append(stripped)

    # 合并：忽略空行，在转折词处切
    merged = []
    current = ""
    for line in clean_lines:
        if not line:
            continue

        # 场景转折词触发切割（当前已有足够内容时）
        if is_scene_break(line) and len(current) >= MIN_CHUNK_SIZE:
            merged.append(current.strip())
            current = line
            continue

        if current:
            current += line
        else:
            current = line

        # 长度超限：在句号/问号/感叹号处找切点
        if len(current) >= MAX_CHUNK_SIZE:
            cut_at = -1
            for j in range(len(current) - 1, max(0, len(current) - 150), -1):
                if current[j] in "。？！？!…" and len(current[:j + 1]) >= MIN_CHUNK_SIZE:
                    cut_at = j + 1
                    break
            if cut_at > 0:
                merged.append(current[:cut_at].strip())
                current = current[cut_at:].strip()
            else:
                merged.append(current[:MAX_CHUNK_SIZE].strip())
                current = current[MAX_CHUNK_SIZE:].strip()

    if current.strip():
        merged.append(current.strip())

    # 合并太短的块
    final = []
    buffer = ""
    for m in merged:
        if len(buffer) + len(m) < MAX_CHUNK_SIZE:
            buffer += m
        else:
            if buffer and len(buffer) >= MIN_CHUNK_SIZE:
                final.append(buffer.strip())
                buffer = m
            else:
                buffer += m
    if buffer:
        if len(buffer) < MIN_CHUNK_SIZE and final:
            final[-1] = (final[-1] + buffer).strip()
        else:
            final.append(buffer.strip())

    # 构建输出
    result = []
    for i, chunk in enumerate(final):
        result.append({
            "text": chunk,
            "filename": filename,
            "chunk_index": i,
            "chapter": chapter_title,
            "char_count": len(chunk),
        })
    return result


# ============================================================
# TF-IDF 关键词检索器
# ============================================================
class KeywordRetriever:
    """
    轻量级 TF-IDF 关键词检索
    - 对中文做字符级别 unigram + bigram 分词
    - 查询时扫描匹配文档，计算 TF-IDF 得分
    - 适用于专有名词、章节号等向量检索不擅长的场景
    """

    def __init__(self):
        self.doc_texts: Dict[str, str] = {}          # {doc_id: text}
        self.doc_metas: Dict[str, dict] = {}         # {doc_id: metadata}
        self.doc_lengths: Dict[str, int] = {}        # {doc_id: char_count}
        self.term_df: Dict[str, int] = {}            # {term: document_frequency}
        self.N: int = 0                              # 总文档数

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """中文分词：字符unigram + bigram，过滤标点"""
        # 去标点和空白
        clean = re.sub(r'[\s，。！？；：、""''（）《》【】…—　·]', '', text)
        if not clean:
            return []
        # unigrams
        tokens = list(clean)
        # bigrams（相邻字符对）
        tokens += [clean[i] + clean[i + 1] for i in range(len(clean) - 1)]
        return tokens

    def index(self, doc_id: str, text: str, metadata: dict):
        """向索引中添加一个文档"""
        tokens = self._tokenize(text)
        unique_terms = set(tokens)

        self.doc_texts[doc_id] = text
        self.doc_metas[doc_id] = metadata
        self.doc_lengths[doc_id] = len(text)
        self.N += 1

        for term in unique_terms:
            self.term_df[term] = self.term_df.get(term, 0) + 1

    def search(self, query: str, top_k: int = TOP_K_KEYWORD) -> List[Tuple[str, float]]:
        """
        搜索：提取查询词 → 全量扫描匹配文档 → 计算TF-IDF得分 → 返回top_k
        返回: [(doc_id, score), ...]
        """
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        # 过滤掉不在索引中的term
        active_terms = [t for t in query_terms if t in self.term_df]
        if not active_terms:
            return []

        # 预计算IDF
        idf_map = {}
        for term in active_terms:
            df = self.term_df[term]
            idf_map[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

        # 单次全量扫描：对所有文档同时检查所有query term
        # 18,559文档 × 300字 ≈ 5.6MB文本扫描，<100ms
        scores: Dict[str, float] = defaultdict(float)
        avgdl = sum(self.doc_lengths.values()) / max(self.N, 1)
        k1, b = 1.5, 0.75

        for doc_id, text in self.doc_texts.items():
            doc_score = 0.0
            doc_len = self.doc_lengths[doc_id]
            for term in active_terms:
                tf = text.count(term)
                if tf > 0:
                    doc_score += idf_map[term] * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
            if doc_score > 0:
                scores[doc_id] = doc_score

        if not scores:
            return []

        # 排序返回
        sorted_docs = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        # 归一化到0-1
        max_score = sorted_docs[0][1]
        sorted_docs = [(did, s / max_score) for did, s in sorted_docs]

        return sorted_docs

    def get_text(self, doc_id: str) -> str:
        return self.doc_texts.get(doc_id, "")

    def get_meta(self, doc_id: str) -> dict:
        return self.doc_metas.get(doc_id, {})

    def save(self, path: str):
        """持久化为JSON"""
        data = {
            "N": self.N,
            "term_df": self.term_df,
            "doc_lengths": self.doc_lengths,
            "doc_texts": self.doc_texts,
            "doc_metas": self.doc_metas,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        import os as _os
        size_mb = _os.path.getsize(path) / (1024 * 1024)
        print(f"[BM25] 索引已保存: {path} ({size_mb:.1f}MB)")

    def load(self, path: str):
        """从JSON加载"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.N = data["N"]
        self.term_df = data["term_df"]
        self.doc_lengths = data["doc_lengths"]
        self.doc_texts = data["doc_texts"]
        self.doc_metas = data["doc_metas"]
        print(f"[BM25] 索引已加载: {self.N} 个文档")

    def remove_docs_by_filename(self, filename: str):
        """删除指定文件的所有文档（用于force_rebuild）"""
        to_remove = [did for did, meta in self.doc_metas.items()
                     if meta.get("filename") == filename]
        for did in to_remove:
            # 更新 term_df
            if did in self.doc_texts:
                tokens = set(self._tokenize(self.doc_texts[did]))
                for term in tokens:
                    if term in self.term_df:
                        self.term_df[term] -= 1
                        if self.term_df[term] <= 0:
                            del self.term_df[term]
            self.doc_texts.pop(did, None)
            self.doc_metas.pop(did, None)
            self.doc_lengths.pop(did, None)
            self.N -= 1


# ============================================================
# RRF 融合
# ============================================================
def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = RRF_K,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion：合并多个排序列表
    每路结果的得分 = 1/(k + rank)，rank从1开始
    """
    rrf_scores: Dict[str, float] = defaultdict(float)

    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: -x[1])


# ============================================================
# Ollama 嵌入函数适配 ChromaDB
# ============================================================
class OllamaEmbeddingFunction:
    """ChromaDB 自定义嵌入函数，基于 Ollama"""

    def name(self) -> str:
        return f"ollama-{EMBED_MODEL}"

    def __call__(self, input):
        if isinstance(input, str):
            return [ollama_embed(input)]
        return [ollama_embed(text) for text in input]

    def embed_query(self, input: str) -> List[float]:
        return ollama_embed(input)

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return [ollama_embed(text) for text in input]


# ============================================================
# RAG 主逻辑 v4
# ============================================================
class NovelRAG:
    def __init__(self, novel_dir: str, chroma_dir: str, bm25_path: str):
        self.novel_dir = novel_dir
        self.chroma_dir = chroma_dir
        self.bm25_path = bm25_path

        # ChromaDB
        self.client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.embed_fn = OllamaEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embed_fn,
            metadata={"description": "RAG v4 — 父子检索 + 双路召回"},
        )

        # TF-IDF关键词检索
        self.keyword = KeywordRetriever()
        if os.path.exists(bm25_path):
            self.keyword.load(bm25_path)

        # CPU监控
        self.cpu_monitor = CPUMonitor(target=CPU_TARGET)

    # ============================================================
    # 构建索引
    # ============================================================
    def build_index(self, force_rebuild: bool = False):
        os.makedirs(self.novel_dir, exist_ok=True)
        txt_files = sorted([f for f in os.listdir(self.novel_dir) if f.endswith(".txt")])
        if not txt_files:
            print(f"[RAG] 目录 {self.novel_dir} 中没有 txt 文件")
            return

        print(f"[RAG] 扫描到 {len(txt_files)} 个文件")

        # 强制重建
        if force_rebuild:
            print("[RAG] 强制重建，清空所有旧数据...")
            try:
                self.client.delete_collection(COLLECTION_NAME)
            except:
                pass
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embed_fn,
            )
            self.keyword = KeywordRetriever()
            if os.path.exists(self.bm25_path):
                os.remove(self.bm25_path)

        # 增量检测：找出未索引的文件
        existing_files = set()
        if self.collection.count() > 0 and not force_rebuild:
            try:
                metas = self.collection.get()["metadatas"]
                existing_files = {m["filename"] for m in metas if m}
            except:
                pass

        new_files = [f for f in txt_files if f not in existing_files]
        if not new_files:
            print(f"[RAG] 所有文件已索引（共 {self.collection.count()} 块）")
            return

        print(f"[RAG] 新文件: {len(new_files)} 个 | 已索引: {len(existing_files)} 个")
        print(f"[RAG] 分块: {MIN_CHUNK_SIZE}-{MAX_CHUNK_SIZE}字/块 "
              f"| 父上下文范围: ±{PARENT_RANGE}块")

        total_chunks = 0
        t_start = time.time()

        for fi, filename in enumerate(new_files):
            filepath = os.path.join(self.novel_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            # 结构分块（小块）
            chunks = split_text_structural_small(text, filename)
            if not chunks:
                continue

            # 分批嵌入
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i + BATCH_SIZE]

                ids = [f"{filename}_{c['chunk_index']}" for c in batch]
                texts = [c["text"] for c in batch]
                metadatas = [{
                    "filename": c["filename"],
                    "chunk_index": c["chunk_index"],
                    "chapter": c["chapter"],
                    "char_count": c["char_count"],
                    "total_chunks": len(chunks),  # 新增：记录本章总块数
                } for c in batch]

                # 写入 ChromaDB
                self.collection.add(ids=ids, documents=texts, metadatas=metadatas)

                # 写入关键词索引
                for cid, ctext, cmeta in zip(ids, texts, metadatas):
                    self.keyword.index(cid, ctext, cmeta)

                # CPU限速
                self.cpu_monitor.regulate()

                # 进度
                pct = (i + len(batch)) / len(chunks) * 100
                cpu_info = self.cpu_monitor.status()
                print(f"\r  {fi + 1}/{len(new_files)} {filename[:30]} "
                      f"[{min(i + len(batch), len(chunks))}/{len(chunks)}] "
                      f"{pct:.0f}% | {cpu_info}",
                      end="")

            total_chunks += len(chunks)
            print()

        elapsed = time.time() - t_start
        print(f"\n[RAG] 本次导入: {len(new_files)} 文件, {total_chunks} 块, "
              f"耗时 {elapsed / 60:.0f}分")

        # 保存关键词索引
        self.keyword.save(self.bm25_path)

        print(f"[RAG] 数据库总计: {self.collection.count()} 块")
        print(f"[RAG] 关键词索引: {self.keyword.N} 文档")

    # ============================================================
    # 父上下文扩展
    # ============================================================
    def _expand_to_parent(self, doc_id: str) -> Tuple[str, dict]:
        """
        根据小块ID找到同章的邻块，拼接为父上下文
        返回: (parent_text, merged_metadata)
        """
        # 解析ID: "0001_第1章 陆尘.txt_5"
        parts = doc_id.rsplit("_", 1)
        if len(parts) != 2:
            return self.keyword.get_text(doc_id), self.keyword.get_meta(doc_id)

        filename = parts[0]
        try:
            center_idx = int(parts[1])
        except ValueError:
            return self.keyword.get_text(doc_id), self.keyword.get_meta(doc_id)

        # 取邻块
        neighbor_texts = []
        neighbor_metas = []
        for offset in range(-PARENT_RANGE, PARENT_RANGE + 1):
            idx = center_idx + offset
            if idx < 0:
                continue
            neighbor_id = f"{filename}_{idx}"
            text = self.keyword.get_text(neighbor_id)
            if text:
                neighbor_texts.append(text)
                meta = self.keyword.get_meta(neighbor_id)
                if meta:
                    neighbor_metas.append(meta)

        if not neighbor_texts:
            return self.keyword.get_text(doc_id), self.keyword.get_meta(doc_id)

        parent_text = "\n".join(neighbor_texts)
        merged_meta = {
            "filename": filename,
            "chunk_range": f"{center_idx - PARENT_RANGE}-{center_idx + PARENT_RANGE}",
            "neighbor_count": len(neighbor_texts),
            "total_chars": len(parent_text),
            "chapter": neighbor_metas[0].get("chapter", "?") if neighbor_metas else "?",
        }
        return parent_text, merged_meta

    # ============================================================
    # LLM 重排序
    # ============================================================
    def _llm_rerank(self, question: str, candidates: List[Tuple[str, float, str, dict]]
                     ) -> List[Tuple[str, str, dict]]:
        """
        让DeepSeek从候选块中选出最相关的 FINAL_K 个
        candidates: [(doc_id, rrf_score, parent_text, merged_meta), ...]
        返回: [(parent_text, chapter, meta), ...]
        """
        if len(candidates) <= FINAL_K:
            return [(t, m.get("chapter", "?"), m) for _, _, t, m in candidates]

        # 构建排序请求
        chunks_desc = []
        for idx, (doc_id, score, text, meta) in enumerate(candidates):
            preview = text[:200].replace("\n", " ")
            ch = meta.get("chapter", "?")
            chunks_desc.append(
                f"[{idx}] 章节={ch} | 字数={len(text)} | 融合分={score:.4f}\n"
                f"    内容预览: {preview}..."
            )

        prompt = f"""以下是小说搜索返回的 {len(candidates)} 个候选片段。
用户问题是：「{question}」

请选出与问题最相关的 {FINAL_K} 个片段，返回编号列表（如: 3,7,12,5,9）。
只返回编号，用逗号分隔，不要解释。

候选片段：
{chr(10).join(chunks_desc)}"""

        try:
            response = deepseek_chat(prompt, system="你是文本相关性判断助手。只返回编号。",
                                     max_tokens=50)
            # 解析编号
            nums = [int(n.strip()) for n in re.findall(r'\d+', response)]
            selected = [candidates[n] for n in nums if 0 <= n < len(candidates)]

            if len(selected) >= 2:
                print(f"[Rerank] DeepSeek 从 {len(candidates)} 个中选了 {len(selected)} 个")
                return [(t, m.get("chapter", "?"), m)
                        for _, _, t, m in selected[:FINAL_K]]
        except Exception as e:
            print(f"[Rerank] DeepSeek排序失败 ({e})，回退到RRF排序")

        # 降级：直接按RRF分数取前FINAL_K
        return [(t, m.get("chapter", "?"), m)
                for _, _, t, m in candidates[:FINAL_K]]

    # ============================================================
    # 查询预处理（查询改写 + HyDE + 章节检测）
    # ============================================================
    def _preprocess_query(self, question: str) -> Tuple[str, str, List[int]]:
        """
        一次 API 调用完成三件事：
        1. 查询改写：口语问题 → 搜索关键词（匹配小说原文风格）
        2. HyDE：生成假设答案（风格更接近原文，向量检索更准）
        3. 章节检测：提取问题中提到的章节号
        返回: (rewritten_query, hyde_answer, chapter_numbers)
        """
        prompt = f"""你是一个小说搜索助手。用户正在搜索一部中国网文小说。
用户问题：{question}

请完成以下三个任务，严格按照格式输出：

1. 查询改写：把问题改写成适合搜索的关键词，覆盖人物名、地名、事件、修为、法器、关系等。输出一行，以"搜索词："开头。

2. 假设回答：猜一个可能的答案，用小说原文风格的文字写3-5句话。输出一行，以"假设回答："开头。

3. 章节号：如果问题中提到了具体章节号（如"第3章""第500章"），提取出来。输出一行，以"章节："开头，多个用逗号分隔。没有则输出"无"。

示例：
搜索词：陆江仙 来历 修为 师承 法器 身份 前世
假设回答：陆江仙乃是玄鉴元府中的重要人物，曾获青霜剑认主，修为深不可测，前世与太阴太阳有关。
章节：无"""

        try:
            response = deepseek_chat(prompt, max_tokens=300)
        except Exception as e:
            print(f"[预处理] DeepSeek调用失败 ({e})，使用原始问题")
            return question, question, []

        # 解析响应
        rewritten = question
        hyde = question
        chapters = []

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("搜索词：") or line.startswith("搜索词:"):
                rewritten = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("假设回答：") or line.startswith("假设回答:"):
                hyde = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("章节：") or line.startswith("章节:"):
                ch_str = line.split("：", 1)[-1].split(":", 1)[-1].strip()
                if ch_str and ch_str != "无":
                    chapters = re.findall(r"\d+", ch_str)

        if rewritten and rewritten != question:
            print(f"[预处理] 搜索词: {rewritten[:80]}...")
        if hyde and hyde != question:
            print(f"[预处理] HyDE: {hyde[:80]}...")
        if chapters:
            print(f"[预处理] 章节过滤: {chapters}")

        return rewritten, hyde, [int(c) for c in chapters]

    # ============================================================
    # 查询
    # ============================================================
    def query(self, question: str) -> str:
        t0 = time.time()

        # ========== 第0步：查询预处理 ==========
        rewritten, hyde_answer, chapter_nums = self._preprocess_query(question)

        # ========== 第1步：多路召回 ==========
        # 路1：向量检索 — 用 HyDE 假设答案（风格更接近原文）
        hyde_vec = ollama_embed(hyde_answer)
        vector_results = self.collection.query(
            query_embeddings=[hyde_vec],
            n_results=TOP_K_VECTOR,
        )
        vector_ids = vector_results["ids"][0]
        vector_distances = vector_results.get("distances", [[]])[0]

        # 路2：关键词检索 — 用改写后的搜索词（覆盖更多关键词）
        keyword_results = self.keyword.search(rewritten, top_k=TOP_K_KEYWORD)

        # 路3（额外）：原始问题也做一次关键词检索，捕捉改写可能遗漏的词
        if rewritten != question:
            extra_keyword = self.keyword.search(question, top_k=TOP_K_KEYWORD // 2)
        else:
            extra_keyword = []

        total_kw = len(keyword_results) + len(extra_keyword)
        print(f"[检索] 向量(HyDE): {len(vector_ids)} | "
              f"关键词(改写): {len(keyword_results)} | 关键词(原文): {len(extra_keyword)}")

        # ========== 第2步：RRF多路融合 ==========
        vector_ranked = []
        if vector_distances:
            max_dist = max(vector_distances) if vector_distances else 1
            for did, dist in zip(vector_ids, vector_distances):
                sim = 1.0 - (dist / max(max_dist * 1.2, 1))
                vector_ranked.append((did, max(0, sim)))

        fused = reciprocal_rank_fusion([
            vector_ranked,
            [(did, score) for did, score in keyword_results],
            [(did, score) for did, score in extra_keyword],
        ])
        print(f"[RRF] 三路融合: {len(fused)} 候选")

        # ========== 第3步：章节主动注入 + 过滤 ==========
        if chapter_nums:
            # 主动扫描所有文档，找到章节号匹配的，直接注入候选列表
            injected = []
            for doc_id, meta in self.keyword.doc_metas.items():
                chapter_name = meta.get("chapter", "")
                ch_match = re.search(r"第\s*(\d+)\s*章", chapter_name)
                if ch_match:
                    doc_chapter = int(ch_match.group(1))
                    for ch in chapter_nums:
                        if abs(doc_chapter - ch) <= 1:  # 精确匹配±1章
                            injected.append((doc_id, 0.5))  # 给一个中等分数
                            break

            if injected:
                print(f"[章节注入] {chapter_nums} → 主动注入 {len(injected)} 个章节匹配块")
                fused = injected + fused  # 注入的排前面

            # 过滤：只保留章节匹配的候选（注入+原有匹配的）
            filtered = []
            for doc_id, rrf_score in fused:
                meta = self.keyword.get_meta(doc_id)
                chapter_name = meta.get("chapter", "")
                ch_match = re.search(r"第\s*(\d+)\s*章", chapter_name)
                if ch_match:
                    doc_chapter = int(ch_match.group(1))
                    for ch in chapter_nums:
                        if abs(doc_chapter - ch) <= 1:
                            filtered.append((doc_id, rrf_score))
                            break
            if filtered:
                fused = filtered
                print(f"[章节过滤] 最终 {len(fused)} 候选")
            else:
                print(f"[章节过滤] 未找到匹配，保留全部")

        # ========== 第4步：父上下文扩展（±1块） ==========
        expanded = []
        seen_texts = set()

        for doc_id, rrf_score in fused:
            parent_text, merged_meta = self._expand_to_parent(doc_id)
            text_key = parent_text[:50]
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)
            expanded.append((doc_id, rrf_score, parent_text, merged_meta))

        print(f"[父上下文] 扩展去重: {len(expanded)} 个 (±{PARENT_RANGE}块)")

        # ========== 第5步：LLM重排序 ==========
        ranked = self._llm_rerank(question, expanded)

        # ========== 第6步：构建上下文 + 生成回答 ==========
        context_parts = []
        total_chars = 0
        for idx, (text, chapter, meta) in enumerate(ranked):
            context_parts.append(
                f"[片段{idx + 1} | {chapter} | {meta.get('char_count', len(text))}字]\n{text}"
            )
            total_chars += len(text)

        context = "\n\n---\n\n".join(context_parts)
        print(f"[上下文] {len(ranked)} 块 | 约 {total_chars} 字")

        system_prompt = """你是一个小说阅读助手。请根据下面提供的小说片段回答问题。
规则：
1. 只根据提供的小说片段回答，不要编造
2. 引用原文中的具体细节
3. 如果片段中没有相关信息，说"根据现有片段无法确认"
4. 用中文，简洁准确"""

        user_prompt = f"""以下是小说的相关片段：
{context}
---
问题：{question}
请根据以上片段回答。"""

        print(f"[RAG] DeepSeek 生成中...")
        answer = deepseek_chat(user_prompt, system=system_prompt)

        elapsed = time.time() - t0
        print(f"[RAG] 查询完成 ({elapsed:.1f}s)")

        return answer

    # ============================================================
    # 统计信息
    # ============================================================
    def stats(self):
        c = self.collection.count()
        print(f"\n  ChromaDB: {CHROMA_DB_DIR}")
        print(f"  向量块数: {c}")
        print(f"  关键词索引: {self.keyword.N} 文档")
        print(f"  词汇量: {len(self.keyword.term_df)}")

        if c > 0:
            try:
                metas = self.collection.get(limit=10000)["metadatas"]
                files = set(m["filename"] for m in metas if m)
                sizes = [m["char_count"] for m in metas if m and m.get("char_count")]
                print(f"  文件数: {len(files)}")
                if sizes:
                    print(f"  块大小: 最小{min(sizes)} 中位数{sorted(sizes)[len(sizes)//2]} "
                          f"最大{max(sizes)} 平均{sum(sizes)//len(sizes)}")
                # 前10个文件
                file_counts = Counter(m["filename"] for m in metas if m)
                print(f"  文件示例 (前10):")
                for fn, cnt in sorted(file_counts.items())[:10]:
                    print(f"    {fn}: {cnt} 块")
                if len(file_counts) > 10:
                    print(f"    ... 共 {len(file_counts)} 个文件")
            except Exception as e:
                print(f"  (统计出错: {e})")


# ============================================================
# 交互界面
# ============================================================
def main():
    print("=" * 60)
    print("  小说 RAG 问答系统 v4")
    print("  父子检索 + 双路召回 + RRF + LLM重排 + CPU限速")
    print("=" * 60)

    rag = NovelRAG(NOVEL_DIR, CHROMA_DB_DIR, BM25_CACHE)
    rag.build_index()
    rag.stats()

    print("\n  命令: stats | rebuild | quit")
    print("=" * 60 + "\n")

    while True:
        try:
            q = input("你的问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not q:
            continue
        if q in ("quit", "exit", "q"):
            break
        if q == "stats":
            rag.stats()
            continue
        if q == "rebuild":
            rag.build_index(force_rebuild=True)
            continue
        try:
            ans = rag.query(q)
            print(f"\n回答:\n{ans}\n" + "-" * 60 + "\n")
        except Exception as e:
            import traceback
            print(f"出错: {e}")
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()
