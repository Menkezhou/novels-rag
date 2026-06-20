# 玄鉴仙族 RAG 问答系统

基于 RAG（检索增强生成）的小说智能问答系统。将长篇小说分块嵌入为向量，查询时自动检索相关片段，交给大模型生成精准回答。

**适用场景**：任何长文本（小说、技术文档、法律合同）的智能问答。

## 特性

- **父子检索**：小块嵌入（200-300字）保证检索精度，邻块扩展保证上下文完整
- **双路召回**：向量语义搜 + TF-IDF 关键词搜，RRF 融合，互补盲区
- **查询预处理**：自动改写查询 + HyDE 假设答案 + 章节号检测
- **LLM 重排序**：TOP-20 粗排 → DeepSeek 精排 TOP-5
- **增量导入**：支持断点续跑，新增文件自动识别
- **CPU 限速**：psutil 自监控，嵌入时不占满 CPU

## 架构

```
小说目录 → 结构分块 → Ollama嵌入 → ChromaDB + BM25双索引 → 查询
                                              ↓
用户问题 → 预处理(改写+HyDE) → 三路RRF检索 → LLM重排 → DeepSeek生成
```

## 硬件要求

- 嵌入服务：Ollama（推荐 4GB+ 内存）
- 向量数据库：ChromaDB（本地 SQLite，零配置）
- 生成服务：DeepSeek API（云端，不占本地资源）
- 实测可运行环境：树莓派 5（4GB）+ Ubuntu Server

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 安装 Ollama 并拉取嵌入模型

```bash
# 安装 Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 拉取嵌入模型（推荐 bge-m3 中文最强，或 nomic-embed-text 节省资源）
ollama pull nomic-embed-text
# ollama pull bge-m3  # 中文更好但需要更多内存
```

### 3. 设置环境变量

```bash
export DEEPSEEK_API_KEY="sk-xxxxxxxx"   # 必填，DeepSeek API Key
export DEEPSEEK_BASE_URL="https://api.deepseek.com"  # 可选
export DEEPSEEK_MODEL="deepseek-chat"   # 可选
export OLLAMA_URL="http://localhost:1111"  # 可选
export EMBED_MODEL="nomic-embed-text"   # 可选，或用 bge-m3
```

### 4. 准备小说文件

将 txt 文件放入 `novels/` 目录，每章一个文件：

```
novels/
├── 0001_第1章 开篇.txt
├── 0002_第2章 初遇.txt
├── 0003_第3章 比试.txt
└── ...
```

文件命名建议：`序号_章节名.txt`，程序会自动识别章节边界。

### 5. 构建索引

```bash
python3 -u rag_novel_v4.py
```

首次运行会自动：
1. 扫描 `novels/` 目录
2. 将每章切成 200-300 字小块
3. 通过 Ollama 嵌入为向量
4. 存入 ChromaDB 向量库
5. 构建 BM25 关键词索引
6. 保存索引到磁盘

之后重新运行会自动跳过已索引文件（增量导入）。

限制 CPU 占用：

```bash
# 只用 3 个核给 Ollama，留 1 个核给系统
OLLAMA_NUM_THREAD=3 python3 -u rag_novel_v4.py
```

### 6. 开始提问

构建完成后自动进入交互界面：

```
你的问题: 陆江仙是谁
你的问题: 第500章讲了什么
你的问题: stats       # 查看索引统计
你的问题: rebuild     # 强制重建
你的问题: quit        # 退出
```

## 配置说明

| 参数 | 默认值 | 说明 |
|------|------|------|
| `MIN_CHUNK_SIZE` | 100 | 最小块大小（字） |
| `MAX_CHUNK_SIZE` | 300 | 最大块大小（字） |
| `PARENT_RANGE` | 1 | 父上下文邻块范围 |
| `TOP_K_VECTOR` | 20 | 向量召回数 |
| `TOP_K_KEYWORD` | 20 | 关键词召回数 |
| `FINAL_K` | 5 | 最终返回给 LLM 的块数 |

## 准确率优化

本项目在不换嵌入模型的前提下，通过以下纯代码优化大幅提升准确率：

| 优化 | 方法 | 代价 |
|------|------|------|
| 父子检索 | 小块搜 + 大块答 | 重建DB |
| 查询改写 | 口语 → 搜索关键词 | 1次 API |
| HyDE | 编假设答案去搜 | 0（复用） |
| BM25 关键词 | 专有名词精确匹配 | 29MB 索引 |
| LLM 重排序 | TOP-20 → 精挑 5 个 | 1次 API |
| 章节注入 | 检测章节号主动注入 | 0 |

## 项目结构

```
.
├── rag_novel_v4.py      # 主脚本
├── requirements.txt      # Python依赖
├── novels/               # 小说源文件（不提交）
├── chroma_db/            # 向量数据库（不提交）
├── bm25_index.json       # 关键词索引（不提交）
├── .gitignore
└── README.md
```

## 常见问题

**Q: 检索不准确？**
换用 bge-m3 嵌入模型：`export EMBED_MODEL=bge-m3`，然后 `rebuild`。

**Q: 支持其他生成模型？**
修改 `deepseek_chat()` 函数即可对接任意 OpenAI 兼容 API。

## License

MIT
