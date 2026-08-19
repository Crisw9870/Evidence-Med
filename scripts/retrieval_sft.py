"""
Retrieval-Augmented SFT Pipeline

从 C-Eval dev 题目出发，用 BGE-M3 + FAISS 从 SFT 数据中召回最相关的训练样本，
生成 retrieval_sft.jsonl 用于后续 LoRA SFT 训练。

用法:
    python scripts/retrieval_sft.py \
        --sample_size 100000 \
        --top_k 50 \
        --model_path ./bge-m3 \
        --ceval_dir ./data/ceval \
        --sft_dir ./data/sft \
        --output_dir ./data/sft_retrieval
"""
import torch
import argparse
import json
import os
import random
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_ceval(ceval_dir: str, subjects: list[str]) -> list[dict]:
    """加载 C-Eval val 集，构造查询文本"""
    queries = []
    for subject in subjects:
        dev_path = os.path.join(ceval_dir, subject, "val-00000-of-00001.parquet")
        if not os.path.exists(dev_path):
            print(f"[WARN] dev file not found: {dev_path}, skipping")
            continue
        df = pd.read_parquet(dev_path)
        print(f"  {subject}: {len(df)} dev samples")
        for _, row in df.iterrows():
            query = (f"{row['question']}\n")
            # query = (
            #     f"{row['question']}\n"
            #     f"A.{row['A']}\n"
            #     f"B.{row['B']}\n"
            #     f"C.{row['C']}\n"
            #     f"D.{row['D']}"
            # )
            queries.append({
                "subject": subject,
                "id": row["id"],
                "query": query,
            })
    print(f"Total queries: {len(queries)}")
    return queries


def load_sft_candidates(sft_dir: str, sample_size: int) -> list[dict]:
    """加载 SFT 数据，构造候选文本，随机抽样"""
    all_data = []
    jsonl_files = sorted(Path(sft_dir).glob("**/*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files in {sft_dir}")

    for fpath in jsonl_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                conversations = item.get("conversations", [])
                if len(conversations) < 2:
                    continue
                # 提取 human 和 gpt 的文本
                human_text = ""
                gpt_text = ""
                for turn in conversations:
                    if turn.get("from") in ("human", "user") and not human_text:
                        human_text = turn.get("value", "")
                    elif turn.get("from") in ("gpt", "assistant") and not gpt_text:
                        gpt_text = turn.get("value", "")
                if not human_text or not gpt_text:
                    continue
                # candidate_text = human_text + "\n" + gpt_text
                candidate_text = human_text  # 只用 human 部分作为检索文本
                all_data.append({
                    "text": candidate_text,
                    "original": item,  # 保留原始数据用于输出
                })

    print(f"Total SFT candidates loaded: {len(all_data)}")

    if sample_size > 0 and sample_size < len(all_data):
        all_data = random.sample(all_data, sample_size)
        print(f"Sampled: {len(all_data)}")

    return all_data


def encode_texts(model, texts, batch_size=32, desc="Encoding", mmap_path=None):
    """编码文本，支持 memmap 直写磁盘避免内存爆炸"""
    n = len(texts)
    total_batches = (n + batch_size - 1) // batch_size

    # 先跑一个 batch 拿到 embedding 维度
    with torch.inference_mode():
        probe = model.encode(texts[:1], convert_to_numpy=True)
    dim = probe.shape[1]

    if mmap_path:
        # memmap: 零拷贝，直接写磁盘，不占内存
        embs = np.memmap(mmap_path, dtype="float32", mode="w+", shape=(n, dim))
    else:
        embs = np.zeros((n, dim), dtype="float32")

    for i in tqdm(range(0, n, batch_size), desc=desc, total=total_batches):
        batch = texts[i:i + batch_size]
        end = min(i + batch_size, n)

        with torch.inference_mode():
            batch_embs = model.encode(
                batch,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        embs[i:end] = np.asarray(batch_embs, dtype="float32")

    if mmap_path:
        embs.flush()
    return embs


def _l2_normalize(arr):
    """numpy L2 归一化，避免 faiss.normalize_L2 对 memmap 段错误"""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return arr / norms


def build_faiss_index(embeddings, batch_size=50000):
    """分批构建 FAISS 索引，避免一次性拷贝大数组导致 OOM"""
    n, dim = embeddings.shape
    index = faiss.IndexFlatIP(dim)

    for start in tqdm(range(0, n, batch_size), desc="Building FAISS index"):
        end = min(start + batch_size, n)
        batch = np.array(embeddings[start:end], dtype="float32")  # 拷贝到内存
        batch = _l2_normalize(batch)
        index.add(batch)

    return index



def retrieve_topk(index, query_emb, candidates, top_k, dedup=True):

    query_emb = _l2_normalize(query_emb)
    D, I = index.search(query_emb, top_k)

    results = []
    # 去重时用 dict 保留每个候选的最高分
    best_per_idx = {}

    for qi in range(len(I)):
        seen_local = set()

        for rank, idx in enumerate(I[qi]):
            if idx < 0:
                continue
            if idx in seen_local:
                continue
            seen_local.add(idx)

            score = float(D[qi][rank])

            if dedup:
                if idx not in best_per_idx or score > best_per_idx[idx]["score"]:
                    best_per_idx[idx] = {
                        "original": candidates[idx]["original"],
                        "score": score,
                    }
            else:
                results.append({
                    "original": candidates[idx]["original"],
                    "score": score,
                })

    if dedup:
        results = list(best_per_idx.values())

    print(f"[RETRIEVE] {len(results)} samples (dedup={dedup})")
    return results


def quality_filter(results, max_samples=8000, min_answer_len=10):
    """按相似度分数排序 + 质量过滤，取 top max_samples"""
    # 1. 过滤回答过短的
    filtered = []
    for r in results:
        answer = ""
        for turn in r["original"].get("conversations", []):
            if turn.get("from") in ("gpt", "assistant"):
                answer = turn.get("value", "")
                break
        if len(answer) < min_answer_len:
            continue
        filtered.append(r)

    print(f"[FILTER] {len(results)} -> {len(filtered)} (removed {len(results)-len(filtered)} short answers)")

    # 2. 按相似度分数降序排序
    filtered.sort(key=lambda x: x["score"], reverse=True)

    # 3. 取前 max_samples 条
    selected = filtered[:max_samples]
    print(f"[SELECT] top {len(selected)} by similarity score")

    # 4. 提取原始数据
    return [r["original"] for r in selected]


def main():
    parser = argparse.ArgumentParser(description="Retrieval-Augmented SFT data selection")
    parser.add_argument("--ceval_dir", type=str, default="./data/ceval", help="C-Eval data directory")
    parser.add_argument("--sft_dir", type=str, default="./data/sft", help="SFT data directory")
    parser.add_argument("--output_dir", type=str, default="./data/sft_retrieval", help="Output directory")
    parser.add_argument("--model_path", type=str, default="./bge-m3", help="BGE-M3 model path")
    parser.add_argument("--sample_size", type=int, default=100000, help="SFT candidate sample size (0=all)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K per query")
    parser.add_argument("--batch_size", type=int, default=64, help="Encoding batch size")
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["basic_medicine", "clinical_medicine", "physician"],
        help="C-Eval subjects",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip_encoding", action="store_true", help="跳过编码，加载缓存的 embeddings")
    parser.add_argument("--no_dedup", action="store_true", help="不去重（默认去重，每条候选只保留最高分的那次命中）")
    parser.add_argument("--max_samples", type=int, default=0, help="质量过滤后保留的最大条数（0=不过滤）")
    parser.add_argument("--min_answer_len", type=int, default=5, help="回答最短字符数，低于此值丢弃")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 缓存路径
    cache_query_embs = os.path.join(args.output_dir, "query_embs.npy")
    cache_cand_embs = os.path.join(args.output_dir, "candidate_embs.mmap")
    cache_candidates = os.path.join(args.output_dir, "candidates.json")
    cache_cand_dim = os.path.join(args.output_dir, "candidate_dim.txt")

    if args.skip_encoding and os.path.exists(cache_query_embs) and os.path.exists(cache_candidates):
        # ---- 从缓存加载，跳过编码 ----
        print("-" * 60)
        print("Loading cached embeddings (skip_encoding=True)...")

        queries = load_ceval(args.ceval_dir, args.subjects)
        query_embs = np.load(cache_query_embs)
        print(f"Query embeddings: {query_embs.shape}")

        n_cand = sum(1 for _ in open(cache_candidates, "r"))
        dim = int(open(cache_cand_dim).read().strip())
        candidate_embs = np.memmap(cache_cand_embs, dtype="float32", mode="r", shape=(n_cand, dim))
        print(f"Candidate embeddings: {candidate_embs.shape}")

        candidates = []
        with open(cache_candidates, "r", encoding="utf-8") as f:
            for line in f:
                candidates.append(json.loads(line))

    else:
        # ---- 完整编码流程 ----
        # Step 1-2: 加载 C-Eval dev 查询
        print("-" * 60)
        print("Step 1-2: Loading C-Eval dev queries...")
        queries = load_ceval(args.ceval_dir, args.subjects)

        # Step 3-5: 加载 SFT 候选集
        print("-" * 60)
        print("Step 3-5: Loading SFT candidates...")
        candidates = load_sft_candidates(args.sft_dir, args.sample_size)

        # Step 4: Embedding
        print("-" * 60)
        print("Step 4: Loading BGE-M3 model...")
        model = SentenceTransformer(args.model_path)

        print("Encoding queries...")
        query_texts = [q["query"] for q in queries]
        query_embs = encode_texts(model, query_texts, args.batch_size, desc="Encoding queries")
        print(f"Query embeddings shape: {query_embs.shape}")

        print("Encoding candidates...")
        candidate_texts = [c["text"] for c in candidates]
        candidate_embs = encode_texts(model, candidate_texts, args.batch_size,
                                      desc="Encoding candidates", mmap_path=cache_cand_embs)
        print(f"Candidate embeddings shape: {candidate_embs.shape}")

        # 缓存到磁盘
        np.save(cache_query_embs, query_embs)
        with open(cache_cand_dim, "w") as f:
            f.write(str(candidate_embs.shape[1]))
        with open(cache_candidates, "w", encoding="utf-8") as f:
            for c in candidates:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"Cached embeddings to {args.output_dir}")

    # Step 6: 建立 FAISS 索引
    print("-" * 60)
    print("Step 6: Building FAISS index...")
    index = build_faiss_index(candidate_embs)
    print(f"Index total: {index.ntotal}")

    # Step 7: 检索
    print("-" * 60)
    dedup = not args.no_dedup
    print(f"Step 7: Retrieving top-{args.top_k} per query (dedup={dedup})...")
    retrieved = retrieve_topk(index, query_embs, candidates, args.top_k, dedup=dedup)

    # Step 8: 质量过滤
    if args.max_samples > 0:
        print("-" * 60)
        print(f"Step 8: Quality filtering (max_samples={args.max_samples}, min_answer_len={args.min_answer_len})...")
        retrieved = quality_filter(retrieved, args.max_samples, args.min_answer_len)

    # Step 9: 保存
    print("-" * 60)
    output_path = os.path.join(args.output_dir, "retrieval_sft.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in retrieved:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Step 9: Saved {len(retrieved)} samples to {output_path}")
    print("-" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
