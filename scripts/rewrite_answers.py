"""
强模型重写回答

读取 retrieval_sft.jsonl，用强模型（DeepSeek / GPT-4 / Qwen-Max）为每条数据
生成带推理链的高质量回答，输出为 ShareGPT 格式用于 SFT 训练。

"""
import os
import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import OpenAI



SYSTEM_PROMPT = """你是一名资深全科医生。请根据患者的描述，给出专业、详细的医学回答。

要求：
1. 先进行推理分析（推理过程用自然语言描述）
2. 再给出最终回答
3. 使用标准医学术语
4. 回答要有理有据，引用病例中的具体信息

返回严格符合以下JSON格式：
{
    "reasoning": "你的推理分析过程",
    "answer": "最终的专业回答"
}"""


def load_input(path: str) -> list[dict]:
    """加载召回数据"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} samples from {path}")
    return data


def extract_question(item: dict) -> str:
    """从 ShareGPT 格式中提取问题"""
    for turn in item.get("conversations", []):
        if turn.get("from") in ("human", "user"):
            return turn.get("value", "")
    return ""


def call_model(client: OpenAI, model: str, question: str, max_retries: int = 3) -> dict | None:
    """调用强模型重写回答"""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                max_tokens=4096,
                temperature=0.3,
            )
            content = resp.choices[0].message.content

            # 尝试提取 JSON 中的 answer 字段
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > 0:
                try:
                    result = json.loads(content[start:end])
                    answer = result.get("answer", "")
                    if answer:
                        return {"answer": answer}
                except json.JSONDecodeError:
                    pass

            # JSON 解析失败或无 answer 字段，直接用原始文本
            return {"answer": content}

        except Exception as e:
            print(f"[ERROR] {e}, attempt {attempt+1}")
            time.sleep(2 ** attempt)

    return None


def extract_answer(item: dict) -> str:
    """从 ShareGPT 格式中提取回答"""
    for turn in item.get("conversations", []):
        if turn.get("from") in ("gpt", "assistant"):
            return turn.get("value", "")
    return ""


def rewrite_single(client: OpenAI, model: str, item: dict, min_rewrite_len: int = 200) -> dict | None:
    """重写单条数据，只重写回答过短的"""
    question = extract_question(item)
    original_answer = extract_answer(item)
    if not question:
        return None

    # 回答已经足够长，保留原文
    if len(original_answer) >= min_rewrite_len:
        return item

    # 回答太短，调用强模型重写
    result = call_model(client, model, question)
    if not result or not result["answer"]:
        return item  # 重写失败就保留原文

    new_item = {
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": result["answer"]},
        ],
    }
    return new_item


def main():
    parser = argparse.ArgumentParser(description="强模型重写回答")
    parser.add_argument("--input", required=True, help="输入文件 (retrieval_sft.jsonl)")
    parser.add_argument("--output", required=True, help="输出文件")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""), help="API Key")
    parser.add_argument("--base_url", default="https://api.xiaomimimo.com/v1", help="API Base URL")
    parser.add_argument("--model", default="mimo-v2.5-pro", help="模型名称")
    parser.add_argument("--max_workers", type=int, default=10, help="并发数")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条（0=全部）")
    parser.add_argument("--min_rewrite_len", type=int, default=200, help="回答长度 >= 此值保留原文，< 此值才重写")
    args = parser.parse_args()

    if not args.api_key:
        print("请通过 --api_key 或 OPENAI_API_KEY 环境变量提供 API Key")
        return

    # 加载数据
    data = load_input(args.input)
    if args.limit > 0:
        data = data[:args.limit]
        print(f"Limiting to first {args.limit} samples")

    # 初始化客户端
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # 统计需要重写的数量
    short_count = sum(1 for item in data if len(extract_answer(item)) < args.min_rewrite_len)
    print(f"Total: {len(data)}, need rewrite (answer < {args.min_rewrite_len} chars): {short_count}")
    print(f"Keep original (answer >= {args.min_rewrite_len} chars): {len(data) - short_count}")

    # 并发重写
    results = []
    failed = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(rewrite_single, client, args.model, item, args.min_rewrite_len): i
            for i, item in enumerate(data)
        }

        with open(args.output, "w", encoding="utf-8") as fout:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Rewriting"):
                result = future.result()
                if result:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    results.append(result)
                else:
                    failed += 1

    print(f"\nDone! Success: {len(results)}, Failed: {failed}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
