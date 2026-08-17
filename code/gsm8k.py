"""Shared, local GSM8K test-set loader for every experiment."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")
DATA_PATH = Path(__file__).parent / "data" / "gsm8k_test_100.raw.json"
FINAL_ANSWER = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*$")


def load_test_100(factory: Callable[[str, str, int], T]) -> tuple[T, ...]:
    """Load exactly the first 100 official GSM8K ``main/test`` records.

    The file is the unmodified response from Hugging Face's dataset server.
    ``factory`` lets individual experiments retain their own Sample dataclass.
    """
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    rows = payload["rows"]
    if len(rows) != 100:
        raise RuntimeError(f"期望 GSM8K 测试集前 100 条，实际为 {len(rows)} 条。")
    samples: list[T] = []
    for index, item in enumerate(rows, start=1):
        row = item["row"]
        match = FINAL_ANSWER.search(row["answer"])
        if not match:
            raise ValueError(f"GSM8K 样本 {index} 未包含 #### 最终答案。")
        value = match.group(1).replace(",", "")
        if "." in value:
            raise ValueError(f"GSM8K 样本 {index} 的非整数答案暂不受支持：{value}")
        samples.append(factory(f"gsm8k-test-{index:03d}", row["question"], int(value)))
    return tuple(samples)
