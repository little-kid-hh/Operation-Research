from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class Item:
    l: float
    w: float
    h: float
    v: float


@dataclass(frozen=True)
class Bin:
    L: float
    W: float
    H: float
    V: float


@dataclass
class OrderCase:
    dispatch_id: int
    items: List[Item]
    bin: Bin
    ground_truth: bool


def is_feasible(items, bin):
    """3D装箱可行性粗筛 - 工业级进化版 (v4.0)"""
    L, W, H = bin.L, bin.W, bin.H
    bin_dims = sorted([L, W, H])
    bin_min, bin_med, bin_max = bin_dims[0], bin_dims[1], bin_dims[2]
    bin_vol = L * W * H

    total_vol = 0
    max_item_max = 0
    max_item_med = 0
    sum_min_dim = 0
    min_item_dim = float("inf")
    min_item_dims = []
    N = len(items)

    for item in items:
        dims = sorted([item.l, item.w, item.h])
        a, b, c = dims[0], dims[1], dims[2]

        # 1. 单物品三阶超限（必要条件）
        if c > bin_max or b > bin_med or a > bin_min:
            return False

        total_vol += item.v
        if total_vol > bin_vol:  # 提前剪枝
            return False

        # 2. 更新全局统计量
        max_item_max = max(max_item_max, c)
        max_item_med = max(max_item_med, b)
        sum_min_dim += a
        min_item_dim = min(min_item_dim, a)
        min_item_dims.append(a)

    # 3. 体积超限（保留2%碎片空间）
    fill_ratio = total_vol / bin_vol
    if total_vol > bin_vol * 0.98:
        return False

    # 4. 方差补偿碎片率检测
    if N > 1:
        mean_min = sum_min_dim / N
        variance_min = sum((x - mean_min) ** 2 for x in min_item_dims) / N
        sigma_min = variance_min**0.5
    else:
        sigma_min = 0

    base_threshold = 2.8 * bin_max * (1 - 0.15 * fill_ratio)
    fragmentation_threshold = base_threshold + 0.3 * sigma_min

    if sum_min_dim > fragmentation_threshold:
        return False

    # 5. 高填充率维度锁死检测
    if fill_ratio > 0.95:
        # 5.1 关键维度阻塞
        if max_item_max / bin_max > 0.95:
            return False

        # 5.2 最小连通块检测
        min_remaining_dim = min(bin_min, bin_med, bin_max) * (1 - fill_ratio) ** 0.5
        if min_remaining_dim < min_item_dim * 1.2:
            return False

    # 6. 极端尺寸分布检测
    if max_item_med / bin_med > 0.85 and fill_ratio > 0.85:
        if min_item_dim / bin_min < 0.15 and N > 10:
            return False

    return True


def is_feasible_with_trace(items: Sequence[Item], bin_obj: Bin) -> Tuple[bool, str]:
    """Same logic as is_feasible, plus reason code for debugging."""
    bin_dims = sorted([bin_obj.L, bin_obj.W, bin_obj.H])
    bin_min, bin_med, bin_max = bin_dims
    bin_vol = bin_obj.V

    total_vol = 0.0
    max_item_max = 0.0
    max_item_med = 0.0
    sum_min_dim = 0.0
    min_item_dim = float("inf")
    min_item_dims: List[float] = []
    N = len(items)

    for item in items:
        dims = sorted([item.l, item.w, item.h])
        a, b, c = dims

        if c > bin_max or b > bin_med or a > bin_min:
            return False, "single_item_dimension_exceed"

        total_vol += item.v
        if total_vol > bin_vol:
            return False, "volume_overflow_early"

        max_item_max = max(max_item_max, c)
        max_item_med = max(max_item_med, b)
        sum_min_dim += a
        min_item_dim = min(min_item_dim, a)
        min_item_dims.append(a)

    fill_ratio = total_vol / bin_vol if bin_vol > 0 else 0.0
    if total_vol > bin_vol * 0.98:
        return False, "volume_overflow_98pct"

    if N > 1:
        mean_min = sum_min_dim / N
        variance_min = sum((x - mean_min) ** 2 for x in min_item_dims) / N
        sigma_min = variance_min**0.5
    else:
        sigma_min = 0.0

    base_threshold = 2.8 * bin_max * (1 - 0.15 * fill_ratio)
    fragmentation_threshold = base_threshold + 0.3 * sigma_min
    if sum_min_dim > fragmentation_threshold:
        return False, "variance_compensated_fragmentation"

    if fill_ratio > 0.95:
        if max_item_max / bin_max > 0.95:
            return False, "high_fill_max_dim_block"
        min_remaining_dim = min(bin_min, bin_med, bin_max) * (1 - fill_ratio) ** 0.5
        if min_remaining_dim < min_item_dim * 1.2:
            return False, "high_fill_connectivity_block"

    if max_item_med / bin_med > 0.85 and fill_ratio > 0.85:
        if min_item_dim / bin_min < 0.15 and N > 10:
            return False, "extreme_dimension_distribution"

    return True, "pass"


def _to_float(row: dict, key: str) -> float:
    val = row.get(key, "")
    return float(val) if val not in ("", None) else math.nan


def load_bin_map_from_training(training_csv: Path) -> Dict[int, Bin]:
    """Read dispatch -> bin dimensions from training_2orientations.csv."""
    if not training_csv.exists():
        return {}

    bin_map: Dict[int, Bin] = {}
    with training_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dispatch_id = int(row["发车号"])
            L = _to_float(row, "vehicle_length")
            W = _to_float(row, "vehicle_width")
            H = _to_float(row, "vehicle_height")
            if math.isnan(L) or math.isnan(W) or math.isnan(H):
                continue
            bin_map[dispatch_id] = Bin(L=L, W=W, H=H, V=L * W * H)
    return bin_map


def load_cases(
    item_types_csv: Path,
    default_bin: Bin,
    training_csv_for_bin: Path | None = None,
) -> List[OrderCase]:
    by_dispatch = defaultdict(list)
    gt_by_dispatch: Dict[int, bool] = {}

    with item_types_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"dispatch_id", "if_loaded", "dim_s", "dim_m", "dim_l", "n_items"}
        missing = required_cols.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row in reader:
            dispatch_id = int(row["dispatch_id"])
            if_loaded = int(row["if_loaded"]) == 1
            gt_by_dispatch.setdefault(dispatch_id, if_loaded)

            l = float(row["dim_s"])
            w = float(row["dim_m"])
            h = float(row["dim_l"])
            n_items = int(row["n_items"])

            v = l * w * h
            for _ in range(n_items):
                by_dispatch[dispatch_id].append(Item(l=l, w=w, h=h, v=v))

    bin_map: Dict[int, Bin] = {}
    if training_csv_for_bin:
        bin_map = load_bin_map_from_training(training_csv_for_bin)

    cases: List[OrderCase] = []
    for dispatch_id, items in by_dispatch.items():
        cases.append(
            OrderCase(
                dispatch_id=dispatch_id,
                items=items,
                bin=bin_map.get(dispatch_id, default_bin),
                ground_truth=gt_by_dispatch.get(dispatch_id, True),
            )
        )

    cases.sort(key=lambda x: x.dispatch_id)
    return cases


def order_features(case: OrderCase) -> dict:
    items = case.items
    dims = [sorted([it.l, it.w, it.h]) for it in items]
    a_vals = [d[0] for d in dims]
    b_vals = [d[1] for d in dims]
    c_vals = [d[2] for d in dims]
    asr_vals = [c / a for a, c in zip(a_vals, c_vals) if a > 0]
    total_vol = sum(it.v for it in items)
    fill_ratio = total_vol / case.bin.V if case.bin.V > 0 else math.nan

    type_counter = Counter((int(d[0]), int(d[1]), int(d[2])) for d in dims)
    top_types = type_counter.most_common(3)

    return {
        "item_count": len(items),
        "distinct_types": len(type_counter),
        "total_vol": total_vol,
        "bin_vol": case.bin.V,
        "fill_ratio": fill_ratio,
        "a_min": min(a_vals) if a_vals else math.nan,
        "a_max": max(a_vals) if a_vals else math.nan,
        "b_min": min(b_vals) if b_vals else math.nan,
        "b_max": max(b_vals) if b_vals else math.nan,
        "c_min": min(c_vals) if c_vals else math.nan,
        "c_max": max(c_vals) if c_vals else math.nan,
        "asr_max": max(asr_vals) if asr_vals else math.nan,
        "asr_avg": mean(asr_vals) if asr_vals else math.nan,
        "top_types": top_types,
    }


def pick_typical_errors(
    errors: Sequence[dict], max_samples: int = 2, use_reason_priority: bool = True
) -> List[dict]:
    if not errors:
        return []

    if use_reason_priority:
        reason_count = Counter(e["reason"] for e in errors)
    else:
        reason_count = Counter()

    def _sort_key(e: dict):
        feat = e["features"]
        return (
            reason_count.get(e["reason"], 0),
            feat["fill_ratio"],
            feat["item_count"],
            feat["distinct_types"],
        )

    ranked = sorted(errors, key=_sort_key, reverse=True)
    return ranked[:max_samples]


def render_error_case(case_dict: dict, index: int) -> str:
    case: OrderCase = case_dict["case"]
    reason = case_dict["reason"]
    feat = case_dict["features"]
    gt_label = "True(可装)" if case.ground_truth else "False(不可装)"
    pred_label = "True" if case_dict["pred"] else "False"

    top_types_text = ", ".join(
        f"{dims} x{cnt}" for dims, cnt in feat["top_types"]
    ) or "N/A"

    return (
        f"#### 样本 {index}: dispatch_id={case.dispatch_id}\n"
        f"- GT={gt_label}, Pred={pred_label}, 失败原因标签=`{reason}`\n"
        f"- 箱体尺寸(L,W,H)=({case.bin.L:.2f}, {case.bin.W:.2f}, {case.bin.H:.2f}), 箱体体积={feat['bin_vol']:.2f}\n"
        f"- 总件数={feat['item_count']}, 尺寸类型数={feat['distinct_types']}\n"
        f"- 总体积={feat['total_vol']:.2f}, 体积占比(fill_ratio)={feat['fill_ratio']:.4f}\n"
        f"- 尺寸分布: min边[{feat['a_min']:.2f}, {feat['a_max']:.2f}], "
        f"med边[{feat['b_min']:.2f}, {feat['b_max']:.2f}], max边[{feat['c_min']:.2f}, {feat['c_max']:.2f}]\n"
        f"- 长细比: max={feat['asr_max']:.4f}, avg={feat['asr_avg']:.4f}\n"
        f"- 主导尺寸类型(top3): {top_types_text}\n"
    )


def build_report(
    all_cases: Sequence[OrderCase], fn_cases: Sequence[dict], fp_cases: Sequence[dict]
) -> str:
    total = len(all_cases)
    gt_true = sum(1 for c in all_cases if c.ground_truth)
    gt_false = total - gt_true
    fn = len(fn_cases)
    fp = len(fp_cases)

    fn_rate_in_true = (fn / gt_true * 100.0) if gt_true else 0.0
    fp_rate_in_false = (fp / gt_false * 100.0) if gt_false else 0.0

    fn_samples = pick_typical_errors(fn_cases, max_samples=2, use_reason_priority=True)
    fp_samples = pick_typical_errors(fp_cases, max_samples=2, use_reason_priority=False)

    lines = [
        "# 3D-BPP `is_feasible` 错题本与进化反馈报告",
        "",
        "## 1) 整体指标",
        f"- 评估总订单数: **{total}**",
        f"- GT=True(可装)订单数: **{gt_true}**",
        f"- GT=False(不可装)订单数: **{gt_false}**",
        f"- 假阴性 FN: **{fn} / {gt_true} = {fn_rate_in_true:.2f}%**",
        f"- 假阳性 FP: **{fp} / {gt_false} = {fp_rate_in_false:.2f}%**",
        "",
        "## 2) 错题抽样（典型样本）",
        "",
        "### 2.1 FN 典型错题（真实可装，却被判 False）",
    ]

    if fn_samples:
        for i, s in enumerate(fn_samples, 1):
            lines.append(render_error_case(s, i))
    else:
        lines.append("- 无 FN 样本。")

    lines += [
        "",
        "### 2.2 FP 典型错题（真实不可装，却被判 True）",
    ]
    if fp_samples:
        for i, s in enumerate(fp_samples, 1):
            lines.append(render_error_case(s, i))
    else:
        lines.append("- 无 FP 样本。")

    lines += [
        "",
        "## 3) Debug 引导词（直接喂给大模型）",
        "",
        "请分析以上错题数据，找出你当前 `is_feasible` 函数中哪些代数判断条件（如阈值设定、求和公式）违背了真实 3D 物理规律，并输出优化后的下一代 Python 函数代码。",
    ]
    return "\n".join(lines).strip() + "\n"


def evaluate(cases: Sequence[OrderCase]) -> Tuple[List[dict], List[dict]]:
    fn_cases: List[dict] = []
    fp_cases: List[dict] = []

    for case in cases:
        pred, reason = is_feasible_with_trace(case.items, case.bin)
        features = order_features(case)
        entry = {"case": case, "pred": pred, "reason": reason, "features": features}

        if case.ground_truth and not pred:
            fn_cases.append(entry)
        elif (not case.ground_truth) and pred:
            fp_cases.append(entry)

    return fn_cases, fp_cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate is_feasible on dispatch-level real historical data."
    )
    parser.add_argument(
        "--item-csv",
        type=Path,
        default=Path(r"C:\Operation Research\FunSearch_test\training_dispatch_item_types.csv"),
        help="Path of training_dispatch_item_types.csv",
    )
    parser.add_argument(
        "--training-csv",
        type=Path,
        default=Path(r"C:\Operation Research\FunSearch_test\training_2orientations.csv"),
        help="Optional: path of training_2orientations.csv for per-dispatch vehicle dims",
    )
    parser.add_argument(
        "--bin-dims",
        type=float,
        nargs=3,
        metavar=("L", "W", "H"),
        default=(60.0, 25.0, 30.0),
        help="Fallback bin dims when training-csv is missing dispatch bins",
    )
    parser.add_argument(
        "--save-report",
        type=Path,
        default=None,
        help="Optional output markdown file path",
    )
    args = parser.parse_args()

    L, W, H = args.bin_dims
    default_bin = Bin(L=L, W=W, H=H, V=L * W * H)

    cases = load_cases(
        item_types_csv=args.item_csv,
        default_bin=default_bin,
        training_csv_for_bin=args.training_csv if args.training_csv else None,
    )

    if not cases:
        raise RuntimeError("No dispatch cases loaded from CSV.")

    fn_cases, fp_cases = evaluate(cases)
    report_md = build_report(cases, fn_cases, fp_cases)

    print(report_md)

    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(report_md, encoding="utf-8")
        print(f"\n[Saved] {args.save_report}")


if __name__ == "__main__":
    main()
