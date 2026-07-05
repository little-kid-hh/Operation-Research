from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .evaluator import Box
from .features import OrderSummary


OrientationLabel = Literal["label_2ori", "label_6ori"]


@dataclass(frozen=True)
class MilpBoxSetScore:
    packaging_factor: float
    mean_box_volume: float
    mean_order_volume: float
    coverage_rate: float
    uncovered_orders: int
    unknown_pairs: int
    orders_with_unknown: int
    assignments: tuple[int | None, ...]


def score_milp_feasibility_matrix(
    orders: list[OrderSummary],
    boxes: list[Box],
    feasible: np.ndarray,
    *,
    unknown: np.ndarray | None = None,
    uncovered_penalty_factor: float = 100.0,
) -> MilpBoxSetScore:
    if feasible.shape != (len(orders), len(boxes)):
        raise ValueError(
            f"feasible matrix shape {feasible.shape} does not match "
            f"{len(orders)} orders x {len(boxes)} boxes"
        )
    if not orders:
        raise ValueError("orders must be non-empty")
    if not boxes:
        raise ValueError("boxes must be non-empty")
    if unknown is None:
        unknown = np.zeros_like(feasible, dtype=bool)
    if unknown.shape != feasible.shape:
        raise ValueError(f"unknown matrix shape {unknown.shape} does not match feasible shape {feasible.shape}")

    box_volumes = np.asarray([box.volume for box in boxes], dtype=np.float64)
    order_volumes = np.asarray([order.total_volume for order in orders], dtype=np.float64)
    feasible_volumes = np.where(feasible, box_volumes[None, :], np.inf)
    chosen_idx = np.argmin(feasible_volumes, axis=1)
    covered = np.isfinite(feasible_volumes[np.arange(len(orders)), chosen_idx])
    penalty = uncovered_penalty_factor * float(np.max(box_volumes))
    assigned_volumes = np.where(covered, box_volumes[chosen_idx], penalty)
    assignments = tuple(
        int(boxes[int(idx)].box_id) if is_covered else None
        for idx, is_covered in zip(chosen_idx, covered)
    )
    mean_order_volume = float(np.mean(order_volumes))
    mean_box_volume = float(np.mean(assigned_volumes))
    uncovered = int(len(orders) - np.count_nonzero(covered))
    return MilpBoxSetScore(
        packaging_factor=mean_box_volume / mean_order_volume if mean_order_volume > 0 else float("inf"),
        mean_box_volume=mean_box_volume,
        mean_order_volume=mean_order_volume,
        coverage_rate=(len(orders) - uncovered) / len(orders),
        uncovered_orders=uncovered,
        unknown_pairs=int(np.count_nonzero(unknown)),
        orders_with_unknown=int(np.count_nonzero(np.any(unknown, axis=1))),
        assignments=assignments,
    )


class MilpLabelTableOracle:
    """MILP oracle backed by precomputed OR2023 BSP labels for fixed packages.

    This only supports boxes whose dimensions exactly match rows in the package
    label table. It is useful for evaluating the 90 existing packages, but not
    for searching continuous generated box dimensions.
    """

    def __init__(
        self,
        *,
        labels_path: Path,
        orientation_label: OrientationLabel = "label_6ori",
        dimension_tolerance: float = 1e-6,
    ) -> None:
        if orientation_label not in {"label_2ori", "label_6ori"}:
            raise ValueError(f"unknown orientation label: {orientation_label}")
        self.labels_path = labels_path
        self.orientation_label = orientation_label
        self.dimension_tolerance = float(dimension_tolerance)
        self._rows_by_order_and_package: dict[tuple[str, int], int] = {}
        self._package_by_dims: dict[tuple[float, float, float], int] = {}
        self._load()

    def _dims_key(self, length: float, width: float, height: float) -> tuple[float, float, float]:
        tol = self.dimension_tolerance
        return tuple(round(float(x) / tol) * tol for x in (length, width, height))

    def _load(self) -> None:
        with self.labels_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                package_id = int(row["package_id"])
                dims = self._dims_key(row["package_l"], row["package_w"], row["package_h"])
                self._package_by_dims[dims] = package_id
                self._rows_by_order_and_package[(str(row["order_id"]), package_id)] = int(
                    row[self.orientation_label]
                )

    def evaluate(self, orders: list[OrderSummary], boxes: list[Box]) -> MilpBoxSetScore:
        package_ids = []
        for box in boxes:
            dims = self._dims_key(box.length, box.width, box.height)
            package_id = self._package_by_dims.get(dims)
            if package_id is None:
                raise ValueError(
                    "MILP label table oracle only supports pre-labeled package dimensions; "
                    f"no package match for box_id={box.box_id} dims={dims}"
                )
            package_ids.append(package_id)

        statuses = np.zeros((len(orders), len(boxes)), dtype=np.int8)
        for i, order in enumerate(orders):
            for j, package_id in enumerate(package_ids):
                statuses[i, j] = self._rows_by_order_and_package.get((str(order.order_id), package_id), 0)
        return score_milp_feasibility_matrix(orders, boxes, statuses == 1, unknown=statuses < 0)


class JavaMilpOracle:
    """Online MILP oracle using GeneratePerminPackageLabels.

    This supports arbitrary generated box dimensions by writing a temporary
    packages file and invoking the Java/Gurobi labeler. It is intentionally
    expensive; use small order limits and iteration counts first.
    """

    def __init__(
        self,
        *,
        xml_path: Path,
        java_classpath: str,
        labeler_class: str = "org.example.GeneratePerminPackageLabels",
        orientation_label: OrientationLabel = "label_6ori",
        time_limit_seconds: float = 30.0,
        allow_bsp_derived_data: bool = True,
        cache_dir: Path | None = None,
        orders_offset: int = 0,
    ) -> None:
        if orientation_label not in {"label_2ori", "label_6ori"}:
            raise ValueError(f"unknown orientation label: {orientation_label}")
        if orders_offset < 0:
            raise ValueError("orders_offset must be non-negative")
        self.xml_path = xml_path
        self.java_classpath = java_classpath
        self.labeler_class = labeler_class
        self.orientation_label = orientation_label
        self.time_limit_seconds = float(time_limit_seconds)
        self.allow_bsp_derived_data = allow_bsp_derived_data
        self.cache_dir = cache_dir
        self.orders_offset = int(orders_offset)
        self._feasibility_cache: dict[
            tuple[tuple[tuple[str, str], ...], tuple[float, float, float]],
            tuple[int, ...],
        ] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.disk_cache_hits = 0
        self.evaluate_calls = 0
        self.prefetch_calls = 0
        self.prefetch_cache_hits = 0
        self.prefetch_disk_hits = 0
        self.prefetch_misses = 0
        self.uncached_batches = 0
        self.uncached_boxes = 0
        self.subprocess_seconds = 0.0
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, orders: list[OrderSummary], boxes: list[Box]) -> MilpBoxSetScore:
        self.evaluate_calls += 1
        self._validate_environment()
        self._validate_orders_for_xml(orders)
        signature = self._order_signature(orders)
        statuses = np.zeros((len(orders), len(boxes)), dtype=np.int8)
        unknown_boxes: list[Box] = []
        unknown_columns: list[int] = []
        unknown_keys: list[tuple[float, float, float]] = []
        for j, box in enumerate(boxes):
            dims_key = self._box_dims_key(box)
            cache_key = (signature, dims_key)
            cached = self._feasibility_cache.get(cache_key)
            if cached is None:
                cached = self._read_disk_cache(signature, dims_key)
                if cached is not None:
                    self._feasibility_cache[cache_key] = cached
                    self.disk_cache_hits += 1
            if cached is None:
                self.cache_misses += 1
                unknown_boxes.append(box)
                unknown_columns.append(j)
                unknown_keys.append(dims_key)
                continue
            self.cache_hits += 1
            statuses[:, j] = np.asarray(cached, dtype=np.int8)

        if unknown_boxes:
            self.uncached_batches += 1
            self.uncached_boxes += len(unknown_boxes)
            unknown_statuses = self._evaluate_uncached(orders, unknown_boxes)
            for local_j, (global_j, dims_key) in enumerate(zip(unknown_columns, unknown_keys)):
                statuses[:, global_j] = unknown_statuses[:, local_j]
                status_col = tuple(int(x) for x in unknown_statuses[:, local_j])
                self._feasibility_cache[(signature, dims_key)] = status_col
                self._write_disk_cache(signature, dims_key, status_col)
        return score_milp_feasibility_matrix(orders, boxes, statuses == 1, unknown=statuses < 0)

    def prefetch_box_statuses(self, orders: list[OrderSummary], boxes: list[Box]) -> None:
        """Populate feasibility cache for arbitrary box dimensions.

        The local-search runner can call this before scoring a group of
        candidate box sets. Cache keys depend on the order signature and box
        dimensions, not on candidate-local box ids, so dimensions are deduped
        and sent to Java in one batch.
        """

        self.prefetch_calls += 1
        if not boxes:
            return
        self._validate_environment()
        self._validate_orders_for_xml(orders)
        signature = self._order_signature(orders)

        unique_by_dims: dict[tuple[float, float, float], tuple[float, float, float]] = {}
        for box in boxes:
            dims_key = self._box_dims_key(box)
            unique_by_dims.setdefault(dims_key, dims_key)

        unknown_dims: list[tuple[float, float, float]] = []
        for dims_key in unique_by_dims:
            cache_key = (signature, dims_key)
            cached = self._feasibility_cache.get(cache_key)
            if cached is not None:
                self.prefetch_cache_hits += 1
                continue
            cached = self._read_disk_cache(signature, dims_key)
            if cached is not None:
                self._feasibility_cache[cache_key] = cached
                self.disk_cache_hits += 1
                self.prefetch_disk_hits += 1
                continue
            unknown_dims.append(dims_key)

        if not unknown_dims:
            return

        self.cache_misses += len(unknown_dims)
        self.prefetch_misses += len(unknown_dims)
        self.uncached_batches += 1
        self.uncached_boxes += len(unknown_dims)
        synthetic_boxes = [
            Box(box_id=idx, length=dims[0], width=dims[1], height=dims[2])
            for idx, dims in enumerate(unknown_dims)
        ]
        unknown_statuses = self._evaluate_uncached(orders, synthetic_boxes)
        for local_j, dims_key in enumerate(unknown_dims):
            status_col = tuple(int(x) for x in unknown_statuses[:, local_j])
            self._feasibility_cache[(signature, dims_key)] = status_col
            self._write_disk_cache(signature, dims_key, status_col)

    def cache_info(self) -> dict[str, int | float]:
        return {
            "entries": len(self._feasibility_cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "disk_hits": self.disk_cache_hits,
            "evaluate_calls": self.evaluate_calls,
            "prefetch_calls": self.prefetch_calls,
            "prefetch_cache_hits": self.prefetch_cache_hits,
            "prefetch_disk_hits": self.prefetch_disk_hits,
            "prefetch_misses": self.prefetch_misses,
            "uncached_batches": self.uncached_batches,
            "uncached_boxes": self.uncached_boxes,
            "subprocess_seconds": self.subprocess_seconds,
        }

    def _labeler_window_args(self, orders: list[OrderSummary], boxes: list[Box]) -> tuple[int, int, int]:
        """Return Java labeler window arguments for the selected order slice."""

        return (
            self.orders_offset + len(orders),
            self.orders_offset * len(boxes),
            len(orders) * len(boxes),
        )

    def _evaluate_uncached(self, orders: list[OrderSummary], boxes: list[Box]) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix="or2023_milp_oracle_") as tmp:
            tmp_dir = Path(tmp)
            packages_path = tmp_dir / "packages.txt"
            output_path = tmp_dir / "labels.csv"
            with packages_path.open("w", encoding="utf-8") as f:
                for box in boxes:
                    f.write(f"{box.box_id}\t{box.length:.6f}\t{box.width:.6f}\t{box.height:.6f}\n")

            max_orders_per_xml, start_global_task, max_tasks = self._labeler_window_args(orders, boxes)
            cmd = [
                "java",
                f"-Dor2023.bpp.timeLimit2ori={self.time_limit_seconds}",
                f"-Dor2023.bpp.timeLimit6ori={self.time_limit_seconds}",
                "-cp",
                self.java_classpath,
                self.labeler_class,
                str(self.xml_path.parent),
                str(packages_path),
                str(output_path),
                "1",
                str(max_orders_per_xml),
                str(len(boxes)),
                f"^{self.xml_path.name}$",
                str(start_global_task),
                str(max_tasks),
                "true" if self.allow_bsp_derived_data else "false",
            ]
            subprocess_started = time.perf_counter()
            try:
                subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    "Java MILP oracle failed with exit code "
                    f"{exc.returncode}\nSTDOUT:\n{exc.stdout}\nSTDERR:\n{exc.stderr}"
                ) from exc
            finally:
                self.subprocess_seconds += time.perf_counter() - subprocess_started
            return self._read_output_statuses(output_path, orders, boxes)

    def _validate_environment(self) -> None:
        if shutil.which("java") is None:
            raise RuntimeError(
                "Java runtime not found. Install Java and configure Gurobi before using "
                "the online MILP oracle."
            )
        classpath_parts = self.java_classpath.split(os.pathsep)
        if not any((Path(part) / "org/example/GeneratePerminPackageLabels.class").exists() for part in classpath_parts):
            raise RuntimeError(
                "GeneratePerminPackageLabels.class not found on java_classpath. "
                "Compile MILP_3DBPP with the Java/Gurobi dependencies first."
            )

    def _validate_orders_for_xml(self, orders: list[OrderSummary]) -> None:
        expected_instance = self.xml_path.name
        seen: set[str] = set()
        for order in orders:
            order_id = str(order.order_id)
            if order.instance_name != expected_instance:
                raise ValueError(
                    "JavaMilpOracle orders must come from the same XML passed to the Java labeler; "
                    f"expected instance={expected_instance}, got ({order.instance_name}, {order.order_id})"
                )
            if order_id in seen:
                raise ValueError(f"duplicate order_id in oracle input: {order_id}")
            seen.add(order_id)

    @staticmethod
    def _order_signature(orders: list[OrderSummary]) -> tuple[tuple[str, str], ...]:
        return tuple((order.instance_name, str(order.order_id)) for order in orders)

    @staticmethod
    def _box_dims_key(box: Box) -> tuple[float, float, float]:
        return round(box.length, 6), round(box.width, 6), round(box.height, 6)

    def _cache_payload(
        self,
        signature: tuple[tuple[str, str], ...],
        dims_key: tuple[float, float, float],
    ) -> dict:
        return {
            "xml_name": self.xml_path.name,
            "orders": [[instance_name, order_id] for instance_name, order_id in signature],
            "dims": list(dims_key),
            "orientation_label": self.orientation_label,
            "time_limit_seconds": self.time_limit_seconds,
            "labeler_class": self.labeler_class,
        }

    def _disk_cache_path(
        self,
        signature: tuple[tuple[str, str], ...],
        dims_key: tuple[float, float, float],
    ) -> Path | None:
        if self.cache_dir is None:
            return None
        payload = self._cache_payload(signature, dims_key)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _read_disk_cache(
        self,
        signature: tuple[tuple[str, str], ...],
        dims_key: tuple[float, float, float],
    ) -> tuple[int, ...] | None:
        path = self._disk_cache_path(signature, dims_key)
        if path is None or not path.exists():
            return None
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        expected = self._cache_payload(signature, dims_key)
        for key, expected_value in expected.items():
            if payload.get(key) != expected_value:
                return None
        return tuple(int(x) for x in payload["statuses"])

    def _write_disk_cache(
        self,
        signature: tuple[tuple[str, str], ...],
        dims_key: tuple[float, float, float],
        statuses: tuple[int, ...],
    ) -> None:
        path = self._disk_cache_path(signature, dims_key)
        if path is None:
            return
        payload = self._cache_payload(signature, dims_key)
        payload["statuses"] = list(statuses)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        tmp_path.replace(path)

    def _read_output_statuses(self, output_path: Path, orders: list[OrderSummary], boxes: list[Box]) -> np.ndarray:
        box_index = {box.box_id: idx for idx, box in enumerate(boxes)}
        order_index = {str(order.order_id): idx for idx, order in enumerate(orders)}
        statuses = np.zeros((len(orders), len(boxes)), dtype=np.int8)
        seen = np.zeros((len(orders), len(boxes)), dtype=bool)
        with output_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                order_id = str(row["order_id"])
                package_id = int(row["package_id"])
                if order_id not in order_index:
                    raise ValueError(
                        "Java MILP oracle returned an order outside the selected window: "
                        f"order_id={order_id}, selected_orders={list(order_index.keys())[:5]}"
                    )
                if package_id not in box_index:
                    raise ValueError(
                        "Java MILP oracle returned a package outside the requested boxes: "
                        f"package_id={package_id}, requested_package_ids={list(box_index.keys())[:5]}"
                    )
                i = order_index[order_id]
                j = box_index[package_id]
                if seen[i, j]:
                    raise ValueError(f"Java MILP oracle returned duplicate label for order={order_id}, package={package_id}")
                statuses[i, j] = int(row[self.orientation_label])
                seen[i, j] = True
        if not np.all(seen):
            missing = int((~seen).sum())
            raise ValueError(
                "Java MILP oracle returned an incomplete label matrix: "
                f"missing_pairs={missing}, expected_pairs={len(orders) * len(boxes)}"
            )
        return statuses
