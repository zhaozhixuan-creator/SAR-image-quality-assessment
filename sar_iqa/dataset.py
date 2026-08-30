"""数据集级质检：目录扫描 → 逐图评估 → 聚合统计 / 批次合格判定 / 离群检测 / 分组审计 / 落库。

对应 README §二 尚未落码的「落库与看板」与「人工抽检」层，及 §七「分组留存率审计」。
复用 `single.assess_image` 的单图单元，把整个目录的逐图结果汇总为数据集整体质量结论。

四项能力：
1. 聚合统计分布   —— 跨图汇总每项指标的中位数 / 均值±std / 分位 / MAD / IQR。
2. 批次合格判定   —— 按 GB/T 24356 把逐图分级汇总为整批合格 / 不合格 + 抽样建议。
3. 跨图一致性/离群 —— 稳健 MAD / IQR 离群阈值标记异常图像；元数据边车一致性。
4. 结果落库+分组审计 —— 逐图结果持久化 JSON，按产品级/元数据键分组统计留存率。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from . import spec
from .single import assess_image, _json_default, _rule_to_dict
from .report import generate_html
from .grading import sample_size

IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

# 元数据一致性默认关注的「身份键」：这些字段在一批图里应当一致（时间戳等逐图变化字段除外）。
DEFAULT_IDENTITY_KEYS = (
    "processor_version", "software_version", "sensor",
    "nominal_resolution", "pixel_spacing", "acquisition_mode",
    "crs", "calibration_factor", "coherence_window",
)


@dataclass
class ImageRecord:
    """单张图像的落库记录（纯可序列化，不含活动对象）。"""

    stem: str
    path: str
    status: str = "ok"                # "ok" | "error"
    error: Optional[dict] = None      # {"stage", "type", "message"}
    grade: Optional[dict] = None
    pipeline: Optional[dict] = None   # PipelineResult.as_dict()
    metrics: Optional[list] = None    # [MetricResult.as_dict(), ...]
    despeckling: Optional[dict] = None
    rule_results: Optional[list] = None
    metadata: Optional[dict] = None
    report_html: Optional[str] = None
    report_json: Optional[str] = None

    def as_dict(self) -> dict:
        d = {"stem": self.stem, "path": self.path, "status": self.status,
             "report_html": self.report_html, "report_json": self.report_json}
        if self.error is not None:
            d["error"] = self.error
        if self.grade is not None:
            d["grade"] = self.grade
        if self.pipeline is not None:
            d["pipeline"] = self.pipeline
        if self.metrics is not None:
            d["metrics"] = self.metrics
        if self.despeckling is not None:
            d["despeckling"] = self.despeckling
        if self.rule_results is not None:
            d["rule_results"] = self.rule_results
        if self.metadata is not None:
            d["metadata"] = self.metadata
        return d


@dataclass
class MetricAggregate:
    """单指标跨图分布。"""

    key: str
    name: str
    dimension: str
    unit: str
    kind: str
    level: str
    phase: int
    n_value: int
    n_pass: int
    n_warn: int
    n_fail: int
    n_nodata: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    p10: Optional[float] = None
    p90: Optional[float] = None
    mad: Optional[float] = None
    iqr: Optional[float] = None
    outliers: list = field(default_factory=list)

    @property
    def n_evaluated(self) -> int:
        return self.n_pass + self.n_warn + self.n_fail

    def as_dict(self) -> dict:
        return {
            "key": self.key, "name": self.name, "dimension": self.dimension,
            "unit": self.unit, "kind": self.kind, "level": self.level, "phase": self.phase,
            "n_evaluated": self.n_evaluated,
            "n_pass": self.n_pass, "n_warn": self.n_warn, "n_fail": self.n_fail,
            "n_nodata": self.n_nodata, "n_value": self.n_value,
            "mean": self.mean, "std": self.std, "min": self.min, "max": self.max,
            "median": self.median, "p10": self.p10, "p90": self.p90,
            "mad": self.mad, "iqr": self.iqr, "outliers": self.outliers,
        }


@dataclass
class GroupStat:
    """分组审计结果。"""

    group: str
    n: int
    n_fail: int
    retention_rate: float
    mean_score: float
    score_std: float
    defect_counts: dict

    def as_dict(self) -> dict:
        return {
            "group": self.group, "n": self.n, "n_fail": self.n_fail,
            "retention_rate": self.retention_rate, "mean_score": self.mean_score,
            "score_std": self.score_std, "defect_counts": self.defect_counts,
        }


@dataclass
class DatasetResult:
    """一次数据集级质检的完整结果（全部可序列化）。"""

    root: str
    level: str
    config: dict
    records: list
    aggregates: list
    batch: dict
    group_audit: dict
    metadata_consistency: dict
    score_distribution: dict
    generated_at: str = ""
    group_by: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "schema": "dataset-qc/v1",
            "generated_at": self.generated_at,
            "root": self.root, "level": self.level, "config": self.config,
            "n_images_total": len(self.records),
            "n_images_ok": sum(1 for r in self.records if r.status == "ok"),
            "n_images_error": sum(1 for r in self.records if r.status == "error"),
            "score_distribution": self.score_distribution,
            "batch": self.batch,
            "aggregates": [a.as_dict() for a in self.aggregates],
            "group_audit": self.group_audit,
            "metadata_consistency": self.metadata_consistency,
            "records": [_record_summary(r, self.group_by) for r in self.records],
        }


# ---------------------------------------------------------------------------
# 目录扫描
# ---------------------------------------------------------------------------
def discover_images(directory: str, exts=IMAGE_EXTS, recursive: bool = False) -> list[dict]:
    """扫描目录，返回 [{path, stem, metadata}]，按 stem 升序。

    metadata = 同名 .json 边车绝对路径，或 None（无边车）。
    跳过 *_report.* 等自身输出，避免二次扫描。
    """
    found: list[dict] = []

    def _collect(root: str, fn: str) -> None:
        ext = os.path.splitext(fn)[1].lower()
        if ext not in exts:
            return
        stem = os.path.splitext(fn)[0]
        if stem.endswith("_report"):
            return
        path = os.path.join(root, fn)
        meta = os.path.join(root, stem + ".json")
        if not os.path.isfile(meta):
            meta = None
        found.append({"path": path, "stem": stem, "metadata": meta})

    if recursive:
        for root, _dirs, files in os.walk(directory):
            for fn in sorted(files):
                _collect(root, fn)
    else:
        try:
            entries = os.listdir(directory)
        except OSError:
            return []
        for fn in sorted(entries):
            p = os.path.join(directory, fn)
            if os.path.isfile(p):
                _collect(directory, fn)

    found.sort(key=lambda d: d["stem"])
    return found


# ---------------------------------------------------------------------------
# 逐图评估
# ---------------------------------------------------------------------------
def _process_one(f: dict, *, level, domain, channels, config, metadata_extra,
                 per_image: bool, per_image_dir: str) -> ImageRecord:
    rec = ImageRecord(stem=f["stem"], path=f["path"])
    try:
        a = assess_image(f["path"], domain=domain, channels=channels,
                         metadata_path=f["metadata"], metadata_extra=metadata_extra,
                         level=level, config=config)
    except Exception as e:  # 单图失败不阻断整批
        rec.status = "error"
        rec.error = {"stage": "assess", "type": type(e).__name__, "message": str(e)}
        return rec

    rec.grade = a.grade
    rec.pipeline = a.pipeline.as_dict()
    rec.metrics = [r.as_dict() for r in a.results]
    rec.despeckling = a.despeckling
    rec.rule_results = [_rule_to_dict(r) for r in a.rule_results]
    rec.metadata = a.metadata if a.metadata else None

    if per_image:
        rel_html = os.path.join("per_image", f"{rec.stem}.html")
        rel_json = os.path.join("per_image", f"{rec.stem}.json")
        rec.report_html = rel_html
        rec.report_json = rel_json
        html_text = generate_html(a.results, a.ctx, a.grade, a.despeckling, a.rule_results,
                                  pipeline=a.pipeline, sample=a.sample)
        with open(os.path.join(per_image_dir, f"{rec.stem}.html"), "w", encoding="utf-8") as fh:
            fh.write(html_text)
        with open(os.path.join(per_image_dir, f"{rec.stem}.json"), "w", encoding="utf-8") as fh:
            json.dump(a.payload(), fh, ensure_ascii=False, indent=2, default=_json_default)
    return rec


# ---------------------------------------------------------------------------
# 聚合统计分布 + 离群检测
# ---------------------------------------------------------------------------
def aggregate(records: list[ImageRecord], mad_k: float = 3.0) -> list[MetricAggregate]:
    """对每项指标（spec.SPEC 顺序）汇总跨图分布，并做稳健离群检测。"""
    ok = [r for r in records if r.status == "ok" and r.metrics]
    by_key = {r.stem: {m["key"]: m for m in r.metrics} for r in ok}

    aggs: list[MetricAggregate] = []
    for s in spec.SPEC:
        values: list[tuple[str, float]] = []
        status_counts = {"pass": 0, "warn": 0, "fail": 0, "nodata": 0}
        units: set[str] = set()
        for r in ok:
            mm = by_key[r.stem].get(s.key)
            if mm is None:
                status_counts["nodata"] += 1
                continue
            st = mm.get("status", "nodata")
            status_counts[st] = status_counts.get(st, 0) + 1
            if mm.get("unit"):
                units.add(mm["unit"])
            v = mm.get("value")
            if v is not None and st != "nodata":
                values.append((r.stem, float(v)))
        unit = next(iter(units), "")
        aggs.append(_build_aggregate(s, values, status_counts, unit, mad_k))
    return aggs


def _build_aggregate(s, values: list[tuple[str, float]], status_counts: dict,
                     unit: str, mad_k: float) -> MetricAggregate:
    n = len(values)
    agg = MetricAggregate(
        key=s.key, name=s.name, dimension=s.dimension, unit=unit,
        kind=s.kind, level=s.level, phase=s.phase,
        n_value=n,
        n_pass=status_counts.get("pass", 0),
        n_warn=status_counts.get("warn", 0),
        n_fail=status_counts.get("fail", 0),
        n_nodata=status_counts.get("nodata", 0),
    )
    if n == 0:
        return agg

    nums = np.array([v for _, v in values], dtype=np.float64)
    agg.mean = float(nums.mean())
    agg.min = float(nums.min())
    agg.max = float(nums.max())
    agg.median = float(np.median(nums))
    agg.p10 = float(np.percentile(nums, 10))
    agg.p90 = float(np.percentile(nums, 90))
    if n >= 2:
        agg.std = float(nums.std(ddof=1))
    q25, q75 = np.percentile(nums, [25, 75])
    iqr = float(q75 - q25)
    agg.iqr = iqr
    mad = float(np.median(np.abs(nums - agg.median)))
    agg.mad = mad

    # 稳健离群检测：MAD（1.4826 缩放）为主，MAD=0 时回退 IQR（1.5×）。
    sigma = 1.4826 * mad
    if sigma > 0:
        for stem, v in values:
            z = (v - agg.median) / sigma
            if abs(z) > mad_k:
                agg.outliers.append({"image": stem, "value": float(v), "z": float(z)})
    elif iqr > 0:
        lo, hi = q25 - 1.5 * iqr, q75 + 1.5 * iqr
        for stem, v in values:
            if v < lo or v > hi:
                agg.outliers.append({"image": stem, "value": float(v),
                                     "z": float((v - agg.median) / iqr)})
    return agg


# ---------------------------------------------------------------------------
# 批次合格判定（GB/T 24356）
# ---------------------------------------------------------------------------
def _level_of(score: float) -> str:
    if score >= 90:
        return "优"
    if score >= 75:
        return "良"
    if score >= 60:
        return "合格"
    return "不合格"


def batch_accept(records: list[ImageRecord], batch_size: Optional[int] = None) -> dict:
    """把逐图分级汇总为整批合格 / 不合格 + 抽样建议。"""
    ok = [r for r in records if r.status == "ok" and r.grade]
    n = len(ok)
    if n == 0:
        return {"n_images": 0,
                "sampling": sample_size(batch_size) if batch_size is not None else None,
                "defect_counts": {"A": 0, "B": 0, "C": 0, "D": 0}, "any_a": False,
                "score_mean": None, "score_std": None, "score_min": None, "score_max": None,
                "level": None, "verdict": "无可用图像", "note": "目录中没有成功质检的图像。"}

    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for r in ok:
        c = r.grade.get("counts", {})
        for k in counts:
            counts[k] += int(c.get(k, 0))
    scores = np.array([float(r.grade["score"]) for r in ok])
    mean = float(scores.mean())
    std = float(scores.std(ddof=1)) if n >= 2 else 0.0

    any_a = counts["A"] > 0
    if any_a:
        verdict, level, note = "整批不合格", "不合格", "出现 A 类（严重）缺陷，整批判不合格（GB/T 24356）。"
    else:
        verdict, level, note = "整批合格", _level_of(mean), ""
    sampling = sample_size(batch_size if batch_size is not None else n)
    return {"n_images": n, "sampling": sampling, "defect_counts": counts, "any_a": any_a,
            "score_mean": mean, "score_std": std,
            "score_min": float(scores.min()), "score_max": float(scores.max()),
            "level": level, "verdict": verdict, "note": note}


# ---------------------------------------------------------------------------
# 分组审计（留存率）
# ---------------------------------------------------------------------------
def _canon(v) -> str:
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    return str(v)


def _group_value(r: ImageRecord, group_by: Optional[str]) -> str:
    if group_by in (None, "level"):
        return (r.pipeline or {}).get("level_label") or (r.pipeline or {}).get("level") or "未知"
    val = (r.metadata or {}).get(group_by)
    if val is None:
        return "(未分组)"
    return _canon(val)


def group_audit(records: list[ImageRecord], group_by: Optional[str] = None) -> dict:
    """按 group_by（默认产品级）分组，统计留存率（非「不合格」即留存）。"""
    ok = [r for r in records if r.status == "ok" and r.grade]
    groups: dict[str, list[ImageRecord]] = {}
    for r in ok:
        groups.setdefault(_group_value(r, group_by), []).append(r)

    out = []
    for g, rs in sorted(groups.items()):
        n = len(rs)
        scores = np.array([float(r.grade["score"]) for r in rs])
        counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        for r in rs:
            c = r.grade.get("counts", {})
            for k in counts:
                counts[k] += int(c.get(k, 0))
        n_fail = sum(1 for r in rs if r.grade.get("level") == "不合格")
        out.append(GroupStat(
            group=g, n=n, n_fail=n_fail,
            retention_rate=round((n - n_fail) / n * 100.0, 2),
            mean_score=round(float(scores.mean()), 2),
            score_std=round(float(scores.std(ddof=1)), 2) if n >= 2 else 0.0,
            defect_counts=counts,
        ))
    return {"group_by": group_by or "level", "groups": [g.as_dict() for g in out]}


# ---------------------------------------------------------------------------
# 元数据一致性
# ---------------------------------------------------------------------------
def metadata_consistency(records: list[ImageRecord],
                         keys: tuple = DEFAULT_IDENTITY_KEYS) -> dict:
    """跨图元数据身份键一致性 + 元数据规则引擎（CEOS-ARD/CARD4L）达标率汇总。"""
    ok = [r for r in records if r.status == "ok"]
    n_total = len(ok)
    fields = []
    all_consistent = True
    for key in keys:
        values: dict[str, dict] = {}
        n_present = 0
        for r in ok:
            m = r.metadata or {}
            if key not in m or m[key] is None:
                continue
            n_present += 1
            v = m[key]
            ck = _canon(v)
            bucket = values.setdefault(ck, {"value": v, "count": 0, "images": []})
            bucket["count"] += 1
            bucket["images"].append(r.stem)
        n_missing = n_total - n_present
        consistent = (n_missing == 0) and len(values) == 1
        if not consistent:
            all_consistent = False
        fields.append({"key": key, "n_present": n_present, "n_missing": n_missing,
                       "values": [dict(v) for v in values.values()],
                       "consistent": consistent})

    # 元数据规则引擎达标率（跨图累计）
    th_ok = th_tot = gl_ok = gl_tot = 0
    for r in ok:
        for rr in r.rule_results or []:
            if rr.get("passed") is None:
                continue
            if rr.get("level") == "Threshold":
                th_tot += 1
                th_ok += int(bool(rr["passed"]))
            elif rr.get("level") == "Goal":
                gl_tot += 1
                gl_ok += int(bool(rr["passed"]))
    return {"consistent": all_consistent, "n_images": n_total, "fields": fields,
            "rules": {"threshold_achieved": th_ok, "threshold_total": th_tot,
                      "goal_achieved": gl_ok, "goal_total": gl_tot}}


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def _score_distribution(records: list[ImageRecord]) -> dict:
    dist = {"优": 0, "良": 0, "合格": 0, "不合格": 0, "error": 0}
    for r in records:
        if r.status == "error":
            dist["error"] += 1
        elif r.grade:
            lv = r.grade.get("level")
            if lv in dist:
                dist[lv] += 1
    dist["total"] = len(records)
    return dist


def _record_summary(r: ImageRecord, group_by: Optional[str]) -> dict:
    return {"stem": r.stem, "path": r.path, "status": r.status,
            "score": (r.grade or {}).get("score"),
            "level": (r.grade or {}).get("level"),
            "counts": (r.grade or {}).get("counts"),
            "group": _group_value(r, group_by),
            "error": r.error,
            "report_html": r.report_html, "report_json": r.report_json}


def _write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# 编排入口
# ---------------------------------------------------------------------------
def run_dataset(directory: str, *, level: Optional[str] = None, domain: str = "auto",
                channels: Optional[int] = None, config: Optional[dict] = None,
                metadata_extra: Optional[dict] = None, out_dir: Optional[str] = None,
                per_image: bool = True, recursive: bool = False,
                group_by: Optional[str] = None, batch_size: Optional[int] = None,
                mad_k: float = 3.0, metadata_keys: Optional[tuple] = None,
                exts=IMAGE_EXTS) -> DatasetResult:
    """数据集级质检编排：扫描 → 逐图 → 聚合/批次/离群/一致性/分组 → 落库 + 看板。"""
    from .pipeline import resolve_level
    from .dataset_report import generate_dataset_html

    files = discover_images(directory, exts=exts, recursive=recursive)
    if out_dir is None:
        clean_dir = directory.rstrip("/\\")
        out_dir = f"{clean_dir}_qc"
    per_image_dir = os.path.join(out_dir, "per_image")
    os.makedirs(per_image_dir, exist_ok=True)

    records = [_process_one(f, level=level, domain=domain, channels=channels,
                            config=config, metadata_extra=metadata_extra,
                            per_image=per_image, per_image_dir=per_image_dir) for f in files]

    aggs = aggregate(records, mad_k=mad_k)
    batch = batch_accept(records, batch_size=batch_size)
    g_audit = group_audit(records, group_by=group_by)
    mcons = metadata_consistency(records, keys=metadata_keys or DEFAULT_IDENTITY_KEYS)
    sdist = _score_distribution(records)
    lv = resolve_level(level)

    result = DatasetResult(
        root=out_dir, level=lv, config=config or {}, records=records,
        aggregates=aggs, batch=batch, group_audit=g_audit,
        metadata_consistency=mcons, score_distribution=sdist,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        group_by=group_by,
    )

    _write_json(os.path.join(out_dir, "records.json"),
                {"schema": "dataset-qc/v1", "generated_at": result.generated_at,
                 "root": out_dir, "level": lv,
                 "records": [r.as_dict() for r in records]})
    _write_json(os.path.join(out_dir, "dataset_summary.json"), result.as_dict())
    with open(os.path.join(out_dir, "dashboard.html"), "w", encoding="utf-8") as fh:
        fh.write(generate_dataset_html(result))
    return result
