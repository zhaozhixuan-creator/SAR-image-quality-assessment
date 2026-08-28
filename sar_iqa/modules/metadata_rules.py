"""自研模块 3 · 元数据规则引擎（对应 docs/10）。

以 stac-check 的规则结构为范本，把 CEOS-ARD / CARD4L 条款逐条做成规则，
逐条输出 Threshold / Goal 自评（pass / fail / warning + 理由），而非笼统结论。

输入：产品元数据字典（--metadata 边车 JSON）。若无元数据，则全部规则返回"未提供"。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuleResult:
    name: str
    clause: str
    level: str            # Threshold / Goal
    passed: bool | None   # None 表示无法判断（缺元数据）
    achieved: str         # "Threshold" / "Goal" / "None" / "N/A"
    reason: str


class Rule:
    """规则统一接口。"""

    def __init__(self, name: str, clause_ref: str, level: str = "Threshold"):
        self.name = name
        self.clause_ref = clause_ref
        self.level = level

    def check(self, meta: dict) -> RuleResult:
        raise NotImplementedError


class FieldPresenceRule(Rule):
    """字段存在性校验。"""

    def __init__(self, name, clause_ref, keys, level="Threshold"):
        super().__init__(name, clause_ref, level)
        self.keys = keys

    def check(self, meta):
        has = any(k in meta for k in self.keys)
        return RuleResult(
            name=self.name, clause=self.clause_ref, level=self.level,
            passed=has,
            achieved=self.level if has else "None",
            reason="" if has else f"缺少字段: {self.keys[0]}",
        )


# CEOS-ARD / CARD4L 条款清单（节选核心条款）
RULES: list[Rule] = [
    FieldPresenceRule("产品族标识", "1 通用元数据", ["product_family", "family"]),
    FieldPresenceRule("采集参数", "1 通用元数据", ["acquisition", "acquisition_mode", "mode"]),
    FieldPresenceRule("坐标系/参考框架", "1 通用元数据", ["crs", "projection", "reference_frame"]),
    FieldPresenceRule("局部入射角", "2 逐像素元数据", ["incidence_angle", "local_incidence_angle"]),
    FieldPresenceRule("散射面积", "2 逐像素元数据", ["scattering_area", "normalized_scattering_area"]),
    FieldPresenceRule("噪声功率矢量", "3 辐射校正", ["noise_range", "noise_azimuth", "noise_vector", "nesz"]),
    FieldPresenceRule("几何精度数值", "4.3 Geometric Accuracy", ["geometric_accuracy", "ale", "geolocation_accuracy"]),
    FieldPresenceRule("几何标准差/质量标记", "4.4 Geometric Refined Accuracy",
                      ["geometric_sigma", "geometric_std", "quality_flag"]),
    FieldPresenceRule("定标因子", "版本台账", ["calibration_factor", "calibration_constant", "sigma0_lut"]),
    FieldPresenceRule("处理器版本", "版本台账", ["processor_version", "software_version"]),
    FieldPresenceRule("辅助数据版本", "版本台账", ["auxiliary_version", "orbit_version", "attitude_version"]),
    FieldPresenceRule("相干性估计窗口", "干涉质量", ["coherence_kernel", "coherence_window"], level="Goal"),
]


def run_rules(meta: dict | None) -> list[RuleResult]:
    if not meta:
        return [
            RuleResult(name=r.name, clause=r.clause_ref, level=r.level,
                       passed=None, achieved="N/A", reason="未提供元数据边车")
            for r in RULES
        ]
    return [r.check(meta) for r in RULES]


def summarize(results: list[RuleResult]) -> dict:
    """汇总自评：Threshold / Goal 达成率。"""
    thresh = [r for r in results if r.level == "Threshold" and r.passed is not None]
    goal = [r for r in results if r.level == "Goal" and r.passed is not None]
    th_ok = sum(1 for r in thresh if r.passed)
    gl_ok = sum(1 for r in goal if r.passed)
    return {
        "threshold_achieved": th_ok,
        "threshold_total": len(thresh),
        "goal_achieved": gl_ok,
        "goal_total": len(goal),
    }
