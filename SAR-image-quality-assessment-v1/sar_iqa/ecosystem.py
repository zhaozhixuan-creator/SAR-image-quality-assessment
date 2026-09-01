"""开源生态选型与许可合规注册表（对应 README §八 + 各维度「实现」列）。

把参考开源生态结构化：名称 / 仓库 / 许可证 / 角色 / 合规提示。
spec.py 中各指标的 refs 以本模块的工具键（Tool.name）引用。

许可合规口径（README §六.2）：
- permissive   MIT / Apache-2.0 / BSD —— 可集成
- gpl          GPL-3.0 —— 只宜研读，不可链接 / 分发
- check        NOASSERTION —— 需逐一核对许可证
- noncomm      含非商业条款 —— 需取得商业授权
- data/service 数据集 / 公开服务 —— 仅作交叉校验参考
"""
from __future__ import annotations

from dataclasses import dataclass

# 许可类别
PERMISSIVE = "permissive"
GPL = "gpl"
CHECK = "check"
NONCOMM = "noncomm"
DATA = "data"


@dataclass(frozen=True)
class Tool:
    name: str            # 工具键（spec.refs 引用此键）
    repo: str            # 仓库 / 显示名
    url: str = ""        # 主页 / 仓库地址
    license: str = ""    # 许可证标识
    category: str = CHECK  # 合规类别
    role: str = ""       # 在本方案中的角色


# ---------------------------------------------------------------------------
# 工具注册表。键与 spec.py 的 refs 一致。
# ---------------------------------------------------------------------------
TOOLS: dict[str, Tool] = {
    # —— 核心质检引擎 ——
    "sct": Tool("sct", "aresys-srl/sct", "https://github.com/aresys-srl/sct",
                "MIT", PERMISSIVE, "IRF/PSLR/ISLR/SSLR/RCS/SCR/ALE/NESZ/扇贝/模糊比全套 CLI"),
    # —— 噪声与伪影 ——
    "sentinel1denoised": Tool("sentinel1denoised", "nansencenter/sentinel1denoised",
                              "https://github.com/nansencenter/sentinel1denoised",
                              "GPL-3.0", GPL, "热噪声减除（负值率参考）"),
    "xarray-sentinel": Tool("xarray-sentinel", "bopen/xarray-sentinel",
                            "https://github.com/bopen/xarray-sentinel",
                            "Apache-2.0", PERMISSIVE, "NESZ 查找表"),
    # —— 几何与定标 ——
    "sarsen": Tool("sarsen", "bopen/sarsen", "https://github.com/bopen/sarsen",
                   "Apache-2.0", PERMISSIVE, "RTC 地形校正 / 叠掩阴影"),
    "gecoris": Tool("gecoris", "csiro-auscalval/gecoris", "https://github.com/csiro-auscalval/gecoris",
                    "GPL-3.0", GPL, "角反射器定标 / 相位噪声"),
    "opensartoolkit": Tool("opensartoolkit", "OpenSarToolkit", "https://github.com/birgander2/OpenSARkit",
                           "", CHECK, "叠掩阴影掩膜"),
    # —— 干涉质量 ——
    "licsbas": Tool("licsbas", "yumorishita/LiCSBAS", "https://github.com/yumorishita/LiCSBAS",
                    "GPL-3.0", GPL, "相位闭合残差 / InSAR QC"),
    "mintpy": Tool("mintpy", "insarlab/MintPy", "https://github.com/insarlab/MintPy",
                   "NOASSERTION", CHECK, "配准 / 相位噪声 / 轨道斜坡"),
    "dolphin": Tool("dolphin", "isce-framework/dolphin", "https://github.com/isce-framework/dolphin",
                    "NOASSERTION", CHECK, "干涉质量"),
    "gmtsar": Tool("gmtsar", "gmtsar/gmtsar", "https://github.com/gmtsar/gmtsar",
                   "", CHECK, "轨道残余斜坡"),
    "raider": Tool("raider", "dbekaert/RAiDER", "https://github.com/dbekaert/RAiDER",
                   "", CHECK, "大气相位屏"),
    "pyaps": Tool("pyaps", "insarlab/PyAPS", "https://github.com/insarlab/PyAPS",
                  "", CHECK, "大气相位屏"),
    "pysolid": Tool("pysolid", "insarlab/PySolid", "https://github.com/insarlab/PySolid",
                    "", CHECK, "固体潮改正"),
    # —— InSAR 处理（几何/配准） ——
    "isce2": Tool("isce2", "isce-framework/isce2", "https://github.com/isce-framework/isce2",
                  "NOASSERTION", CHECK, "配准 / 几何"),
    "isce3": Tool("isce3", "isce-framework/isce3", "https://github.com/isce-framework/isce3",
                  "NOASSERTION", CHECK, "RTC / 配准 / 干涉"),
    # —— ARD 与元数据 ——
    "s1ard": Tool("s1ard", "SAR-ARD/s1ard", "https://github.com/SAR-ARD/s1ard",
                  "Apache-2.0", PERMISSIVE, "CARD4L 注释层与合规输出"),
    "stac-check": Tool("stac-check", "stac-utils/stac-check", "https://github.com/stac-utils/stac-check",
                       "MIT", PERMISSIVE, "元数据规则引擎骨架"),
    "s1tiling": Tool("s1tiling", "CNES/s1tiling", "https://github.com/CNES/s1tiling",
                     "", CHECK, "有效覆盖率按瓦片核算"),
    # —— 去斑与评价 ——
    "deepdespeckling": Tool("deepdespeckling", "hi-paris/deepdespeckling",
                            "https://github.com/hi-paris/deepdespeckling",
                            "", CHECK, "去斑基线算法"),
    "sewar": Tool("sewar", "andrewekhalel/sewar", "https://github.com/andrewekhalel/sewar",
                  "", CHECK, "去斑评价指标"),
    "unassisted": Tool("unassisted", "Raydonal/UNASSISTED", "https://github.com/Raydonal/UNASSISTED",
                       "", CHECK, "M-index 权威参考实现"),
    # —— RFI ——
    "sentinel1-rfi-detection": Tool("sentinel1-rfi-detection", "zephr-xyz/sentinel1-rfi-detection",
                                    "https://github.com/zephr-xyz/sentinel1-rfi-detection",
                                    "", CHECK, "S1 SLC 距离谱 GNSS 干扰检测（算法参考）"),
    "s1rfimap": Tool("s1rfimap", "ESA / Aresys s1rfimap", "",
                     "", DATA, "公开 RFI 地图（外部交叉校验）"),
    # —— 数据集 ——
    "sar-iids": Tool("sar-iids", "SAR-IIDS 干扰数据集", "",
                     "", DATA, "Sentinel-1 RFI 标注数据集"),
    "rfisd": Tool("rfisd", "RFISD 数据集", "", "", DATA, "RFI 标注数据集"),
    # —— 内部占位（自研 / 版本台账） ——
    "self": Tool("self", "自研（本引擎）", "", "Proprietary", PERMISSIVE, "自研补缺模块"),
    "version-ledger": Tool("version-ledger", "版本台账（内部）", "", "Proprietary", PERMISSIVE,
                           "阈值 / 定标与处理器、辅助数据版本绑定审计"),
}

# 许可合规提示文案
COMPLIANCE_NOTE = {
    PERMISSIVE: "可集成 / 可引用",
    GPL: "GPL-3.0 —— 只宜研读，不可链接 / 分发",
    CHECK: "许可证待核对（NOASSERTION / 未标注）—— 集成前逐一确认",
    NONCOMM: "含非商业条款 —— 商用需取得授权",
    DATA: "数据集 / 公开服务 —— 仅作交叉校验参考",
}


def get_tool(name: str) -> Tool | None:
    return TOOLS.get(name)


def resolve_refs(refs) -> list[Tool]:
    """把指标 refs（工具键列表）解析为 Tool 列表，未知键跳过。"""
    return [TOOLS[k] for k in refs if k in TOOLS]


def compliance_summary() -> dict:
    """按许可类别汇总生态选型（供报告「开源生态与许可合规」小节）。"""
    groups: dict[str, list[str]] = {c: [] for c in COMPLIANCE_NOTE}
    for name, t in TOOLS.items():
        groups.setdefault(t.category, []).append(name)
    return {
        "total": len(TOOLS),
        "by_category": {c: names for c, names in groups.items() if names},
        "note": dict(COMPLIANCE_NOTE),
    }
