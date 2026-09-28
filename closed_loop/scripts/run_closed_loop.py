#!/usr/bin/env python3
"""闭环反馈主入口：自然语言生成需求 → 解析 → 路由 → 评估 → 反馈 → 复评 → 画像。

用法（在 closed_loop/ 目录下）：
    python scripts/run_closed_loop.py --request "参考图换视角，要求背景统计一致"
    python scripts/run_closed_loop.py --request "给 5 类各生成 100 张，方位角均衡" --dry-run

默认「复用模式」：读取 v3 既有评估结果完成反馈与验证（不消耗 GPU）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.requirement_parser import RequirementParser
from agent.router import ResourceRouter
from agent.workflow_planner import WorkflowPlanner
from agent.workflow_executor import WorkflowExecutor


def run(request: str, dry_run: bool = False) -> dict:
    parser = RequirementParser()
    router = ResourceRouter()
    planner = WorkflowPlanner()
    executor = WorkflowExecutor()

    task_schema = parser.parse(request)
    route = router.route(task_schema)
    plan = planner.plan(task_schema, route)

    if dry_run:
        return {"task_schema": task_schema, "route": route, "plan": plan}
    return executor.execute(task_schema, route, plan)


def summarize(result: dict) -> dict:
    return {
        "task_id": result.get("task_id"),
        "status": result.get("status"),
        "selected_model": result.get("selected_model"),
        "selected_variant": result.get("selected_variant"),
        "verdict": result.get("verdict"),
        "feedback_action": (result.get("feedback") or {}).get("action"),
        "improvement": result.get("improvement"),
        "message": result.get("message"),
        "portrait": result.get("portrait"),
        "task_record": result.get("task_record"),
    }


def main() -> None:
    cli = argparse.ArgumentParser(description="SAR 生成图像质检闭环反馈。")
    cli.add_argument("--request", required=True, help="自然语言生成需求。")
    cli.add_argument("--dry-run", action="store_true", help="只解析+路由+规划，不评估/反馈。")
    cli.add_argument("--verbose", action="store_true", help="输出完整 JSON（含 portrait 细节）。")
    args = cli.parse_args()

    result = run(args.request, dry_run=args.dry_run)
    if args.dry_run:
        output = result  # 解析/路由/规划 schema 原样输出
    else:
        output = result if args.verbose else summarize(result)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
