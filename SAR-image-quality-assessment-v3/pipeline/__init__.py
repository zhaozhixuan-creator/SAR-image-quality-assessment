"""v3 流水线阶段模块。每个 stage 是一个独立模块，暴露 ``run(cfg, args)``。

各 stage 之间通过磁盘产物（workspace/results 下的 JSON 契约）解耦，后续可 1:1
映射为独立 agent，无需改动相互接口。
"""
