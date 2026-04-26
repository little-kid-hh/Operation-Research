#!/usr/bin/env python3
"""
小规模 / 试跑 ReEvo：在本文件顶部的 CONFIG 里改 API 与超参数，然后在项目根目录执行:

    python run_reevo.py

说明:
- API 密钥可写在 CONFIG["api_key"]，或留空并在系统环境变量里设置（见 PROVIDER_ENV）。
- 不要提交含真实密钥的 CONFIG；建议密钥只放在环境变量里，CONFIG["api_key"] 留空。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# =============================================================================
# 用户配置 —— 只改这里
#
# BPP 约 1 分钟试跑示例（需已配置 API）：
#   "problem": "bpp_offline_aco",
#   "max_fe": 6, "pop_size": 2, "init_pop_size": 2, "timeout": 120,
#   "temperature": 0.35,
#   "eval_env": {"BPP_SAMPLE_COUNT": "24", "BPP_N_ANTS": "6"},
# 说明：默认 BPP 每次评估要跑很多次采样，比 LLM 还慢；必须压低 BPP_SAMPLE_COUNT /
# BPP_N_ANTS 才可能压到 1 分钟内（结果仅作 smoke test，不代表论文设置）。
# =============================================================================
CONFIG = {
    # --- LLM（对应 cfg/llm_client/<name>.yaml）---
    "llm_client": "bailian",
    # 若下面留空，请设置环境变量：百炼用 DASHSCOPE_API_KEY，OpenAI 用 OPENAI_API_KEY 等
    "api_key": "",
    "model": "qwen2.5-32b-instruct",
    "temperature": 0.4,
    # 可选：覆盖 base_url（一般不用，百炼默认已在 yaml 里）
    "base_url": None,
    # --- 问题与算法 ---
    "problem": "bpp_offline_aco",
    "algorithm": "reevo",
    # --- 小规模试跑建议值（可自行改大）---
    "max_fe": 6,
    "pop_size": 2,
    "init_pop_size": 2,
    "mutation_rate": 0.5,
    "timeout": 120,
    "diversify_init_pop": True,
    # 传给子进程（eval.py 等）：仅 bpp_offline_aco 会读 BPP_*，其它问题可忽略
    "eval_env": {"BPP_SAMPLE_COUNT": "24", "BPP_N_ANTS": "6"},
    # --- 附加 Hydra 覆盖，例如: ["llm_client@llm_long_ref=null"] ---
    "extra_overrides": [],
}
# =============================================================================

# llm_client 名称 -> 应设置的环境变量名（与 cfg/llm_client 中一致）
PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "bailian": "DASHSCOPE_API_KEY",
    "siliconflow": "SILICONFLOW_API_KEY",
    "zhipuai": "ZHIPUAI_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "llama_api": "LLAMA_API_KEY",
    "azure": "AZURE_API_KEY",
}


def _fmt(v) -> str:
    if isinstance(v, bool):
        return str(v)
    return str(v)


def build_command(root: Path) -> list[str]:
    c = CONFIG
    cmd = [
        sys.executable,
        str(root / "main.py"),
        f"llm_client={c['llm_client']}",
        f"llm_client.model={c['model']}",
        f"llm_client.temperature={c['temperature']}",
        f"problem={c['problem']}",
        f"algorithm={c['algorithm']}",
        f"max_fe={c['max_fe']}",
        f"pop_size={c['pop_size']}",
        f"init_pop_size={c['init_pop_size']}",
        f"mutation_rate={c['mutation_rate']}",
        f"timeout={c['timeout']}",
        f"diversify_init_pop={_fmt(c['diversify_init_pop'])}",
    ]
    if c.get("base_url"):
        cmd.append(f"llm_client.base_url={c['base_url']}")
    for o in c.get("extra_overrides") or []:
        if o:
            cmd.append(o)
    return cmd


def apply_api_key_env() -> None:
    key = (CONFIG.get("api_key") or "").strip()
    name = CONFIG.get("llm_client") or "openai"
    env_name = PROVIDER_ENV.get(name)
    if key and env_name:
        os.environ[env_name] = key


def apply_eval_env() -> None:
    for k, v in (CONFIG.get("eval_env") or {}).items():
        if v is not None and str(v) != "":
            os.environ[str(k)] = str(v)


def main() -> int:
    root = Path(__file__).resolve().parent
    os.chdir(root)

    apply_api_key_env()
    apply_eval_env()
    env_name = PROVIDER_ENV.get(CONFIG.get("llm_client") or "")
    if env_name:
        ok = os.environ.get(env_name) or (CONFIG.get("api_key") or "").strip()
        if not ok:
            print(
                f"错误: 未设置 API 密钥。请在 CONFIG['api_key'] 中填写，"
                f"或设置环境变量 {env_name}。",
                file=sys.stderr,
            )
            return 1

    cmd = build_command(root)
    print("工作目录:", root)
    print("启动命令:", " ".join(cmd[:3]), "...")  # 避免刷屏
    print("完整参数:", cmd)
    return subprocess.call(cmd, cwd=root, env=os.environ.copy())


if __name__ == "__main__":
    raise SystemExit(main())
