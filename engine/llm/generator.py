"""LLM 报告生成。

默认使用 OpenAI 兼容协议，可对接 DeepSeek、OpenAI、Anthropic（通过兼容层）等。
未配置 API key 时返回基于规则的 mock 报告。
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx


def is_configured() -> bool:
    return bool(os.environ.get("XUNLONG_LLM_API_KEY", ""))


def _build_prompt(
    candidates: list[dict],
    terrain: dict,
    style: str = "traditional",
    language: str = "zh",
) -> str:
    """构造 prompt。"""
    is_zh = language == "zh"

    if is_zh:
        sys_prompt = (
            "你是传统风水学研究员 + GIS 分析师。请基于提供的 GIS 计算结果，"
            "撰写一份'环境格局评估报告'。要求：\n"
            "1. 必须严格基于输入数据，禁止凭空预测吉凶/财运/寿命。\n"
            "2. 解释每个候选穴的'四兽'、'明堂'、'得水'等指标的含义。\n"
            "3. 不超过 800 字，使用条目化结构。\n"
            "4. 结尾注明：本报告基于 GIS 地形分析，仅供传统文化研究参考。"
        )
    else:
        sys_prompt = (
            "You are a traditional Feng Shui researcher and GIS analyst. "
            "Write a 'landscape pattern assessment report' based on the provided GIS results. "
            "Requirements:\n"
            "1. Strictly based on the input data; do not invent predictions.\n"
            "2. Explain each candidate's four-beasts, Mingtang, water relation.\n"
            "3. Use bullet points, max 600 words.\n"
            "4. End with disclaimer: 'For cultural reference only.'"
        )

    terrain_summary = json.dumps(terrain, ensure_ascii=False, indent=2)
    cand_summary = json.dumps(
        [
            {
                "id": c["id"],
                "x": c.get("x"),
                "y": c.get("y"),
                "elevation_m": c.get("elevation_m"),
                "form_type": c.get("form_type"),
                "overall_score": c.get("overall_score"),
                "scores": c.get("scores"),
                "geography": c.get("geography"),
            }
            for c in candidates[:5]
        ],
        ensure_ascii=False,
        indent=2,
    )

    if is_zh:
        user = f"## 区域地形指标\n```json\n{terrain_summary}\n```\n\n## 候选穴（Top 5）\n```json\n{cand_summary}\n```\n\n请撰写格局评估报告。"
    else:
        user = (
            f"## Regional Terrain\n```json\n{terrain_summary}\n```\n\n"
            f"## Top Candidates\n```json\n{cand_summary}\n```\n\n"
            f"Please write the assessment report."
        )
    return sys_prompt, user


def _mock_report(candidates: list[dict], terrain: dict, language: str = "zh") -> str:
    """未配置 LLM 时返回基于规则的简洁报告。"""
    if not candidates:
        return "（无可用候选穴）"

    top = candidates[0]
    score = top.get("overall_score", 0)
    form = top.get("form_type", "")
    elev = top.get("elevation_m", 0)
    water_dir = top.get("geography", {}).get("nearest_water_dir", "未知")
    water_dist = top.get("geography", {}).get("nearest_water_m", "未知")

    if language == "zh":
        return (
            f"## 环境格局评估（基于规则）\n\n"
            f"**区域地形**：均高 {terrain.get('mean_elevation', 0):.0f} m，"
            f"高差 {terrain.get('relief', 0):.0f} m，"
            f"主坡向 {terrain.get('dominant_aspect')}，"
            f"类型 {terrain.get('terrain_position')}。\n\n"
            f"**最佳候选穴**：{top.get('id')}（综合分 {score}）\n"
            f"- 形态：{form}\n"
            f"- 高程：{elev:.0f} m\n"
            f"- 距最近水体（{water_dir}）：{water_dist} m\n"
            f"- 四象分：龙 {top.get('geography', {}).get('qinglong', '-')} / "
            f"虎 {top.get('geography', {}).get('baihu', '-')} / "
            f"雀 {top.get('geography', {}).get('zhuque', '-')} / "
            f"武 {top.get('geography', {}).get('xuanwu', '-')}\n\n"
            f"**说明**：本报告基于 GIS 地形指标，未使用 LLM。配置 LLM_API_KEY 后可获得更丰富解读。\n\n"
            f"---\n*本报告基于 GIS 地形分析，仅供传统文化研究参考。*"
        )
    return (
        f"## Pattern Assessment (rule-based)\n\n"
        f"**Region**: mean elev {terrain.get('mean_elevation', 0):.0f} m, "
        f"relief {terrain.get('relief', 0):.0f} m, "
        f"aspect {terrain.get('dominant_aspect')}.\n\n"
        f"**Top candidate**: {top.get('id')} (score {score})\n"
        f"- Form: {form}\n- Elevation: {elev:.0f} m\n"
        f"- Nearest water ({water_dir}): {water_dist} m\n\n"
        f"*For cultural reference only.*"
    )


def generate_report(
    candidates: list[dict],
    terrain: dict,
    style: str = "traditional",
    language: str = "zh",
) -> dict[str, Any]:
    """生成 LLM 报告。

    Returns:
        dict { report, model, prompt_tokens, completion_tokens, mock }
    """
    api_key = os.environ.get("XUNLONG_LLM_API_KEY", "")
    base_url = os.environ.get("XUNLONG_LLM_BASE_URL", "https://api.deepseek.com/v1")
    model = os.environ.get("XUNLONG_LLM_MODEL", "deepseek-chat")

    if not api_key:
        return {
            "report": _mock_report(candidates, terrain, language),
            "model": "rule-based-mock",
            "mock": True,
        }

    sys_prompt, user_prompt = _build_prompt(candidates, terrain, style, language)
    try:
        resp = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.5,
                "max_tokens": 1500,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "report": data["choices"][0]["message"]["content"],
            "model": model,
            "prompt_tokens": data.get("usage", {}).get("prompt_tokens"),
            "completion_tokens": data.get("usage", {}).get("completion_tokens"),
            "mock": False,
        }
    except Exception as e:
        return {
            "report": f"⚠️ LLM 调用失败：{e}\n\n" + _mock_report(candidates, terrain, language),
            "model": model,
            "mock": True,
        }
