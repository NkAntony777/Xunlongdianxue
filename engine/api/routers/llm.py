"""LLM 解读路由。"""
from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException

from engine.api.schemas.models import LLMInterpretRequest, LLMInterpretResponse
from engine.llm import generate_report, is_configured

router = APIRouter()


@router.get("/status")
def status():
    """查询 LLM 配置状态。"""
    return {
        "configured": is_configured(),
        "model": os.environ.get("XUNLONG_LLM_MODEL", "deepseek-chat"),
        "base_url": os.environ.get("XUNLONG_LLM_BASE_URL", "https://api.deepseek.com/v1"),
    }


@router.post("/interpret", response_model=LLMInterpretResponse)
def interpret(req: LLMInterpretRequest):
    """对候选穴进行 LLM 解读（生成文化性报告）。"""
    try:
        result = generate_report(
            candidates=[c.model_dump() for c in req.candidates],
            terrain=req.terrain,
            style=req.style,
            language=req.language,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"llm failed: {e}")
    return LLMInterpretResponse(**result)
