"""Pydantic 数据模型。"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class BBox(BaseModel):
    """边界框 (minx, miny, maxx, maxy)。"""

    minx: float
    miny: float
    maxx: float
    maxy: float


class AnalyzeRequest(BaseModel):
    """通用分析请求。"""

    dem_path: str = Field(..., description="DEM GeoTIFF 路径")
    water_path: Optional[str] = Field(None, description="水系 GeoJSON 路径")
    bbox: Optional[BBox] = Field(None, description="可选裁剪范围")
    top_k: int = Field(10, ge=1, le=100)
    min_score: int = Field(40, ge=0, le=100)
    tpi_radius_m: float = Field(100.0, ge=10, le=1000)
    facing: Optional[str] = Field(None, description="坐向：南/东南/东等")


class TerrainResult(BaseModel):
    """地形分析结果。"""

    bbox: list[float]
    resolution_m: float
    mean_elevation: float
    max_elevation: float
    min_elevation: float
    relief: float
    mean_slope: float
    max_slope: float
    dominant_aspect: str
    aspect_degree: float
    terrain_position: str
    terrain_roughness: float


class CandidateItem(BaseModel):
    """单个候选穴。"""

    id: str
    rank: int
    x: float
    y: float
    elevation_m: float
    form_type: str
    overall_score: int
    scores: dict[str, Optional[int]]
    geography: dict[str, Any]
    messages: dict[str, str]
    # 显示项（如 mouth/compass/xuankong 是否进 overall 等元数据）
    meta: dict[str, Any] = Field(default_factory=dict)


class CandidatesResponse(BaseModel):
    """候选穴列表。"""

    metadata: dict[str, Any]
    candidates: list[CandidateItem]


class RidgeLineItem(BaseModel):
    """山脊线段。"""

    length_m: float
    mean_elevation: float
    max_elevation: float
    sinuosity: float
    feature_significance: float
    coords: list[list[int]]


class DragonVeinResponse(BaseModel):
    """龙脉识别结果。"""

    n_ridges: int
    n_major: int
    entrance_xy: Optional[list[float]]
    major_ridges: list[RidgeLineItem]
    # A1-余：蜂腰鹤膝过峡点数（明细见各候选 geography / 后续可扩展）
    n_yaoxia: int = 0


class LLMInterpretRequest(BaseModel):
    """LLM 解读请求。"""

    candidates: list[CandidateItem]
    terrain: dict[str, Any] = Field(
        default_factory=dict,
        description="地形分析结果（dict，可来自 /api/terrain/analyze 或自定义）",
    )
    style: str = Field("traditional", description="traditional / modern / academic")
    language: str = Field("zh", description="zh / en")


class LLMInterpretResponse(BaseModel):
    """LLM 解读结果。"""

    report: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    mock: bool = False
