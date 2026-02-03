"""
LTX-2 Video Generation FastAPI Server

Endpoints:
- POST /api/v1/text-to-video - Single-stage text-to-video
- POST /api/v1/image-to-video - Image-to-video (single or two-stage)
- POST /api/v1/video-conditioning - IC-LoRA video conditioning
- GET /api/v1/status/{task_id} - Check generation status
- GET /api/v1/download/{filename} - Download generated video
"""

import logging
import os
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.utils.constants import (
    AUDIO_SAMPLE_RATE,
    DEFAULT_1_STAGE_HEIGHT,
    DEFAULT_1_STAGE_WIDTH,
    DEFAULT_2_STAGE_HEIGHT,
    DEFAULT_2_STAGE_WIDTH,
    DEFAULT_FRAME_RATE,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
    DEFAULT_VIDEO_GUIDER_PARAMS,
    DEFAULT_AUDIO_GUIDER_PARAMS,
)
from ltx_pipelines.utils.media_io import encode_video

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).parent.resolve()
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
OUTPUTS_DIR = BASE_DIR / "outputs" / "api"
INPUTS_DIR = BASE_DIR / "inputs"

# Checkpoint paths
CHECKPOINT_PATH = str(CHECKPOINTS_DIR / "ltx-2-19b-distilled-fp8.safetensors")
GEMMA_ROOT = str(CHECKPOINTS_DIR / "gemma-3-12b")
DISTILLED_LORA_PATH = str(CHECKPOINTS_DIR / "ltx-2-19b-distilled-lora-384.safetensors")
SPATIAL_UPSAMPLER_PATH = str(CHECKPOINTS_DIR / "ltx-2-spatial-upscaler-x2-1.0.safetensors")
IC_LORA_PATH = str(CHECKPOINTS_DIR / "ltx-2-19b-ic-lora-pose-control.safetensors")

# Ensure output directory exists
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Enums & Models
# =============================================================================


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class GuidanceParams(BaseModel):
    """Guidance parameters for video/audio generation."""
    cfg_scale: float = Field(default=DEFAULT_VIDEO_GUIDER_PARAMS.cfg_scale, description="Classifier-free guidance scale")
    stg_scale: float = Field(default=DEFAULT_VIDEO_GUIDER_PARAMS.stg_scale, description="Spatio-Temporal Guidance scale")
    rescale_scale: float = Field(default=DEFAULT_VIDEO_GUIDER_PARAMS.rescale_scale, description="Rescale scale")
    modality_scale: float = Field(default=DEFAULT_VIDEO_GUIDER_PARAMS.modality_scale, description="Cross-modality guidance scale")
    skip_step: int = Field(default=DEFAULT_VIDEO_GUIDER_PARAMS.skip_step, description="Skip step for guidance")
    stg_blocks: list[int] = Field(default=[29], description="Transformer blocks for STG")


class ImageCondition(BaseModel):
    """Image conditioning for video generation."""
    path: str = Field(..., description="Absolute path to the image file")
    frame_idx: int = Field(default=0, description="Target frame index for the image")
    strength: float = Field(default=0.8, ge=0.0, le=1.0, description="Conditioning strength")


class VideoCondition(BaseModel):
    """Video reference conditioning for IC-LoRA."""
    path: str = Field(..., description="Absolute path to the video file")
    strength: float = Field(default=0.2, ge=0.0, le=1.0, description="Conditioning strength")


class LoraConfig(BaseModel):
    """LoRA configuration."""
    path: str = Field(..., description="Path to LoRA file")
    strength: float = Field(default=1.0, ge=0.0, description="LoRA strength")


# =============================================================================
# Request Models
# =============================================================================


class TextToVideoRequest(BaseModel):
    """Request model for text-to-video generation."""
    prompt: str = Field(..., description="Text prompt describing the desired video")
    negative_prompt: str = Field(default=DEFAULT_NEGATIVE_PROMPT, description="Negative prompt")
    seed: int = Field(default=DEFAULT_SEED, description="Random seed for reproducibility")
    height: int = Field(default=DEFAULT_1_STAGE_HEIGHT, description="Video height in pixels (divisible by 32)")
    width: int = Field(default=DEFAULT_1_STAGE_WIDTH, description="Video width in pixels (divisible by 32)")
    num_frames: int = Field(default=DEFAULT_NUM_FRAMES, description="Number of frames (1 + 8*k)")
    frame_rate: float = Field(default=DEFAULT_FRAME_RATE, description="Frame rate (fps)")
    num_inference_steps: int = Field(default=DEFAULT_NUM_INFERENCE_STEPS, description="Denoising steps")
    enhance_prompt: bool = Field(default=False, description="Use AI to enhance the prompt")
    enable_fp8: bool = Field(default=True, description="Enable FP8 mode for lower memory")
    video_guidance: Optional[GuidanceParams] = Field(default=None, description="Video guidance parameters")
    audio_guidance: Optional[GuidanceParams] = Field(default=None, description="Audio guidance parameters")
    loras: list[LoraConfig] = Field(default=[], description="Additional LoRAs to apply")


class ImageToVideoRequest(TextToVideoRequest):
    """Request model for image-to-video generation."""
    images: list[ImageCondition] = Field(default=[], description="Image conditioning inputs")
    two_stage: bool = Field(default=False, description="Use two-stage pipeline for higher quality")


class VideoConditioningRequest(BaseModel):
    """Request model for video conditioning (IC-LoRA) generation."""
    prompt: str = Field(..., description="Text prompt describing the desired video")
    seed: int = Field(default=DEFAULT_SEED, description="Random seed")
    height: int = Field(default=DEFAULT_2_STAGE_HEIGHT, description="Video height (divisible by 64)")
    width: int = Field(default=DEFAULT_2_STAGE_WIDTH, description="Video width (divisible by 64)")
    num_frames: int = Field(default=DEFAULT_NUM_FRAMES, description="Number of frames")
    frame_rate: float = Field(default=DEFAULT_FRAME_RATE, description="Frame rate")
    enhance_prompt: bool = Field(default=False, description="Use AI to enhance the prompt")
    enable_fp8: bool = Field(default=True, description="Enable FP8 mode")
    images: list[ImageCondition] = Field(default=[], description="Image conditioning inputs")
    video_conditioning: list[VideoCondition] = Field(..., description="Video reference conditioning")
    ic_lora_path: Optional[str] = Field(default=None, description="Custom IC-LoRA path (uses default if not set)")
    ic_lora_strength: float = Field(default=1.0, description="IC-LoRA strength")


# =============================================================================
# Response Models
# =============================================================================


class TaskResponse(BaseModel):
    """Response model for task creation."""
    task_id: str
    status: TaskStatus
    message: str


class TaskStatusResponse(BaseModel):
    """Response model for task status query."""
    task_id: str
    status: TaskStatus
    output_path: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


# =============================================================================
# Global State
# =============================================================================


# Task storage (in-memory - use Redis/DB for production)
tasks: dict[str, dict] = {}

# Pipeline cache (lazy loading)
_pipelines: dict[str, object] = {}


def get_pipeline(name: str, **kwargs):
    """Get or create a pipeline instance."""
    global _pipelines
    
    # Clear other pipelines to save GPU memory
    if name not in _pipelines and _pipelines:
        logger.info(f"Clearing existing pipelines to load {name}")
        _pipelines.clear()
        torch.cuda.empty_cache()
    
    if name not in _pipelines:
        logger.info(f"Loading pipeline: {name}")
        if name == "one_stage":
            _pipelines[name] = TI2VidOneStagePipeline(
                checkpoint_path=kwargs.get("checkpoint_path", CHECKPOINT_PATH),
                gemma_root=kwargs.get("gemma_root", GEMMA_ROOT),
                loras=kwargs.get("loras", []),
                fp8transformer=kwargs.get("fp8transformer", True),
            )
        elif name == "two_stage":
            _pipelines[name] = TI2VidTwoStagesPipeline(
                checkpoint_path=kwargs.get("checkpoint_path", CHECKPOINT_PATH),
                distilled_lora=kwargs.get("distilled_lora", [
                    LoraPathStrengthAndSDOps(DISTILLED_LORA_PATH, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)
                ]),
                spatial_upsampler_path=kwargs.get("spatial_upsampler_path", SPATIAL_UPSAMPLER_PATH),
                gemma_root=kwargs.get("gemma_root", GEMMA_ROOT),
                loras=kwargs.get("loras", []),
                fp8transformer=kwargs.get("fp8transformer", True),
            )
        elif name == "ic_lora":
            _pipelines[name] = ICLoraPipeline(
                checkpoint_path=kwargs.get("checkpoint_path", CHECKPOINT_PATH),
                spatial_upsampler_path=kwargs.get("spatial_upsampler_path", SPATIAL_UPSAMPLER_PATH),
                gemma_root=kwargs.get("gemma_root", GEMMA_ROOT),
                loras=kwargs.get("loras", []),
                fp8transformer=kwargs.get("fp8transformer", True),
            )
        logger.info(f"Pipeline {name} loaded successfully")
    
    return _pipelines[name]


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="LTX-2 Video Generation API",
    description="API for text-to-video, image-to-video, and video conditioning generation",
    version="1.0.0",
)


# =============================================================================
# Background Task Processors
# =============================================================================


def _create_lora_list(loras: list[LoraConfig]) -> list[LoraPathStrengthAndSDOps]:
    """Convert LoraConfig list to LoraPathStrengthAndSDOps list."""
    return [
        LoraPathStrengthAndSDOps(lora.path, lora.strength, LTXV_LORA_COMFY_RENAMING_MAP)
        for lora in loras
    ]


def _get_guidance_params(guidance: Optional[GuidanceParams], default: MultiModalGuiderParams) -> MultiModalGuiderParams:
    """Convert GuidanceParams to MultiModalGuiderParams."""
    if guidance is None:
        return default
    return MultiModalGuiderParams(
        cfg_scale=guidance.cfg_scale,
        stg_scale=guidance.stg_scale,
        rescale_scale=guidance.rescale_scale,
        modality_scale=guidance.modality_scale,
        skip_step=guidance.skip_step,
        stg_blocks=guidance.stg_blocks,
    )


@torch.inference_mode()
def process_text_to_video(task_id: str, request: TextToVideoRequest):
    """Background task for text-to-video generation."""
    try:
        tasks[task_id]["status"] = TaskStatus.PROCESSING
        
        loras = _create_lora_list(request.loras)
        pipeline = get_pipeline("one_stage", loras=loras, fp8transformer=request.enable_fp8)
        
        output_filename = f"{task_id}.mp4"
        output_path = str(OUTPUTS_DIR / output_filename)
        
        video, audio = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            height=request.height,
            width=request.width,
            num_frames=request.num_frames,
            frame_rate=request.frame_rate,
            num_inference_steps=request.num_inference_steps,
            video_guider_params=_get_guidance_params(request.video_guidance, DEFAULT_VIDEO_GUIDER_PARAMS),
            audio_guider_params=_get_guidance_params(request.audio_guidance, DEFAULT_AUDIO_GUIDER_PARAMS),
            images=[],
            enhance_prompt=request.enhance_prompt,
        )
        
        encode_video(
            video=video,
            fps=request.frame_rate,
            audio=audio,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            output_path=output_path,
            video_chunks_number=1,
        )
        
        tasks[task_id]["status"] = TaskStatus.COMPLETED
        tasks[task_id]["output_path"] = output_filename
        tasks[task_id]["completed_at"] = datetime.now().isoformat()
        logger.info(f"Task {task_id} completed: {output_path}")
        
    except Exception as e:
        logger.exception(f"Task {task_id} failed")
        tasks[task_id]["status"] = TaskStatus.FAILED
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["completed_at"] = datetime.now().isoformat()


@torch.inference_mode()
def process_image_to_video(task_id: str, request: ImageToVideoRequest):
    """Background task for image-to-video generation."""
    try:
        tasks[task_id]["status"] = TaskStatus.PROCESSING
        
        loras = _create_lora_list(request.loras)
        images = [(img.path, img.frame_idx, img.strength) for img in request.images]
        
        output_filename = f"{task_id}.mp4"
        output_path = str(OUTPUTS_DIR / output_filename)
        
        if request.two_stage:
            pipeline = get_pipeline("two_stage", loras=loras, fp8transformer=request.enable_fp8)
            tiling_config = TilingConfig.default()
            video_chunks_number = get_video_chunks_number(request.num_frames, tiling_config)
            
            video, audio = pipeline(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                seed=request.seed,
                height=request.height,
                width=request.width,
                num_frames=request.num_frames,
                frame_rate=request.frame_rate,
                num_inference_steps=request.num_inference_steps,
                video_guider_params=_get_guidance_params(request.video_guidance, DEFAULT_VIDEO_GUIDER_PARAMS),
                audio_guider_params=_get_guidance_params(request.audio_guidance, DEFAULT_AUDIO_GUIDER_PARAMS),
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=request.enhance_prompt,
            )
        else:
            pipeline = get_pipeline("one_stage", loras=loras, fp8transformer=request.enable_fp8)
            video_chunks_number = 1
            
            video, audio = pipeline(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                seed=request.seed,
                height=request.height,
                width=request.width,
                num_frames=request.num_frames,
                frame_rate=request.frame_rate,
                num_inference_steps=request.num_inference_steps,
                video_guider_params=_get_guidance_params(request.video_guidance, DEFAULT_VIDEO_GUIDER_PARAMS),
                audio_guider_params=_get_guidance_params(request.audio_guidance, DEFAULT_AUDIO_GUIDER_PARAMS),
                images=images,
                enhance_prompt=request.enhance_prompt,
            )
        
        encode_video(
            video=video,
            fps=request.frame_rate,
            audio=audio,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            output_path=output_path,
            video_chunks_number=video_chunks_number,
        )
        
        tasks[task_id]["status"] = TaskStatus.COMPLETED
        tasks[task_id]["output_path"] = output_filename
        tasks[task_id]["completed_at"] = datetime.now().isoformat()
        logger.info(f"Task {task_id} completed: {output_path}")
        
    except Exception as e:
        logger.exception(f"Task {task_id} failed")
        tasks[task_id]["status"] = TaskStatus.FAILED
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["completed_at"] = datetime.now().isoformat()


@torch.inference_mode()
def process_video_conditioning(task_id: str, request: VideoConditioningRequest):
    """Background task for video conditioning (IC-LoRA) generation."""
    try:
        tasks[task_id]["status"] = TaskStatus.PROCESSING
        
        ic_lora_path = request.ic_lora_path or IC_LORA_PATH
        loras = [LoraPathStrengthAndSDOps(ic_lora_path, request.ic_lora_strength, LTXV_LORA_COMFY_RENAMING_MAP)]
        
        pipeline = get_pipeline("ic_lora", loras=loras, fp8transformer=request.enable_fp8)
        
        images = [(img.path, img.frame_idx, img.strength) for img in request.images]
        video_cond = [(vc.path, vc.strength) for vc in request.video_conditioning]
        
        output_filename = f"{task_id}.mp4"
        output_path = str(OUTPUTS_DIR / output_filename)
        
        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(request.num_frames, tiling_config)
        
        video, audio = pipeline(
            prompt=request.prompt,
            seed=request.seed,
            height=request.height,
            width=request.width,
            num_frames=request.num_frames,
            frame_rate=request.frame_rate,
            images=images,
            video_conditioning=video_cond,
            enhance_prompt=request.enhance_prompt,
            tiling_config=tiling_config,
        )
        
        encode_video(
            video=video,
            fps=request.frame_rate,
            audio=audio,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            output_path=output_path,
            video_chunks_number=video_chunks_number,
        )
        
        tasks[task_id]["status"] = TaskStatus.COMPLETED
        tasks[task_id]["output_path"] = output_filename
        tasks[task_id]["completed_at"] = datetime.now().isoformat()
        logger.info(f"Task {task_id} completed: {output_path}")
        
    except Exception as e:
        logger.exception(f"Task {task_id} failed")
        tasks[task_id]["status"] = TaskStatus.FAILED
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["completed_at"] = datetime.now().isoformat()


# =============================================================================
# API Endpoints
# =============================================================================


@app.post("/api/v1/text-to-video", response_model=TaskResponse)
async def text_to_video(request: TextToVideoRequest, background_tasks: BackgroundTasks):
    """
    Generate video from text prompt using single-stage pipeline.
    
    Returns a task_id that can be used to check status and download the result.
    """
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": TaskStatus.PENDING,
        "created_at": datetime.now().isoformat(),
        "type": "text-to-video",
    }
    
    background_tasks.add_task(process_text_to_video, task_id, request)
    
    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="Video generation started. Use /api/v1/status/{task_id} to check progress.",
    )


@app.post("/api/v1/image-to-video", response_model=TaskResponse)
async def image_to_video(request: ImageToVideoRequest, background_tasks: BackgroundTasks):
    """
    Generate video from image(s) and text prompt.
    
    Use two_stage=true for higher quality output (2x resolution with refinement).
    Returns a task_id that can be used to check status and download the result.
    """
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": TaskStatus.PENDING,
        "created_at": datetime.now().isoformat(),
        "type": "image-to-video",
        "two_stage": request.two_stage,
    }
    
    background_tasks.add_task(process_image_to_video, task_id, request)
    
    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message=f"Image-to-video generation started ({'two-stage' if request.two_stage else 'single-stage'}). Use /api/v1/status/{{task_id}} to check progress.",
    )


@app.post("/api/v1/video-conditioning", response_model=TaskResponse)
async def video_conditioning(request: VideoConditioningRequest, background_tasks: BackgroundTasks):
    """
    Generate video with IC-LoRA conditioning from reference video.
    
    Use this for pose transfer, motion control, or style transfer applications.
    Returns a task_id that can be used to check status and download the result.
    """
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": TaskStatus.PENDING,
        "created_at": datetime.now().isoformat(),
        "type": "video-conditioning",
    }
    
    background_tasks.add_task(process_video_conditioning, task_id, request)
    
    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="Video conditioning generation started. Use /api/v1/status/{task_id} to check progress.",
    )


@app.get("/api/v1/status/{task_id}", response_model=TaskStatusResponse)
async def get_status(task_id: str):
    """Check the status of a video generation task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        output_path=task.get("output_path"),
        error=task.get("error"),
        created_at=task["created_at"],
        completed_at=task.get("completed_at"),
    )


@app.get("/api/v1/download/{filename}")
async def download_video(filename: str):
    """Download a generated video file."""
    file_path = OUTPUTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=filename,
    )


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LTX-2 Video Generation API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "text_to_video": "POST /api/v1/text-to-video",
            "image_to_video": "POST /api/v1/image-to-video",
            "video_conditioning": "POST /api/v1/video-conditioning",
            "status": "GET /api/v1/status/{task_id}",
            "download": "GET /api/v1/download/{filename}",
            "health": "GET /api/v1/health",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
