from pydantic import BaseModel, Field, PrivateAttr
from typing import List, Optional, Dict, Any


class Frame(BaseModel):
    frame_path: str
    timestamp: float


class Segment(BaseModel):
    id: Optional[str] = None
    segment_name: Optional[str] = None
    segment_path: Optional[str] = None
    segment_folder_path: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    segment_duration: Optional[float] = None
    total_frames: Optional[int] = None
    number_of_frames: Optional[int] = None
    segment_frame_time_intervals: List[float] = Field(default_factory=list)
    segment_frames_file_path: List[str] = Field(default_factory=list)
    frames: List[Frame] = Field(default_factory=list)
    segment_prompt_path: Optional[str] = None
    processed: Optional[bool] = False
    analysis_completed: Optional[list] = []
    analyzed_result: Dict[str, Any] = Field(default_factory=dict)
    transcription: Optional[str] = None
    transcription_segments: List[Dict] = []
    chapter: Optional[dict] = None
    is_speech_based: Optional[bool] = False
    dominant_colors_hex: List[str] = []

    # --- ADDED for Blob URL approach ---
    frame_urls: Optional[List[str]] = Field(None, description="SAS URLs for uploaded segment frames")
    _blob_names: Optional[List[str]] = PrivateAttr(default=None) # Store blob names for cleanup
    # --- END ADDED ---

    # --- ADDED for dedicated action extraction ---
    extracted_actions: List[Dict[str, Any]] = Field(default_factory=list, description="Actions extracted specifically for this segment via chunked frame analysis")
    # --- END ADDED ---

    model_config = {
        "extra": "allow", # Allow extra fields if needed during processing
        "validate_assignment": True
    }


class SegmentMetadata(BaseModel):
    effective_duration: Optional[float] = None
    num_segments: Optional[int] = None


class SourceVideoMetadata(BaseModel):
    path: Optional[str] = None
    audio_path: Optional[str] = None
    video_found: Optional[bool] = False
    size: List[int] = Field(default_factory=list)
    rotation: Optional[int] = None
    fps: Optional[float] = None
    duration: Optional[float] = None
    nframes: Optional[int] = None
    audio_found: Optional[bool] = False
    audio_duration: Optional[float] = None
    audio_fps: Optional[int] = None
    format_name: Optional[str] = None
    duration_iso: Optional[str] = None
    language_code: Optional[str] = None          # Detected ISO code: 'en', 'fr', 'ar', 'ary', etc.
    language_detection: Optional[dict] = None    # Raw dictionary from languageIdentifier result
    language_code: Optional[str] = None          # Detected ISO code: 'en', 'fr', 'ar', 'ary', etc.
    language_detection: Optional[dict] = None    # Raw dictionary from languageIdentifier result


class SourceAudioMetadata(BaseModel):
    path: Optional[str] = None
    file_size_mb: Optional[float] = None


class TargetPreprocessingParameters(BaseModel):
    output_directory: Optional[str] = None
    segment_length: Optional[float] = None
    fps: Optional[float] = None
    use_scene_detection: Optional[bool] = False
    generate_transcript_flag: Optional[bool] = True
    trim_to_nearest_second: Optional[bool] = False
    allow_partial_segments: Optional[bool] = True
    use_speech_based_segments: Optional[bool] = False
    downscaled_resolution: Optional[List[int]] = None
    run_timestamp: Optional[str] = None


class VideoManifest(BaseModel):
    name: Optional[str] = None
    video_manifest_path: Optional[str] = None
    source_video: SourceVideoMetadata = Field(default_factory=SourceVideoMetadata)
    processing_params: TargetPreprocessingParameters = Field(
        default_factory=TargetPreprocessingParameters
    )
    segment_metadata: SegmentMetadata = Field(default_factory=SegmentMetadata)
    segments: List[Segment] = Field(default_factory=list)
    final_summary: Optional[str] = None
    audio_transcription: Optional[Any] = None
    source_audio: SourceAudioMetadata = Field(default_factory=SourceAudioMetadata)
    global_tags: Dict[str, List] = Field(default_factory=lambda: {"persons": [], "actions": [], "objects": []})
    yolo_tags: Optional[list[dict]] = None
    raw_yolo_tags: Optional[list[dict]] = None # Added for raw YOLO output
    refined_yolo_tags: Optional[list[dict]] = None # Added for refined YOLO output
    # Initial token counts from pre-analysis steps like track refinement
    initial_prompt_tokens: Optional[int] = Field(0, description="Prompt tokens from pre-analysis steps like Re-ID")
    initial_completion_tokens: Optional[int] = Field(0, description="Completion tokens from pre-analysis steps like Re-ID")
    initial_image_tokens: Optional[int] = Field(0, description="Image tokens from pre-analysis steps like Re-ID")
