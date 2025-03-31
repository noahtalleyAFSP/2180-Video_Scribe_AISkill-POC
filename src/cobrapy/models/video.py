from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class Segment(BaseModel):
    segment_name: Optional[str] = None
    segment_folder_path: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    segment_duration: Optional[float] = None
    number_of_frames: Optional[int] = None
    segment_frame_time_intervals: List[float] = Field(default_factory=list)
    segment_frames_file_path: List[str] = Field(default_factory=list)
    segment_prompt_path: Optional[str] = None
    processed: Optional[bool] = False
    analysis_completed: Optional[list] = []
    analyzed_result: Optional[dict] = {}
    transcription: Optional[str] = None
    chapter: Optional[dict] = None
    is_speech_based: Optional[bool] = False


class SegmentMetadata(BaseModel):
    effective_duration: Optional[float] = None
    num_segments: Optional[int] = None


class SourceVideoMetadata(BaseModel):
    path: Optional[str] = None
    video_found: Optional[bool] = False
    size: List[int] = Field(default_factory=list)
    rotation: Optional[int] = None
    fps: Optional[float] = None
    duration: Optional[float] = None
    nframes: Optional[int] = None
    audio_found: Optional[bool] = False
    audio_duration: Optional[float] = None
    audio_fps: Optional[int] = None


class SourceAudioMetadata(BaseModel):
    path: Optional[str] = None
    file_size_mb: Optional[float] = None


class TargetPreprocessingParameters(BaseModel):
    output_directory: Optional[str] = None
    segment_length: Optional[float] = None
    fps: Optional[float] = None
    generate_transcript_flag: Optional[bool] = True
    trim_to_nearest_second: Optional[bool] = False
    allow_partial_segments: Optional[bool] = True
    use_speech_based_segments: Optional[bool] = False


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
    audio_transcription: Optional[Dict] = None
    source_audio: SourceAudioMetadata = Field(default_factory=SourceAudioMetadata)
    global_tags: Dict[str, List] = Field(default_factory=lambda: {"persons": [], "actions": [], "objects": []})
