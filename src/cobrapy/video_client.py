import os
from typing import Union, Type
from ast import literal_eval
from dotenv import load_dotenv
from .cobra_utils import get_file_info

from .video_preprocessor import VideoPreProcessor
from .video_analyzer import VideoAnalyzer
from .models.video import VideoManifest, SourceVideoMetadata
from .models.environment import CobraEnvironment
from .analysis import AnalysisConfig
from .cobra_utils import (
    validate_video_manifest,
    write_video_manifest,
)


class VideoClient:
    manifest: VideoManifest
    video_path: str
    env_file_path: str
    env: CobraEnvironment
    preprocessor: VideoPreProcessor
    analyzer: VideoAnalyzer
    upload_to_azure: bool

    def __init__(
        self,
        video_path: Union[str, None] = None,
        manifest: Union[str, VideoManifest, None] = None,
        env_file_path: str = None,
        upload_to_azure: bool = False,
        # connection_config_list: List[Dict[str, str]] = None, # Not Implemented Yet
    ):
        # Video path is required if manifest is not provided
        if video_path is None and manifest is None:
            raise ValueError(
                "You must either provide a video_path to an input video or the manifest parameter. The manifest parameter can be a string path to a manifest json file or a VideoManifest object."
            )

        # If the manifest is not provided, create a new one
        # If manifest is provided, validate it is the correct type
        if manifest is None:
            manifest = self._prepare_video_manifest(video_path)
        else:
            manifest = validate_video_manifest(manifest)

        self.manifest = manifest

        # If the environment file path is set, attempt to load the environment variables from the file
        self.env_file_path = env_file_path

        if self.env_file_path is not None:
            load_dotenv(dotenv_path=self.env_file_path, override=True)

        # Load the environment variables in the pydantic model
        self.env = CobraEnvironment()

        # Initialize the preprocessor and analyzer
        self.preprocessor = VideoPreProcessor(
            video_manifest=self.manifest, env=self.env
        )
        self.analyzer = VideoAnalyzer(
            video_manifest=self.manifest, env=self.env)

    def preprocess_video(
        self,
        output_directory: str = None,
        segment_length: int = 10,
        fps: float = 0.33,
        generate_transcripts_flag: bool = True,
        max_workers: int = None,
        trim_to_nearest_second=False,
        allow_partial_segments=True,
        overwrite_output=False,
    ):
        video_manifest_path = self.preprocessor.preprocess_video(
            output_directory=output_directory,
            segment_length=segment_length,
            fps=fps,
            generate_transcripts_flag=generate_transcripts_flag,
            max_workers=max_workers,
            trim_to_nearest_second=trim_to_nearest_second,
            allow_partial_segments=allow_partial_segments,
            overwrite_output=overwrite_output,
        )
        write_video_manifest(self.manifest)
        return video_manifest_path

    def analyze_video(
        self,
        analysis_config: Type[AnalysisConfig],
        run_async=False,
        max_concurrent_tasks=None,
        reprocess_segments=False,
        person_group_id=None,
        peoples_list_path=None,
        emotions_list_path=None,
        objects_list_path=None,
        themes_list_path=None,
        actions_list_path=None,
    ):
        # If any list paths are provided, need to create a new analyzer instance with them
        if (peoples_list_path or emotions_list_path or objects_list_path or 
            themes_list_path or actions_list_path):
            self.analyzer = VideoAnalyzer(
                video_manifest=self.manifest,
                env=self.env,
                person_group_id=person_group_id,
                peoples_list_path=peoples_list_path,
                emotions_list_path=emotions_list_path,
                objects_list_path=objects_list_path,
                themes_list_path=themes_list_path,
                actions_list_path=actions_list_path,
            )
        elif person_group_id:
            # If only person_group_id is provided, update the analyzer
            self.analyzer = VideoAnalyzer(
                video_manifest=self.manifest,
                env=self.env,
                person_group_id=person_group_id,
            )

        # Store the results but keep the manifest reference intact
        results = self.analyzer.analyze_video(
            analysis_config,
            run_async=run_async,
            max_concurrent_tasks=max_concurrent_tasks,
            reprocess_segments=reprocess_segments,
            person_group_id=person_group_id,
        )
        
        # Return the manifest object (not the results list)
        return self.manifest

    def _prepare_video_manifest(self, video_path: str, **kwargs) -> VideoManifest:
        manifest = VideoManifest()

        # Check that the video file exists
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"File not found: {video_path}")
        else:
            manifest.name = os.path.basename(video_path)
            manifest.source_video.path = os.path.abspath(video_path)

        # Get video metadata
        file_metadata = get_file_info(video_path)
        if file_metadata is not None:
            manifest_source = {
                "path": video_path,
                "video_found": False,
                "size": [],
                "rotation": 0,
                "fps": 0,
                "duration": file_metadata.get("duration", 0),
                "nframes": 0,
                "audio_found": False,
                "audio_duration": 0,
                "audio_fps": 0,
            }

            if "video_info" in file_metadata:
                video_info = file_metadata["video_info"]
                manifest_source["video_found"] = True
                manifest_source["size"] = [
                    int(video_info.get("width", 0)), 
                    int(video_info.get("height", 0))
                ]
                
                # Handle FPS calculation
                fps_str = video_info.get("r_frame_rate", video_info.get("avg_frame_rate", "0/1"))
                try:
                    if "/" in fps_str:
                        num, den = map(float, fps_str.split("/"))
                        fps = num / den if den != 0 else 0
                    else:
                        fps = float(fps_str)
                    manifest_source["fps"] = fps
                except (ValueError, ZeroDivisionError):
                    manifest_source["fps"] = 0
                
                # Get duration from video stream if not in format
                if not manifest_source["duration"] and "duration" in video_info:
                    manifest_source["duration"] = float(video_info["duration"])
                
                # Calculate or get number of frames
                if "nb_frames" in video_info:
                    manifest_source["nframes"] = int(video_info["nb_frames"])
                elif manifest_source["duration"] and manifest_source["fps"]:
                    manifest_source["nframes"] = int(manifest_source["duration"] * manifest_source["fps"])

                # Handle rotation
                if "side_data_list" in video_info:
                    for side_data in video_info["side_data_list"]:
                        if "rotation" in side_data:
                            manifest_source["rotation"] = int(side_data["rotation"])
                            break

            if "audio_info" in file_metadata:
                audio_info = file_metadata["audio_info"]
                manifest_source["audio_found"] = True
                if "duration" in audio_info:
                    manifest_source["audio_duration"] = float(audio_info["duration"])
                
                # Handle audio FPS calculation
                fps_str = audio_info.get("r_frame_rate", audio_info.get("avg_frame_rate", "0/1"))
                try:
                    if "/" in fps_str:
                        num, den = map(float, fps_str.split("/"))
                        fps = num / den if den != 0 else 0
                    else:
                        fps = float(fps_str)
                    manifest_source["audio_fps"] = fps
                except (ValueError, ZeroDivisionError):
                    manifest_source["audio_fps"] = 0

            manifest.source_video = manifest.source_video.model_copy(
                update=manifest_source
            )

        return manifest
