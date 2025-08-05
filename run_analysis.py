from cobrapy.video_preprocessor import VideoPreProcessor
from cobrapy.video_analyzer import VideoAnalyzer
import logging
import os
import sys
import argparse
import json
import datetime
import asyncio
from dotenv import load_dotenv
from cobrapy.models.video import VideoManifest
from cobrapy.models.environment import CobraEnvironment, GPTVision, AzureSpeech, BlobStorageConfig
from cobrapy.analysis import ActionSummary
from cobrapy.cobra_utils import get_file_info, write_video_manifest
from pydantic import SecretStr
from pathlib import Path
from cobrapy.pipeline.step0_yolo_track import run_yolo
from cobrapy.pipeline.step0b_reid_split_tracks import refine_tracks_over_cuts
from cobrapy.track_refiner import refine_yolo_tracks_across_scenes
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI, AsyncOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)

# Quieten down Azure SDK http logging
azure_loggers = [
    "azure.core.pipeline.policies.http_logging_policy",
    "azure.identity",
    "azure.storage.blob",
    "msal"
]
for logger_name in azure_loggers:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

async def analyze_video_async_wrapper(
    video_path,
    output_dir=None,
    api_key=None,
    api_base=None,
    api_version=None,
    deployment_name=None,
    storage_account_name=None,
    storage_container_name=None,
    storage_connection_string=None,
    storage_sas_token=None,
    peoples_list_path=None,
    emotions_list_path=None,
    objects_list_path=None,
    themes_list_path=None,
    actions_list_path=None,
    copyright_file_path=None,
    fps=1.0,
    segment_duration=30.0,
    transcription_path=None,
    use_speech_based_segments=False,
    run_async_analyzer=False,
    env_file=None,
    downscale_to_max_width=None,
    downscale_to_max_height=None,
    use_scene_detection=False,
    scene_detection_threshold=30.0,
    max_concurrent_tasks=8,
    enable_language_identification: bool = False,
    skip_refinement: bool = False,
    skip_yolo: bool = False,
    skip_thumbnail_saving: bool = False,
    overwrite_output: bool = False
):
    """Analyze a video using frame descriptions and OpenAI."""
    async_llm_client_main_script = None
    analyzer_instance = None

    try:
        video_path_abs = os.path.abspath(video_path)
        video_path_obj = Path(video_path_abs)

        if output_dir:
            output_dir_path = Path(output_dir)
            output_dir_abs = str(output_dir_path.resolve())
        else:
            output_dir_abs = None

        if env_file and os.path.exists(env_file):
            load_dotenv(env_file, override=True)
            print(f"Loaded environment variables from {env_file}")
        elif os.path.exists(".env"):
            load_dotenv(override=True)
            print("Loaded environment variables from .env")

        print(f"Analyzing video: {video_path_abs}")
        print(f"Output directory: {output_dir_abs or '[Will be set by preprocessor]'}")
        
        effective_api_key = api_key or os.environ.get("AZURE_OPENAI_GPT_VISION_API_KEY") or os.environ.get("OPENAI_API_KEY")
        effective_api_base = api_base or os.environ.get("AZURE_OPENAI_GPT_VISION_ENDPOINT") or os.environ.get("OPENAI_API_BASE")
        effective_api_version = api_version or os.environ.get("AZURE_OPENAI_GPT_VISION_API_VERSION") or os.environ.get("OPENAI_API_VERSION")
        effective_deployment_name = deployment_name or os.environ.get("AZURE_OPENAI_GPT_VISION_DEPLOYMENT")

        if not effective_api_key or not effective_api_base:
            print("CRITICAL ERROR: API Key or API Base not configured. Exiting.")
            return None

        vision_config = GPTVision(
            endpoint=effective_api_base,
            api_key=SecretStr(effective_api_key),
            api_version=effective_api_version,
            deployment=effective_deployment_name
        )
        env = CobraEnvironment(vision=vision_config)

        if ".cognitiveservices.azure.com" in env.vision.endpoint or "openai.azure.com" in env.vision.endpoint:
            async_llm_client_main_script = AsyncAzureOpenAI(
                api_key=env.vision.api_key.get_secret_value(),
                api_version=env.vision.api_version or "2024-02-01",
                azure_endpoint=env.vision.endpoint,
            )
        elif env.vision.api_key:
            async_llm_client_main_script = AsyncOpenAI(
                api_key=env.vision.api_key.get_secret_value(),
                base_url=env.vision.endpoint,
            )
        
        if not async_llm_client_main_script and not skip_refinement:
             print("[CRITICAL ERROR] Async LLM client for track refinement could not be initialized. Exiting.")
             return None

        copyright_json_str = None
        if copyright_file_path and os.path.exists(copyright_file_path):
            with open(copyright_file_path, 'r', encoding='utf-8') as f:
                copyright_json_str = f.read()

        manifest = VideoManifest(name=video_path_obj.name)
        manifest.source_video.path = video_path_abs
        manifest.processing_params.fps = fps
        manifest.processing_params.segment_length = segment_duration
        manifest.processing_params.use_speech_based_segments = use_speech_based_segments

        file_metadata = get_file_info(video_path_abs)
        if file_metadata is None: raise ValueError("Could not get video metadata")
        manifest.source_video.audio_found = file_metadata.get("audio_found", False)
        duration_val = file_metadata.get("duration")
        if not duration_val: raise ValueError("Could not determine video duration")
        manifest.source_video.duration = float(duration_val)

        preprocessor = VideoPreProcessor(manifest, env)
        print("Starting preprocessing...")
        asset_dir_path = preprocessor.preprocess_video(
            output_directory=output_dir_abs,
            segment_length=segment_duration,
            fps=fps,
            use_scene_detection=use_scene_detection,
            scene_detection_threshold=scene_detection_threshold,
            use_speech_based_segments=use_speech_based_segments,
            generate_transcripts_flag=manifest.source_video.audio_found,
            overwrite_output=overwrite_output,
            downscale_to_max_width=downscale_to_max_width,
            downscale_to_max_height=downscale_to_max_height,
            enable_language_identification=enable_language_identification
        )
        print(f"Preprocessing completed. Manifest has {len(manifest.segments)} segments.")
        manifest.processing_params.output_directory = asset_dir_path
        output_dir_path = Path(asset_dir_path)

        if not skip_yolo:
            print(f"Running raw YOLO object tracking for {video_path_abs}...")
            raw_yolo_tags = run_yolo(
                video_path=video_path_obj, 
                out_dir=output_dir_path,
                skip_thumbnail_saving=skip_thumbnail_saving
            )
            raw_tags_file_path = output_dir_path / f"{video_path_obj.stem}_manifest_raw_yolo_tags.json"
            with open(raw_tags_file_path, "w") as f:
                json.dump(raw_yolo_tags, f, indent=2)
            print(f"Saved raw YOLO tags to {raw_tags_file_path}")
            manifest.raw_yolo_tags = raw_yolo_tags
            
            tracks_for_refinement = raw_yolo_tags
            if not skip_refinement and manifest.segments and raw_yolo_tags and async_llm_client_main_script:
                print("Refining tracks...")
                refined_tags, _, _, _ = await refine_yolo_tracks_across_scenes(
                    manifest_segments=manifest.segments, 
                    manifest_raw_yolo_tags=tracks_for_refinement,
                    env=env,
                    video_path=video_path_abs,
                    output_dir=str(output_dir_path.resolve()), 
                    async_llm_client=async_llm_client_main_script,
                    skip_thumbnail_saving=skip_thumbnail_saving
                )
                manifest.refined_yolo_tags = refined_tags
        else:
            print("Skipping YOLO tracking and refinement steps as requested.")
            manifest.raw_yolo_tags = []
            manifest.yolo_tags = []
            manifest.refined_yolo_tags = []

        analyzer_instance = VideoAnalyzer(
            video_manifest=manifest,
            env=env,
            peoples_list_path=peoples_list_path,
            emotions_list_path=emotions_list_path,
            objects_list_path=objects_list_path,
            themes_list_path=themes_list_path,
            actions_list_path=actions_list_path
        )
        
        action_summary_instance = ActionSummary()
        
        final_manifest = await analyzer_instance.analyze_video(
            analysis_config=action_summary_instance,
            run_async=run_async_analyzer,
            max_concurrent_tasks=max_concurrent_tasks,
            copyright_json_str=copyright_json_str,
        )
        
        if final_manifest:
            print(f"Full analysis process complete. Final results in {output_dir_path.resolve()}")
        else:
            print("Main analysis failed.")
            
    except Exception as e:
        import traceback
        print(f"Error during video analysis pipeline: {str(e)}")
        traceback.print_exc()
    finally:
        if async_llm_client_main_script:
            await async_llm_client_main_script.close()
        if analyzer_instance:
            await analyzer_instance.close_internal_async_client()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a video using CobraPy pipeline")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--output-dir", help="Directory to save analysis results")
    parser.add_argument("--api-key", help="Azure OpenAI API key")
    parser.add_argument("--api-base", help="Azure OpenAI API base URL")
    parser.add_argument("--api-version", help="Azure OpenAI API version")
    parser.add_argument("--deployment-name", help="Azure OpenAI deployment name for GPT-Vision")
    parser.add_argument("--peoples-list", help="Path to a JSON file for people list")
    parser.add_argument("--emotions-list", help="Path to a JSON file for emotions list")
    parser.add_argument("--objects-list", help="Path to a JSON file for objects list")
    parser.add_argument("--themes-list", help="Path to a JSON file for themes list")
    parser.add_argument("--actions-list", help="Path to a JSON file for actions list")
    parser.add_argument("--copyright-file", help="Path to JSON file with copyright info")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second for main analysis (default: 1.0)")
    parser.add_argument("--segment-duration", type=float, default=30.0, help="Segment duration for non-scene-based (default: 30.0s)")
    parser.add_argument("--transcription-path", help="Path to a pre-existing JSON transcription file")
    parser.add_argument("--use-scene-detection", action="store_true", help="Enable scene detection for segmentation")
    parser.add_argument("--scene-detection-threshold", type=float, default=30.0, help="PySceneDetect threshold (default: 30.0)")
    parser.add_argument("--use-speech-based-segments", action="store_true", help="Enable speech-based segmentation")
    parser.add_argument("--run-async-analyzer", action="store_true", help="Run VideoAnalyzer segment processing asynchronously")
    parser.add_argument("--env-file", help="Path to .env file for credentials")
    parser.add_argument("--downscale-to-max-width", type=int, help="Max width for frame downscaling")
    parser.add_argument("--downscale-to-max-height", type=int, help="Max height for frame downscaling")
    parser.add_argument("--enable-language-id", action="store_true", help="Enable Azure Batch Transcription Language ID")
    parser.add_argument("--skip-refinement", action="store_true", help="Skip the new track refinement step")
    parser.add_argument("--skip-yolo", action="store_true", help="Skip all YOLO tracking and refinement steps.")
    parser.add_argument("--skip-thumbnail-saving", action="store_true", help="If set, skips saving of YOLO and Re-ID thumbnails")
    parser.add_argument("--overwrite-output", action="store_true", help="If set, will overwrite the output directory if it exists (dangerous)")

    args = parser.parse_args()
    
    asyncio.run(analyze_video_async_wrapper(
        args.video_path,
        output_dir=args.output_dir,
        api_key=args.api_key,
        api_base=args.api_base,
        api_version=args.api_version,
        deployment_name=args.deployment_name,
        storage_account_name=None,
        storage_container_name=None,
        storage_connection_string=None,
        storage_sas_token=None,
        peoples_list_path=args.peoples_list,
        emotions_list_path=args.emotions_list,
        objects_list_path=args.objects_list,
        themes_list_path=args.themes_list,
        actions_list_path=args.actions_list,
        copyright_file_path=args.copyright_file,
        fps=args.fps,
        segment_duration=args.segment_duration,
        transcription_path=args.transcription_path,
        use_speech_based_segments=args.use_speech_based_segments,
        run_async_analyzer=args.run_async_analyzer,
        env_file=args.env_file,
        downscale_to_max_width=args.downscale_to_max_width,
        downscale_to_max_height=args.downscale_to_max_height,
        use_scene_detection=args.use_scene_detection,
        scene_detection_threshold=args.scene_detection_threshold,
        max_concurrent_tasks=8,
        enable_language_identification=args.enable_language_id,
        skip_refinement=args.skip_refinement,
        skip_yolo=args.skip_yolo,
        skip_thumbnail_saving=args.skip_thumbnail_saving,
        overwrite_output=args.overwrite_output
    )) 