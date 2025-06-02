from cobrapy.video_preprocessor import VideoPreProcessor
from cobrapy.video_analyzer import VideoAnalyzer
import logging
import os
import sys
import argparse
import json
import datetime
import asyncio # Added for async client
from dotenv import load_dotenv
from cobrapy.models.video import VideoManifest
from cobrapy.models.environment import CobraEnvironment, GPTVision, AzureSpeech, AzureFace, BlobStorageConfig
from cobrapy.analysis import ActionSummary
from cobrapy.cobra_utils import get_file_info
from pydantic import SecretStr
from pathlib import Path
from cobrapy.pipeline.step0_yolo_track import run_yolo
from cobrapy.track_refiner import refine_yolo_tracks_across_scenes # ADDED IMPORT
from azure.identity import DefaultAzureCredential, get_bearer_token_provider # For Async Client
from openai import AsyncAzureOpenAI, AsyncOpenAI # MODIFIED: Added AsyncOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)

# ADD THESE LINES to quiet down Azure SDK http logging
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
    run_async_analyzer=False, # Renamed from run_async to avoid conflict
    env_file=None,
    downscale_to_max_width=None,
    downscale_to_max_height=None,
    use_scene_detection=False,
    scene_detection_threshold=30.0,
    max_concurrent_tasks=8,
    enable_language_identification: bool = False,
    skip_refinement: bool = False # ADDED: Flag to skip refinement for debugging
):
    """Analyze a video using frame descriptions and OpenAI.
    This is an async wrapper to handle async client setup for track refinement.
    """
    async_llm_client_main_script = None # MODIFIED: Renamed for clarity
    analyzer_instance = None # ADDED: To manage VideoAnalyzer's client and its internal client

    try:
        # Normalize paths
        video_path_abs = os.path.abspath(video_path)
        video_path_obj = Path(video_path_abs)

        # Create default output directory if not provided
        if not output_dir:
            video_name_no_ext = video_path_obj.stem
            output_dir_path = video_path_obj.parent / f"{video_name_no_ext}_analysis"
        else:
            output_dir_path = Path(output_dir)
        
        os.makedirs(output_dir_path, exist_ok=True)
        output_dir_abs = str(output_dir_path.resolve())

        # Load environment variables
        if env_file:
            if os.path.exists(env_file):
                load_dotenv(env_file, override=True)
                print(f"Loaded environment variables from {env_file}")
            else:
                print(f"Warning: Environment file {env_file} not found")
        elif os.path.exists(".env"):
            load_dotenv(override=True)
            print("Loaded environment variables from .env")

        # --- Logging configuration ---
        print(f"Analyzing video: {video_path_abs}")
        print(f"Output directory: {output_dir_abs}")
        print(f"Using {fps} frames per second")
        print(f"Segment duration: {segment_duration} seconds")
        
        if peoples_list_path:
            print(f"Using peoples list: {peoples_list_path}")
        if emotions_list_path:
            print(f"Using emotions list: {emotions_list_path}")
        if objects_list_path:
            print(f"Using objects list: {objects_list_path}")
        if themes_list_path:
            print(f"Using themes list: {themes_list_path}")
        if actions_list_path:
            print(f"Using actions list: {actions_list_path}")
        
        if copyright_file_path:
            print(f"Using copyright file: {copyright_file_path}")
        
        # --- Environment and API Config --- 
        # API key from args or env
        effective_api_key = api_key or os.environ.get("AZURE_OPENAI_GPT_VISION_API_KEY") or os.environ.get("OPENAI_API_KEY")
        # API base, version, deployment from args or env
        effective_api_base = api_base or os.environ.get("AZURE_OPENAI_GPT_VISION_ENDPOINT") or os.environ.get("OPENAI_API_BASE")
        effective_api_version = api_version or os.environ.get("AZURE_OPENAI_GPT_VISION_API_VERSION") or os.environ.get("OPENAI_API_VERSION")
        effective_deployment_name = deployment_name or os.environ.get("AZURE_OPENAI_GPT_VISION_DEPLOYMENT")

        if not effective_api_key or not effective_api_base:
            print("CRITICAL ERROR: API Key or API Base not configured. Exiting.")
            return None

        vision_config = GPTVision(
            endpoint=effective_api_base,
            api_key=SecretStr(effective_api_key) if isinstance(effective_api_key, str) else effective_api_key,
            api_version=effective_api_version,
            deployment=effective_deployment_name
        )
        env = CobraEnvironment(vision=vision_config) # Speech, Face, Blob will load from env

        # --- Create Async OpenAI Client (needed for track refiner) --- 
        # This client is for the refine_yolo_tracks_across_scenes call in this script
        if env.vision.endpoint and (
            ".cognitiveservices.azure.com" in env.vision.endpoint
            or "openai.azure.com" in env.vision.endpoint
        ):
            async_llm_client_main_script = AsyncAzureOpenAI( # MODIFIED: Use new variable name
                api_key=env.vision.api_key.get_secret_value(),
                api_version=env.vision.api_version or "2024-02-01",
                azure_endpoint=env.vision.endpoint,
            )
            print("Initialized **AsyncAzureOpenAI** client for Azure (main script for track refiner).")
        elif env.vision.api_key: # Fallback for non-Azure or standard OpenAI
            async_llm_client_main_script = AsyncOpenAI( # MODIFIED: Use new variable name
                api_key=env.vision.api_key.get_secret_value(),
                # Use endpoint as base_url if provided, otherwise default to public OpenAI
                base_url=env.vision.endpoint if "api.openai.com" not in (env.vision.endpoint or "") else (env.vision.endpoint or "https://api.openai.com/v1"),
            )
            if env.vision.endpoint and "api.openai.com" not in env.vision.endpoint:
                print(f"Initialized **AsyncOpenAI** client (custom non-Azure endpoint for main script: {env.vision.endpoint}).")
            else:
                print("Initialized **AsyncOpenAI** client (public OpenAI endpoint or default for main script).")
        
        if not async_llm_client_main_script and not skip_refinement: # MODIFIED: Use new variable name
             print("[CRITICAL ERROR] Async LLM client for track refinement (main script) could not be initialized. To skip refinement, use --skip-refinement. Exiting.")
             return None

        # --- Copyright Info --- 
        copyright_json_str = None
        if copyright_file_path and os.path.exists(copyright_file_path):
            try:
                with open(copyright_file_path, 'r', encoding='utf-8') as f: copyright_json_str = f.read()
                print(f"Successfully read copyright content from {copyright_file_path}")
            except Exception as e:
                print(f"Warning: Failed to read copyright file {copyright_file_path}: {e}")

        # --- Manifest Initialization & Metadata --- 
        manifest = VideoManifest()
        manifest.name = video_path_obj.name
        manifest.source_video.path = video_path_abs
        manifest.processing_params.fps = fps
        manifest.processing_params.segment_length = segment_duration
        manifest.processing_params.use_speech_based_segments = use_speech_based_segments
        manifest.processing_params.output_directory = output_dir_abs

        file_metadata = get_file_info(video_path_abs)
        if file_metadata is None: raise ValueError(f"Could not get video metadata for {video_path_abs}")
        manifest.source_video.audio_found = file_metadata.get("audio_found", False)
        duration_val = file_metadata.get("duration") or file_metadata.get("video_info", {}).get("duration")
        if not duration_val: raise ValueError(f"Could not determine video duration for {video_path_abs}")
        manifest.source_video.duration = float(duration_val)
        print(f"Video duration: {manifest.source_video.duration} seconds. Audio detected: {manifest.source_video.audio_found}")

        if transcription_path and os.path.exists(transcription_path):
            with open(transcription_path, 'r', encoding='utf-8') as f: manifest.audio_transcription = json.load(f)
            print(f"Loaded transcription from {transcription_path}")

        # --- Step 1: Preprocessing (Scene Detection, Frame Extraction, Transcripts) --- 
        preprocessor = VideoPreProcessor(manifest, env)
        print("Starting preprocessing...")
        preprocessor.preprocess_video(
            output_directory=output_dir_abs,
            segment_length=segment_duration,
            fps=fps,
            use_scene_detection=use_scene_detection,
            scene_detection_threshold=scene_detection_threshold,
            use_speech_based_segments=use_speech_based_segments,
            generate_transcripts_flag=manifest.source_video.audio_found,
            overwrite_output=True, # Consider making this configurable
            downscale_to_max_width=downscale_to_max_width,
            downscale_to_max_height=downscale_to_max_height,
            enable_language_identification=enable_language_identification
        )
        print(f"Preprocessing completed. Manifest has {len(manifest.segments)} scene segments.")

        # --- Step 2: Raw YOLO Object Tracking --- 
        print(f"Running raw YOLO object tracking for {video_path_abs}...")
        # run_yolo expects output_dir as Path object
        raw_yolo_tags = run_yolo(video_path=video_path_obj, out_dir=output_dir_path)
        manifest.raw_yolo_tags = raw_yolo_tags # Store raw tags
        print(f"Raw YOLO tracking completed. Found {len(raw_yolo_tags)} raw tracks.")
        # Save raw_yolo_tags to a file in output_dir for inspection
        raw_tags_file = output_dir_path / f"{video_path_obj.stem}_manifest_raw_yolo_tags.json"
        with open(raw_tags_file, "w") as f:
            json.dump(raw_yolo_tags, f, indent=2)
        print(f"Saved raw YOLO tags to {raw_tags_file}")

        # --- Step 3: Refine YOLO Tracks Across Scenes (New) ---
        if not skip_refinement:
            if manifest.segments and manifest.raw_yolo_tags and async_llm_client_main_script: # MODIFIED: Use new variable name
                print("Starting YOLO track refinement across scenes...")
                # refine_yolo_tracks_across_scenes now returns a tuple
                refined_tags_list, reid_prompt_tokens, reid_completion_tokens, reid_image_tokens = await refine_yolo_tracks_across_scenes(
                    manifest_segments=manifest.segments, 
                    manifest_raw_yolo_tags=manifest.raw_yolo_tags,
                    env=env,
                    video_path=video_path_abs,
                    output_dir=output_dir_abs, 
                    async_llm_client=async_llm_client_main_script # MODIFIED: Pass the correctly named client
                )
                manifest.refined_yolo_tags = refined_tags_list
                manifest.initial_prompt_tokens = (manifest.initial_prompt_tokens or 0) + reid_prompt_tokens
                manifest.initial_completion_tokens = (manifest.initial_completion_tokens or 0) + reid_completion_tokens
                manifest.initial_image_tokens = (manifest.initial_image_tokens or 0) + reid_image_tokens
                
                print(f"YOLO track refinement completed. Produced {len(manifest.refined_yolo_tags)} refined tracks.")
                print(f"Re-ID LLM Tokens: P={reid_prompt_tokens}, C={reid_completion_tokens}, Img={reid_image_tokens} (added to manifest initial tokens)")
                # Save refined_yolo_tags to a file
                refined_tags_file = output_dir_path / f"{video_path_obj.stem}_manifest_refined_yolo_tags.json"
                with open(refined_tags_file, "w") as f:
                    json.dump(manifest.refined_yolo_tags, f, indent=2)
                print(f"Saved refined YOLO tags to {refined_tags_file}")
            elif not async_llm_client_main_script: # MODIFIED: Use new variable name
                 print("[WARNING] Skipping track refinement as async LLM client (main script) failed to initialize.")
            else:
                print("[INFO] Skipping track refinement: no scenes, no raw tags, or client issue.")
        else:
            print("[INFO] Skipping track refinement as per --skip-refinement flag.")

        # --- Step 4: Main Video Analysis (using VideoAnalyzer) --- 
        analyzer_instance = VideoAnalyzer( # MODIFIED: Store instance
            video_manifest=manifest, # Manifest now potentially contains raw_yolo_tags and refined_yolo_tags
            env=env,
            # person_group_id can be passed if needed for other face tasks
            peoples_list_path=peoples_list_path,
            emotions_list_path=emotions_list_path,
            objects_list_path=objects_list_path,
            themes_list_path=themes_list_path,
            actions_list_path=actions_list_path
        )
        
        print(f"API Configuration for Analyzer:")
        print(f"  Endpoint: {env.vision.endpoint}")
        print(f"  API Version: {env.vision.api_version}")
        print(f"  Deployment: {env.vision.deployment}")
        
        start_time_dt = datetime.datetime.now()
        print(f"Starting main analysis with VideoAnalyzer at {start_time_dt.strftime('%H:%M:%S')}...")
        
        # The analyze_video method of VideoAnalyzer will need to be aware of refined_yolo_tags
        # and use them as self.active_yolo_tags_source if available.
        action_summary_instance = ActionSummary() # Default analysis config
        if not hasattr(action_summary_instance, 'name'): action_summary_instance.name = "ActionSummary"
        if not hasattr(action_summary_instance, 'analysis_sequence'): action_summary_instance.analysis_sequence = "mapreduce"

        # VideoAnalyzer.analyze_video is currently synchronous. 
        # If it needs to be async (e.g. because it uses the async_llm_client), it would need to be awaited.
        # For now, assuming it's synchronous as per original structure.
        final_manifest = await analyzer_instance.analyze_video( # MODIFIED: Use instance
            analysis_config=action_summary_instance,
            run_async=run_async_analyzer, # This is for segment processing within VideoAnalyzer
            max_concurrent_tasks=max_concurrent_tasks,
            copyright_json_str=copyright_json_str,
            # use_scene_detection and scene_detection_threshold are primarily for preprocessor
            # but can be passed if analyzer uses them for other logic.
        )
        
        end_time_dt = datetime.datetime.now()
        duration = end_time_dt - start_time_dt
        print(f"Main analysis completed at {end_time_dt.strftime('%H:%M:%S')}. Total analysis time: {duration}")
        
        if final_manifest:
            print(f"Full analysis process complete. Final manifest results in {output_dir_abs}")
            # Token usage summary from analyzer:
            if hasattr(analyzer_instance, "token_usage") and isinstance(analyzer_instance.token_usage, dict):
                print("\nToken usage summary (from VideoAnalyzer GPT calls):")
                for k, v in analyzer_instance.token_usage.items(): print(f"  {k}: {v}")
            # Note: Token usage for Re-ID calls in track_refiner is not aggregated here yet.
            return final_manifest
        else:
            print("Main analysis failed to produce a valid final manifest.")
            return None
            
    except Exception as e:
        import traceback
        print(f"Error during video analysis pipeline: {str(e)}")
        traceback.print_exc()
        return None
    finally:
        if async_llm_client_main_script: # MODIFIED: Use new variable name and ensure it's closed
            await async_llm_client_main_script.close()
            print("Closed Async LLM client (main script for track refiner).")
        if analyzer_instance: # ADDED: Close VideoAnalyzer's internal client
            await analyzer_instance.close_internal_async_client()
            print("Closed VideoAnalyzer's internal async client (if any).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a video using CobraPy pipeline")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--output-dir", help="Directory to save analysis results")
    # API related args
    parser.add_argument("--api-key", help="Azure OpenAI API key")
    parser.add_argument("--api-base", help="Azure OpenAI API base URL")
    parser.add_argument("--api-version", help="Azure OpenAI API version")
    parser.add_argument("--deployment-name", help="Azure OpenAI deployment name for GPT-Vision")
    # Storage (Optional, for features like batch transcription if used by preprocessor)
    parser.add_argument("--storage-account-name", help="Azure Storage Account Name")
    parser.add_argument("--storage-container-name", help="Azure Storage Container Name")
    parser.add_argument("--storage-connection-string", help="Azure Storage Connection String")
    parser.add_argument("--storage-sas-token", help="Azure Storage SAS Token")
    # Custom lists paths
    parser.add_argument("--peoples-list", help="Path to a JSON file for people list")
    parser.add_argument("--emotions-list", help="Path to a JSON file for emotions list")
    parser.add_argument("--objects-list", help="Path to a JSON file for objects list")
    parser.add_argument("--themes-list", help="Path to a JSON file for themes list")
    parser.add_argument("--actions-list", help="Path to a JSON file for actions list")
    parser.add_argument("--copyright-file", help="Path to JSON file with copyright info")
    # Processing parameters
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

    args = parser.parse_args()
    
    # Run the async wrapper
    asyncio.run(analyze_video_async_wrapper(
        args.video_path,
        output_dir=args.output_dir,
        api_key=args.api_key,
        api_base=args.api_base,
        api_version=args.api_version,
        deployment_name=args.deployment_name,
        storage_account_name=args.storage_account_name,
        storage_container_name=args.storage_container_name,
        storage_connection_string=args.storage_connection_string,
        storage_sas_token=args.storage_sas_token,
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
        enable_language_identification=args.enable_language_id,
        skip_refinement=args.skip_refinement
    ))