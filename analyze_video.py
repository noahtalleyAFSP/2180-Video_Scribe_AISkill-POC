from cobrapy.video_preprocessor import VideoPreProcessor
from cobrapy.video_analyzer import VideoAnalyzer
import logging
import os
import sys
import argparse
import json
import datetime
from dotenv import load_dotenv
from cobrapy.models.video import VideoManifest
from cobrapy.models.environment import CobraEnvironment, GPTVision, AzureSpeech, AzureFace, BlobStorageConfig
from cobrapy.analysis import ActionSummary
from cobrapy.cobra_utils import get_file_info
from pydantic import SecretStr

# Set up logging
logging.basicConfig(level=logging.INFO)

def analyze_video(
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
    run_async=False,
    env_file=None,
    downscale_to_max_width=None,
    downscale_to_max_height=None,
    use_scene_detection=False,
    scene_detection_threshold=30.0,
):
    """Analyze a video using frame descriptions and OpenAI.
    
    This updated version first runs the preprocessor to generate segment folders
    and extract frames before passing the manifest to the analyzer.
    """
    try:
        # Normalize paths
        video_path = os.path.abspath(video_path)
        
        # Create default output directory if not provided
        if not output_dir:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(os.path.dirname(video_path), f"{video_name}_analysis")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load environment variables from .env file if provided
        if env_file:
            if os.path.exists(env_file):
                load_dotenv(env_file, override=True)
                print(f"Loaded environment variables from {env_file}")
            else:
                print(f"Warning: Environment file {env_file} not found")
        else:
            if os.path.exists(".env"):
                load_dotenv(override=True)
                print("Loaded environment variables from .env")
        
        # Log configuration
        print(f"Analyzing video: {video_path}")
        print(f"Output directory: {output_dir}")
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
        
        # Get API key from environment if not provided
        if not api_key:
            api_key = os.environ.get("AZURE_OPENAI_GPT_VISION_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("Warning: OpenAI API key not provided and not found in environment variables.")
        
        # Get API base, version, and deployment name from environment if not provided
        if not api_base:
            api_base = os.environ.get("AZURE_OPENAI_GPT_VISION_ENDPOINT") or os.environ.get("OPENAI_API_BASE")
        if not api_version:
            api_version = os.environ.get("AZURE_OPENAI_GPT_VISION_API_VERSION") or os.environ.get("OPENAI_API_VERSION")
        if not deployment_name:
            deployment_name = os.environ.get("AZURE_OPENAI_GPT_VISION_DEPLOYMENT")
        
        # Create an instance of ActionSummary with required attributes
        action_summary_instance = ActionSummary()
        if not hasattr(action_summary_instance, 'name'):
            action_summary_instance.name = "ActionSummary"
        if not hasattr(action_summary_instance, 'analysis_sequence'):
            action_summary_instance.analysis_sequence = "mapreduce"
        
        # Create environment configuration using pydantic models
        vision_config = GPTVision(
            endpoint=api_base,
            api_key=SecretStr(api_key) if isinstance(api_key, str) else api_key,
            api_version=api_version,
            deployment=deployment_name
        )
        
        # Create CobraEnvironment - let it load Speech/Face/Blob configs automatically from env
        env = CobraEnvironment(
            vision=vision_config,
        )
        
        # --- Validation/Check after loading ---
        # Check if essential configs were loaded successfully (especially for Batch Transcription)
        if not env.speech.key or not env.speech.region:
             print("ERROR: Azure Speech Key or Region not loaded correctly from environment variables.")
             # Optionally raise an error or exit
             # raise ValueError("Missing critical Azure Speech configuration.")
             return None # Or sys.exit(1)

        if not env.blob_storage.account_name or not env.blob_storage.container_name or \
           (not env.blob_storage.connection_string and not env.blob_storage.sas_token):
            print("WARNING: Azure Blob Storage configuration is incomplete. Batch transcription might fail.")
            # Continue for now, but transcription will likely fail later if needed.

        # Read copyright file content if path is provided
        copyright_json_str = None
        if copyright_file_path:
            if os.path.exists(copyright_file_path):
                try:
                    with open(copyright_file_path, 'r', encoding='utf-8') as f:
                        # Read the entire file content as a string
                        copyright_json_str = f.read()
                    print(f"Successfully read copyright content from {copyright_file_path}")
                except Exception as e:
                    print(f"Warning: Failed to read copyright file {copyright_file_path}: {e}")
            else:
                print(f"Warning: Copyright file not found at {copyright_file_path}")
        
        # Create a video manifest with basic information
        manifest = VideoManifest()
        manifest.name = os.path.basename(video_path)
        manifest.source_video.path = video_path
        manifest.processing_params.fps = fps
        manifest.processing_params.segment_length = segment_duration
        manifest.processing_params.use_speech_based_segments = use_speech_based_segments
        manifest.processing_params.output_directory = output_dir
        
        # Get video metadata and ensure duration is set
        file_metadata = get_file_info(video_path)
        if file_metadata is None:
            raise ValueError(f"Could not get video metadata for {video_path}")
        
        # Set video metadata including audio status
        manifest.source_video.audio_found = file_metadata.get("audio_found", False)
        if "duration" in file_metadata:
            manifest.source_video.duration = float(file_metadata["duration"])
        elif "video_info" in file_metadata and "duration" in file_metadata["video_info"]:
            manifest.source_video.duration = float(file_metadata["video_info"]["duration"])
        else:
            raise ValueError(f"Could not determine video duration for {video_path}")
            
        print(f"Video duration: {manifest.source_video.duration} seconds")
        print(f"Audio detected: {'Yes' if manifest.source_video.audio_found else 'No'}")
        
        # If transcription path is provided, load it
        if transcription_path and os.path.exists(transcription_path):
            with open(transcription_path, 'r', encoding='utf-8') as f:
                manifest.audio_transcription = json.load(f)
                print(f"Loaded transcription from {transcription_path}")
        
        # Run preprocessing with explicit audio handling
        preprocessor = VideoPreProcessor(manifest, env)
        print("Starting preprocessing...")
        preprocessor.preprocess_video(
            output_directory=output_dir,
            segment_length=segment_duration,
            fps=fps,
            use_scene_detection=use_scene_detection,
            scene_detection_threshold=scene_detection_threshold,
            use_speech_based_segments=use_speech_based_segments,
            generate_transcripts_flag=manifest.source_video.audio_found,  # Only try to generate transcripts if audio is found
            overwrite_output=True,
            downscale_to_max_width=downscale_to_max_width,
            downscale_to_max_height=downscale_to_max_height,
        )
        print("Preprocessing completed. Segments generated:", len(manifest.segments))
        
        # Initialize the analyzer using the older interface that accepts a manifest and environment
        analyzer = VideoAnalyzer(
            video_manifest=manifest,
            env=env,
            person_group_id=None,
            peoples_list_path=peoples_list_path,
            emotions_list_path=emotions_list_path,
            objects_list_path=objects_list_path,
            themes_list_path=themes_list_path,
            actions_list_path=actions_list_path
        )
        
        # Log API configuration
        print(f"API Configuration:")
        print(f"  Endpoint: {env.vision.endpoint}")
        print(f"  API Version: {env.vision.api_version}")
        print(f"  Deployment: {env.vision.deployment}")
        
        # Start analysis
        start_time_dt = datetime.datetime.now()
        print(f"Starting analysis at {start_time_dt.strftime('%H:%M:%S')}...")
        
        manifest = analyzer.analyze_video(
            analysis_config=action_summary_instance,
            run_async=run_async,
            copyright_json_str=copyright_json_str,
            use_scene_detection=use_scene_detection,
            scene_detection_threshold=scene_detection_threshold
        )
        
        end_time_dt = datetime.datetime.now()
        duration = end_time_dt - start_time_dt
        print(f"Analysis completed at {end_time_dt.strftime('%H:%M:%S')}")
        print(f"Total analysis time: {duration}")
        
        if manifest:
            print(f"Analysis results saved to {output_dir}")
            # Print token usage summary if available
            if hasattr(analyzer, "token_usage") and isinstance(analyzer.token_usage, dict):
                print("\nToken usage summary:")
                for k, v in analyzer.token_usage.items():
                    print(f"  {k}: {v}")
            return manifest
        else:
            print("Analysis failed to produce a valid manifest.")
            return None
            
    except Exception as e:
        import traceback
        print(f"Error during video analysis: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a video using frame descriptions and OpenAI")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--output-dir", help="Directory to save analysis results")
    parser.add_argument("--api-key", help="OpenAI API key (default: from environment variables)")
    parser.add_argument("--api-base", help="OpenAI API base URL (for Azure OpenAI)")
    parser.add_argument("--api-version", help="OpenAI API version (for Azure OpenAI)")
    parser.add_argument("--deployment-name", help="Azure OpenAI deployment name (for Azure OpenAI)")
    parser.add_argument("--storage-account-name", help="Azure Storage Account Name (or AZURE_STORAGE_ACCOUNT_NAME)")
    parser.add_argument("--storage-container-name", help="Azure Storage Container Name (or AZURE_STORAGE_CONTAINER_NAME)")
    parser.add_argument("--storage-connection-string", help="Azure Storage Connection String (or AZURE_STORAGE_CONNECTION_STRING)")
    parser.add_argument("--storage-sas-token", help="Azure Storage SAS Token (or AZURE_STORAGE_SAS_TOKEN)")
    parser.add_argument("--peoples-list", help="Path to a JSON file containing people to identify in the video")
    parser.add_argument("--emotions-list", help="Path to a JSON file containing emotions to detect in the video")
    parser.add_argument("--objects-list", help="Path to a JSON file containing objects to detect in the video")
    parser.add_argument("--themes-list", help="Path to a JSON file containing themes to classify in the video")
    parser.add_argument("--actions-list", help="Path to a JSON file containing actions to detect in the video")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract (default: 1.0)")
    parser.add_argument("--segment-duration", type=float, default=30.0, help="Length of video segments in seconds (default: 30.0). Ignored if --use-scene-detection is set.")
    parser.add_argument("--transcription-path", help="Path to a JSON file containing speech transcription")
    parser.add_argument("--use-scene-detection", action="store_true", help="Use PySceneDetect to determine segments instead of fixed intervals.")
    parser.add_argument("--scene-detection-threshold", type=float, default=30.0, help="PySceneDetect ContentDetector threshold (default: 30.0). Higher value means fewer scenes.")
    parser.add_argument("--use-speech-based-segments", action="store_true", help="Use speech-based segmentation instead of fixed intervals (priority: scene > speech > time)")
    parser.add_argument("--async", dest="run_async", action="store_true", help="Run analysis asynchronously")
    parser.add_argument("--env-file", help="Path to .env file containing API credentials")
    parser.add_argument("--copyright-file", help="Path to a JSON file containing copyright information")
    parser.add_argument("--downscale-to-max-width", type=int, help="Maximum width for downscaled frames (maintain aspect ratio)")
    parser.add_argument("--downscale-to-max-height", type=int, help="Maximum height for downscaled frames (maintain aspect ratio)")
    args = parser.parse_args()
    
    analyze_video(
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
        run_async=args.run_async,
        env_file=args.env_file,
        downscale_to_max_width=args.downscale_to_max_width,
        downscale_to_max_height=args.downscale_to_max_height,
        use_scene_detection=args.use_scene_detection,
        scene_detection_threshold=args.scene_detection_threshold,
    )