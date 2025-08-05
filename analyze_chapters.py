import argparse
import logging
import os
from dotenv import load_dotenv
from pydantic import SecretStr

from cobrapy.chapter_generator import ChapterGenerator
from cobrapy.models.environment import CobraEnvironment, GPTVision, AzureSpeech, BlobStorageConfig

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Quieten noisy loggers
for logger_name in ["azure.core.pipeline.policies.http_logging_policy", "azure.identity", "azure.storage.blob", "msal"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


def main():
    """
    Main function to run the chapter generation pipeline.
    """
    parser = argparse.ArgumentParser(description="Generate chapters and transcription for a video using a streamlined pipeline.")
    
    # --- Core Arguments ---
    parser.add_argument("video_path", help="Path to the video file to analyze.")
    parser.add_argument("--output-dir", help="Directory to save analysis results. If not set, a timestamped directory will be created.")
    parser.add_argument("--env-file", help="Path to .env file for credentials.")
    parser.add_argument("--overwrite-output", action="store_true", help="If set, will overwrite the output directory if it exists.")

    # --- API Arguments ---
    parser.add_argument("--api-key", help="Azure OpenAI API key. Overrides environment variables.")
    parser.add_argument("--api-base", help="Azure OpenAI API base URL. Overrides environment variables.")
    parser.add_argument("--api-version", help="Azure OpenAI API version. Overrides environment variables.")
    parser.add_argument("--deployment-name", help="Azure OpenAI deployment name for GPT-Vision. Overrides environment variables.")

    # --- Preprocessing Arguments ---
    parser.add_argument("--fps", type=float, default=0.2, help="Frames per second to sample for analysis (default: 0.2). Lower for local models.")
    parser.add_argument("--segment-duration", type=float, default=30.0, help="Segment duration in seconds for fixed-length segmentation (default: 30.0).")
    parser.add_argument("--use-scene-detection", action="store_true", help="Enable scene detection for segmentation instead of fixed-length segments.")
    parser.add_argument("--scene-detection-threshold", type=float, default=30.0, help="Threshold for PySceneDetect ContentDetector (default: 30.0).")
    parser.add_argument("--downscale-to-max-width", type=int, help="Max width for frame downscaling to reduce token usage.")
    parser.add_argument("--downscale-to-max-height", type=int, help="Max height for frame downscaling to reduce token usage.")
    parser.add_argument("--enable-language-id", action="store_true", help="Enable Azure Batch Transcription Language ID.")
    parser.add_argument("--use-local-llm", action="store_true", help="Analyse with on-device Phi-4 instead of Azure GPT-Vision.")
    parser.add_argument("--images-per-segment", type=int, default=4, help="Frames sent to Phi-4 for each segment when offline.")
    parser.add_argument("--whisper-model", type=str, default="base", help="Name of the Whisper model to use for local transcription (e.g., tiny, base, small).")

    args = parser.parse_args()

    # --- Load Environment ---
    if args.env_file:
        if os.path.exists(args.env_file):
            load_dotenv(args.env_file, override=True)
            logging.info(f"Loaded environment variables from {args.env_file}")
        else:
            logging.warning(f"Environment file {args.env_file} not found.")
    elif os.path.exists(".env"):
        load_dotenv(override=True)
        logging.info("Loaded environment variables from .env file.")

    try:
        # --- Configure Environment from args and env vars ---
        env = None
        if not args.use_local_llm:
            # Vision/LLM Config
            effective_api_key = args.api_key or os.environ.get("AZURE_OPENAI_GPT_VISION_API_KEY")
            effective_api_base = args.api_base or os.environ.get("AZURE_OPENAI_GPT_VISION_ENDPOINT")
            effective_api_version = args.api_version or os.environ.get("AZURE_OPENAI_GPT_VISION_API_VERSION")
            effective_deployment = args.deployment_name or os.environ.get("AZURE_OPENAI_GPT_VISION_DEPLOYMENT")

            if not all([effective_api_key, effective_api_base, effective_api_version, effective_deployment]):
                raise ValueError("CRITICAL ERROR: When not using --use-local-llm, one or more Azure OpenAI Vision environment variables/arguments are missing (API_KEY, ENDPOINT, API_VERSION, DEPLOYMENT).")

            vision_config = GPTVision(
                endpoint=effective_api_base,
                api_key=SecretStr(effective_api_key),
                api_version=effective_api_version,
                deployment=effective_deployment
            )

            # Speech and Blob Storage will be loaded automatically from env vars by Pydantic
            env = CobraEnvironment(vision=vision_config)
            
            logging.info(f"Successfully configured environment for Chapter Generation.")
            logging.info(f"LLM Endpoint: {env.vision.endpoint}")
            logging.info(f"LLM Deployment: {env.vision.deployment}")
        else:
            logging.info("Using local LLM. Azure environment will not be configured.")
            # env remains None for the local workflow

        # --- Set Output Directory ---
        output_directory = args.output_dir
        if not output_directory:
            # Default to a directory relative to the video's location on disk
            # This ensures output is saved in the project folder, not the current working directory from where the script is called.
            output_directory = os.path.dirname(os.path.abspath(args.video_path))
            logging.info(f"No output directory specified. Defaulting to: {output_directory}")

        # --- Initialize the generator with the required parameters ---
        generator = ChapterGenerator(
            video_path=args.video_path,
            env=env,
            output_dir=output_directory,
            overwrite_output=args.overwrite_output,
            use_local_llm=args.use_local_llm,
            images_per_segment=args.images_per_segment,
            whisper_model=args.whisper_model,
            # Pass the required parameters for prepare_outputs_directory
            segment_length=args.segment_duration,
            fps=args.fps
        )

        generator.generate(
            fps=args.fps,
            segment_length=args.segment_duration,
            use_scene_detection=args.use_scene_detection,
            scene_detection_threshold=args.scene_detection_threshold,
            downscale_to_max_width=args.downscale_to_max_width,
            downscale_to_max_height=args.downscale_to_max_height,
            enable_language_identification=args.enable_language_id
        )

    except (ValueError, FileNotFoundError) as e:
        logging.critical(f"A configuration error occurred: {e}")
    except Exception as e:
        logging.critical(f"An unexpected error occurred during the pipeline: {e}", exc_info=True)


if __name__ == "__main__":
    main()