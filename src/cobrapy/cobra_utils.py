import os
import time
from typing import Union, Optional, List, Dict, Any, Tuple
from .models.video import VideoManifest
from .models.environment import CobraEnvironment
import subprocess
import concurrent.futures
from shutil import rmtree, Error as ShutilError
import azure.cognitiveservices.speech as speechsdk
from azure.ai.vision.face import FaceAdministrationClient, FaceClient
from azure.ai.vision.face.models import FaceDetectionModel, FaceRecognitionModel, QualityForRecognition
from azure.core.credentials import AzureKeyCredential
import json
import logging
import requests
from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas
from datetime import datetime, timedelta
import math # Add math import for duration conversion
import cv2 # Added import
import base64 # Added import

logger = logging.getLogger(__name__)


def encode_image_base64(image_path, quality=None):
    import base64
    if quality is not None:
        from PIL import Image
        import io
        # Open image and re-encode as JPEG at given quality
        img = Image.open(image_path)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")
    # Fallback: read raw file bytes
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_safe_dir_name(name: str) -> str:
    import re

    # Replace unsafe characters with underscores
    return re.sub(r'[<>:"/\\|?*.]', "_", name).replace(" ", "_")


def generate_transcript_realtime_sdk(audio_file_path: str, env: CobraEnvironment) -> Dict:
    """
    Generate transcript using Azure Speech SDK (Real-time simulation).
    Returns a dictionary with word-level timing information.
    DEPRECATED in favor of generate_batch_transcript for diarization/languageID.
    """
    print("WARNING: Using deprecated generate_transcript_realtime_sdk.")
    # ... (keep the existing SDK code here if needed for reference) ...
    # ... or just remove it if batch is the only way forward ...
    # For now, let's raise an error to enforce Batch API usage
    raise NotImplementedError("Real-time SDK transcription is deprecated. Use generate_batch_transcript.")


def generate_batch_transcript(
    audio_blob_sas_url: str,
    env: CobraEnvironment,
    candidate_locales: List[str] = ["en-US"], # Default to English
    enable_diarization: bool = True,
    enable_word_timestamps: bool = True,
    job_name_prefix: str = "cobra_batch_",
    enable_language_identification: bool = False # ADDED: New flag
) -> Optional[Dict]:
    """
    Generates transcript using Azure Batch Transcription REST API v3.2.

    Args:
        audio_blob_sas_url: SAS URL for the audio file in Azure Blob Storage.
        env: CobraEnvironment containing Speech and Blob Storage config.
        candidate_locales: List of possible languages for identification.
                           If enable_language_identification is False, the first locale in this list
                           will be used as the primary transcription language.
        enable_diarization: Whether to enable speaker separation.
        enable_word_timestamps: Whether to request word-level timestamps.
        job_name_prefix: Prefix for the transcription job display name.
        enable_language_identification: Whether to enable automatic language identification.
                                        If False, uses the first locale in candidate_locales.
                                        If True, candidate_locales must contain >= 2 locales.

    Returns:
        A dictionary containing the parsed transcription results from the Batch API,
        or None if the job fails.
    """
    start_t_batch = time.time()
    print(f"({get_elapsed_time(start_t_batch)}s) Starting Batch Transcription process...")

    # --- Validate Config ---
    if not env.speech or not env.speech.key or not env.speech.region:
        raise ValueError("Speech key and region missing in environment configuration.")
    if not audio_blob_sas_url:
        raise ValueError("Audio Blob SAS URL is required.")

    # --- Prepare API Request ---
    speech_key = env.speech.key.get_secret_value()
    speech_region = env.speech.region
    endpoint = f"https://{speech_region}.api.cognitive.microsoft.com/speechtotext/v3.2/transcriptions"

    headers = {
        "Ocp-Apim-Subscription-Key": speech_key,
        "Content-Type": "application/json"
    }

    properties = {
        "wordLevelTimestampsEnabled": enable_word_timestamps,
        "punctuationMode": "DictatedAndAutomatic",
        "profanityFilterMode": "Masked"
    }

    if enable_diarization:
         properties["diarizationEnabled"] = True

    # --- Modified Language Identification Logic (v3) ---
    if not candidate_locales or len(candidate_locales) < 1:
         raise ValueError("At least one candidate locale must be provided.")
    base_locale = candidate_locales[0] # Used if LID is off, or as primary for LID

    if enable_language_identification:
        print(f"DEBUG: Language Identification ENABLED.")
        id_locales = list(candidate_locales)
        if len(id_locales) < 2:
            # Add a default second locale if only one was provided and LID is on
            default_second_locale = "es-ES"
            if base_locale.lower() != default_second_locale.lower(): # Avoid adding the same locale twice, case-insensitive
                id_locales.append(default_second_locale)
                print(f"DEBUG: Adding '{default_second_locale}' to meet language ID requirement (min 2) because LID is enabled.")
            else:
                id_locales.append("en-US" if base_locale.lower() != "en-us" else "fr-FR") # Pick another common one
                print(f"DEBUG: Adding a different second locale to meet language ID requirement (min 2) because LID is enabled.")
        
        if len(id_locales) < 2: # Still less than 2 after attempting to add one
            raise ValueError("Language Identification is enabled but less than 2 candidate locales were effectively provided.")

        properties["languageIdentification"] = {
            "candidateLocales": id_locales
        }
        # 'locale' field MUST NOT coexist with 'languageIdentification'
        final_locale_for_payload = None # Will be omitted from payload
        print(f"DEBUG: Requesting language identification with candidates: {id_locales}")
    else:
        print(f"DEBUG: Language Identification DISABLED. Using primary locale: {base_locale}")
        final_locale_for_payload = base_locale
        # Ensure languageIdentification is not in properties if LID is off
        if "languageIdentification" in properties:
            del properties["languageIdentification"]
    # --- End Modified Logic (v3) ---


    payload = {
        "contentUrls": [audio_blob_sas_url],
        # "locale": base_locale, # REMOVED: Handled by final_locale_for_payload
        "displayName": f"{job_name_prefix}{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "properties": properties,
    }

    if final_locale_for_payload:
        payload["locale"] = final_locale_for_payload
    # --- End Payload Update ---

    # --- Submit Transcription Job ---
    print(f"({get_elapsed_time(start_t_batch)}s) Submitting job to {endpoint}...")
    try:
        submit_response = requests.post(endpoint, headers=headers, json=payload)
        submit_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        submit_data = submit_response.json()
        job_status_url = submit_data.get("self")
        if not job_status_url:
             raise ValueError("API did not return a job status URL.")
        print(f"({get_elapsed_time(start_t_batch)}s) Job submitted successfully. Status URL: {job_status_url}")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to submit transcription job: {e}")
        if submit_response is not None:
             print(f"Response Status: {submit_response.status_code}")
             print(f"Response Body: {submit_response.text}")
        return None
    except Exception as e:
         print(f"ERROR: Unexpected error submitting job: {e}")
         return None

    # --- Poll Job Status ---
    print(f"({get_elapsed_time(start_t_batch)}s) Polling job status...")
    job_result_files_url = None
    max_polling_time = 1800 # 30 minutes timeout for polling
    polling_start_time = time.time()
    polling_interval = 30 # seconds

    while time.time() - polling_start_time < max_polling_time:
        try:
            status_response = requests.get(job_status_url, headers=headers)
            status_response.raise_for_status()
            status_data = status_response.json()
            job_status = status_data.get("status")

            print(f"({get_elapsed_time(start_t_batch)}s) Job Status: {job_status}")

            if job_status == "Succeeded":
                job_result_files_url = status_data.get("links", {}).get("files")
                if not job_result_files_url:
                     print("ERROR: Job succeeded but no result files URL found.")
                     return None
                print(f"({get_elapsed_time(start_t_batch)}s) Job Succeeded. Result files URL: {job_result_files_url}")
                break
            elif job_status == "Failed":
                print(f"ERROR: Transcription job failed.")
                print(f"Failure details: {status_data.get('properties', {}).get('error')}")
                return None
            elif job_status in ["NotStarted", "Running"]:
                 time.sleep(polling_interval)
            else:
                 print(f"ERROR: Unknown job status encountered: {job_status}")
                 return None

        except requests.exceptions.RequestException as e:
            print(f"ERROR: Failed to poll job status: {e}")
            # Optional: retry polling a few times before failing
            time.sleep(polling_interval * 2) # Wait longer on error
        except Exception as e:
            print(f"ERROR: Unexpected error during polling: {e}")
            return None
    else: # Loop exited due to timeout
         print(f"ERROR: Polling timed out after {max_polling_time} seconds.")
         return None

    # --- Retrieve Result File URL ---
    if not job_result_files_url: return None # Should not happen if loop exited correctly

    print(f"({get_elapsed_time(start_t_batch)}s) Retrieving result file list...")
    try:
        files_response = requests.get(job_result_files_url, headers=headers)
        files_response.raise_for_status()
        files_data = files_response.json()

        transcription_file_url = None
        for file_info in files_data.get("values", []):
            if file_info.get("kind") == "Transcription":
                 transcription_file_url = file_info.get("links", {}).get("contentUrl")
                 break

        if not transcription_file_url:
             print("ERROR: Could not find Transcription file URL in results.")
             return None
        print(f"({get_elapsed_time(start_t_batch)}s) Transcription result file URL: {transcription_file_url}")

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to retrieve result file list: {e}")
        return None
    except Exception as e:
         print(f"ERROR: Unexpected error retrieving file list: {e}")
         return None

    # --- Download and Parse Result File ---
    print(f"({get_elapsed_time(start_t_batch)}s) Downloading transcription result...")
    try:
        # Note: The result file URL from Azure is typically a SAS URL itself
        result_response = requests.get(transcription_file_url)
        result_response.raise_for_status()
        # Ensure content is decoded correctly (Azure usually uses UTF-8)
        result_response.encoding = result_response.apparent_encoding or 'utf-8'
        result_json = result_response.json()
        print(f"({get_elapsed_time(start_t_batch)}s) Transcription result downloaded and parsed.")

        # --- Optional: Delete Job ---
        # Consider deleting the job to clean up Azure resources
        # try:
        #     print(f"({get_elapsed_time(start_t_batch)}s) Deleting transcription job...")
        #     delete_response = requests.delete(job_status_url, headers=headers)
        #     delete_response.raise_for_status()
        #     print(f"({get_elapsed_time(start_t_batch)}s) Job deleted.")
        # except Exception as e:
        #     print(f"Warning: Failed to delete transcription job {job_status_url}: {e}")

        return result_json

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to download result file: {e}")
        if result_response is not None:
            print(f"Download Response Status: {result_response.status_code}")
            print(f"Download Response Body: {result_response.text[:500]}...") # Show start of body
        return None
    except json.JSONDecodeError as e:
         print(f"ERROR: Failed to parse result JSON: {e}")
         print(f"Downloaded content: {result_response.text[:500]}...")
         return None
    except Exception as e:
        print(f"ERROR: Unexpected error processing result file: {e}")
        return None


def parse_transcript(transcript_object: Dict, start_time: float, end_time: float) -> str:
    """
    Parse the transcript object (from Batch API) and return text for the specified time range.
    """
    if not isinstance(transcript_object, dict):
        # print(f"Warning: Expected dict for transcript object, got {type(transcript_object)}")
        return "Transcription data unavailable or invalid."

    relevant_phrases = []
    # Use the 'recognizedPhrases' key expected from Batch API
    for phrase in transcript_object.get("recognizedPhrases", []):
        if not isinstance(phrase, dict): continue

        try:
             # Ticks to seconds conversion
             phrase_start = phrase.get("offsetInTicks", 0) / 10_000_000.0
             phrase_duration = phrase.get("durationInTicks", 0) / 10_000_000.0
             phrase_end = phrase_start + phrase_duration

             # Check for overlap with the segment time range
             # Phrase starts within segment OR Phrase ends within segment OR Phrase engulfs segment
             if (phrase_start < end_time) and (phrase_end > start_time):
                  # Extract the display text from the best hypothesis
                  best_recognition = phrase.get("nBest", [{}])[0]
                  display_text = best_recognition.get("display", None)
                  if display_text:
                       relevant_phrases.append(display_text)
        except (TypeError, IndexError, KeyError, ValueError) as e:
             print(f"Warning: Error processing transcript phrase: {e}. Data: {phrase}")
             continue

    return " ".join(relevant_phrases) if relevant_phrases else "No transcription for this time range."


def get_file_info(video_path):
    """Get video file information using ffprobe."""
    import json
    import subprocess

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        "-select_streams", "a:0",  # Explicitly select first audio stream
        video_path
    ]

    try:
        print(f"[DEBUG] Running ffprobe command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        ffprobe_info = json.loads(result.stdout)
        file_info = {}
        
        # First check for audio stream
        has_audio = False
        for stream in ffprobe_info.get("streams", []):
            if stream["codec_type"] == "audio":
                has_audio = True
                file_info["audio_info"] = stream
                break
        
        # If no audio found through streams, try alternative check
        if not has_audio:
            cmd_check_audio = [
                "ffmpeg",
                "-i", video_path,
                "-af", "volumedetect",
                "-f", "null",
                "-hide_banner",
                "-"
            ]
            try:
                result = subprocess.run(cmd_check_audio, capture_output=True, text=True)
                has_audio = "mean_volume" in result.stderr
            except Exception as e:
                print(f"Secondary audio check failed: {e}")
        
        # Now get video info
        cmd_video = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            "-select_streams", "v:0",  # Select first video stream
            video_path
        ]
        video_result = subprocess.run(cmd_video, capture_output=True, text=True, check=True)
        video_info = json.loads(video_result.stdout)
        
        for stream in video_info.get("streams", []):
            if stream["codec_type"] == "video":
                file_info["video_info"] = stream
                break
        
        if "format" in ffprobe_info:
            file_info["format"] = ffprobe_info["format"]
            if "duration" in ffprobe_info["format"]:
                file_info["duration"] = float(ffprobe_info["format"]["duration"])
            # --- ADDED: Extract format name ---
            if "format_name" in ffprobe_info["format"]:
                 file_info["format_name"] = ffprobe_info["format"]["format_name"]
            # --- END ADDED ---
        
        # Set audio_found flag based on our checks
        file_info["audio_found"] = has_audio
        
        return file_info
    except FileNotFoundError as e:
        print(f"[ERROR] ffprobe not found. Please install FFmpeg and add it to your system PATH.")
        print(f"Full error: {e}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffprobe failed for file {video_path}")
        print(f"Command: {' '.join(cmd)}")
        print(f"stderr: {e.stderr}")
        return None
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse ffprobe output for file {video_path}")
        print(f"Error: {str(e)}\nOutput: {result.stdout}")
        return None


def segment_and_extract(
    start_time, end_time, input_video_path, segment_path, frames_dir, fps,
    downscale_to_max_width=None, downscale_to_max_height=None
):
    segment_file_name = "segment.mp4"
    segment_video_path = os.path.join(segment_path, segment_file_name)
    cmd_extract_segment = [
        "ffmpeg",
        "-ss",
        str(start_time),
        "-to",
        str(end_time),
        "-i",
        input_video_path,
        "-c",
        "copy",
        segment_video_path,
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    subprocess.run(cmd_extract_segment, check=True)

    # Now extract frames from the segment video using the fps filter and optional scale
    output_pattern = os.path.join(frames_dir, "frame_%05d.jpg")
    vf_filters = [f"fps={fps}"]
    scale_filter = None
    if downscale_to_max_width and downscale_to_max_height:
        scale_filter = f"scale='min({downscale_to_max_width},iw)':'min({downscale_to_max_height},ih)':force_original_aspect_ratio=decrease"
    elif downscale_to_max_width:
        scale_filter = f"scale='min({downscale_to_max_width},iw)':-2"
    elif downscale_to_max_height:
        scale_filter = f"scale=-2:'min({downscale_to_max_height},ih)'"
    if scale_filter:
        vf_filters.append(scale_filter)
    vf_arg = ",".join(vf_filters)
    cmd_extract_frames = [
        "ffmpeg",
        "-i",
        segment_video_path,
        "-vf",
        vf_arg,
        "-q:v",
        "2",  # Adjust quality if needed
        output_pattern,
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    subprocess.run(cmd_extract_frames, check=True)


def extract_base_audio(video_path, audio_path):
    # Ensure the output path has a .wav extension
    if not audio_path.lower().endswith(".wav"):
         print(f"Warning: Forcing audio output path to .wav for compatibility: {audio_path}")
         # Correct the path if needed (optional, but good practice)
         audio_path = os.path.splitext(audio_path)[0] + ".wav"

    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        # --- WAV Specific Settings ---
        "-vn",  # Disable video recording
        "-acodec", "pcm_s16le",  # Use 16-bit PCM codec
        "-ar", "16000",  # Set audio sample rate to 16kHz
        "-ac", "1",  # Set audio channels to 1 (mono)
        # --- End WAV Specific Settings ---
        # Remove MP3 quality setting: "-q:a", "0", 
        # Remove map setting as we only want audio: "-map", "a", 
        audio_path,
        "-y",  # Overwrite output file if it exists
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    print(f"Running ffmpeg to extract WAV: {' '.join(cmd)}") # Log the command
    try:
        subprocess.run(cmd, check=True)
        print("ffmpeg WAV extraction completed successfully.")
    except subprocess.CalledProcessError as e:
         print(f"ffmpeg WAV extraction failed: {e}")
         # Optionally capture and print stderr
         # print(f"ffmpeg stderr: {e.stderr}")
         raise # Re-raise the error to stop processing


def extract_audio_chunk(args):
    video_path, start, end, audio_chunk_path = args
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-ss",
        str(start),
        "-to",
        str(end),
        "-q:a",
        "0",
        "-map",
        "a",
        audio_chunk_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    subprocess.run(cmd, check=True)
    return audio_chunk_path, start


def parallelize_audio(extract_args_list, max_workers):
    print(
        f"Extracting audio chunks in parallel using {max_workers} workers...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        extracted_chunks = list(executor.map(
            extract_audio_chunk, extract_args_list))
        return extracted_chunks


def parallelize_transcription(process_args_list):
    print(f"Processing audio chunks in parallel using 2 workers...")
    
    # Process audio chunks in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        transcripts = list(executor.map(process_chunk, process_args_list))

    if not transcripts:
        return {"text": "", "segments": []}
        
    # Combine transcripts
    combined_segments = []
    combined_text = ""
    
    for transcript in transcripts:
        if transcript and "text" in transcript:
            combined_text += transcript["text"] + " "
            if "segments" in transcript:
                combined_segments.extend(transcript["segments"])
    
    # Sort segments by offset
    combined_segments.sort(key=lambda x: x["offset"])
    
    return {
        "text": combined_text.strip(),
        "segments": combined_segments
    }


def process_chunk(args):
    audio_chunk_path, start_time = args
    env = CobraEnvironment()  # Create a default environment to use the speech services
    # transcript = generate_transcript(audio_file_path=audio_chunk_path, env=env) # <-- Commented out problematic line
    transcript = {} # Assign empty dict to avoid subsequent errors if transcript is used

    # Adjust timestamps for words
    for word in transcript.get("segments", []):
        words = word.get("words", [])
        for w in words:
            w["offset"] += start_time
    
    # Adjust timestamps for segments
    for segment in transcript.get("segments", []):
        segment["offset"] += start_time
    
    return transcript


def validate_video_manifest(video_manifest: Union[str, VideoManifest]) -> VideoManifest:
    if isinstance(video_manifest, str):
        # Check if the path is valid
        if not os.path.isfile(video_manifest):
            raise FileNotFoundError(
                f"Input file not found: {video_manifest}"
            )
        
        # Check if the file is a JSON manifest
        if not video_manifest.lower().endswith('.json'):
            raise ValueError(
                f"Expected a path to a JSON manifest file or a VideoManifest object, but received a path to a non-JSON file: {video_manifest}. "
                f"VideoAnalyzer should be initialized directly from a VideoManifest object or a path to its JSON file."
            )

        # If it's a JSON file, read and validate it
        try:
            with open(video_manifest, "r", encoding="utf-8") as f:
                manifest_obj = VideoManifest.model_validate_json(
                    json_data=f.read()
                )
            return manifest_obj
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from manifest file {video_manifest}: {e}")
        except Exception as e:
            raise IOError(f"Error reading manifest file {video_manifest}: {e}")

    elif isinstance(video_manifest, VideoManifest):
        return video_manifest
    else:
        raise ValueError(
            "video_manifest must be a string path to a JSON file or a VideoManifest object")


def get_elapsed_time(start_time):
    import time

    elapsed = time.time() - start_time
    return "{:.1f}s".format(elapsed)


def write_video_manifest(manifest):
    video_manifest_path = os.path.join(
        manifest.processing_params.output_directory, f"_video_manifest.json"
    )
    with open(video_manifest_path, "w", encoding="utf-8") as f:
        # Remove ensure_ascii parameter and handle UTF-8 manually
        json_data = manifest.model_dump_json(indent=4)
        # Load and re-dump with ensure_ascii=False to preserve Unicode characters
        json_obj = json.loads(json_data)
        f.write(json.dumps(json_obj, indent=4, ensure_ascii=False))

    print(f"Video manifest for {manifest.name} saved to {video_manifest_path}")

    manifest.video_manifest_path = video_manifest_path


def prepare_outputs_directory(
    file_name: str,
    segment_length: int,
    frames_per_second: float,
    output_directory: Optional[str] = None,
    overwrite_output=False,
    output_directory_prefix="",
):

    if output_directory is None:
        safe_dir_name = generate_safe_dir_name(file_name)
        asset_directory_name = f"{output_directory_prefix}{safe_dir_name}_{frames_per_second:.2f}fps_{segment_length}sSegs_cobra"
        asset_directory_path = os.path.join(
            ".",
            asset_directory_name,
        )
    else:
        asset_directory_path = output_directory

    # Create output directory if it doesn't exist. If it does exist, check if we should overwrite it
    if os.path.exists(asset_directory_path):
        if overwrite_output is True:
            print(f"Output directory {asset_directory_path} exists. Overwriting...")
            attempts = 3
            delay = 1 # seconds
            for i in range(attempts):
                try:
                    rmtree(asset_directory_path)
                    print("Existing directory removed successfully.")
                    break # Exit loop if successful
                except (OSError, ShutilError) as e:
                    print(f"Warning: Attempt {i+1}/{attempts} failed to remove directory: {e}")
                    if i < attempts - 1:
                        print(f"Retrying in {delay} second(s)...")
                        time.sleep(delay)
                    else:
                        print(f"ERROR: Failed to remove existing directory {asset_directory_path} after {attempts} attempts.")
                        raise # Re-raise the last exception if all attempts fail
            # Recreate the directory after successful removal
            os.makedirs(asset_directory_path)
        else:
            # Directory exists but overwrite is False
            raise FileExistsError(
                f"Directory already exists: {asset_directory_path}. If you would like to overwrite it, set overwrite_output=True"
            )
    else:
        # Directory doesn't exist, create it
        os.makedirs(asset_directory_path)
        print(f"Created output directory: {asset_directory_path}")

    return asset_directory_path


def get_face_client(env: CobraEnvironment) -> FaceClient:
    """Get an Azure Face API client."""
    return FaceClient(
        endpoint=env.face.endpoint,
        credential=AzureKeyCredential(env.face.apikey.get_secret_value())
    )


def get_face_admin_client(env: CobraEnvironment) -> FaceAdministrationClient:
    """Get an Azure Face Administration API client."""
    return FaceAdministrationClient(
        endpoint=env.face.endpoint,
        credential=AzureKeyCredential(env.face.apikey.get_secret_value())
    )


def ensure_person_group_exists(person_group_id: str, env: CobraEnvironment) -> bool:
    """Ensure a person group exists before using it."""
    if not person_group_id:
        return False
        
    try:
        admin_client = get_face_admin_client(env)
        admin_client.large_person_group.get(large_person_group_id=person_group_id)
        logger.info(f"Person group '{person_group_id}' exists")
        return True
    except Exception as e:
        logger.warning(f"Error checking person group: {e}")
        return False


def analyze_faces(image_path: str, env: CobraEnvironment, person_group_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Analyze faces in an image using Azure Face API.
    If person_group_id is provided, attempts to identify the faces.
    
    Returns:
        List of dictionaries with face information including:
        - face_id: The detected face ID
        - rectangle: Face rectangle coordinates
        - confidence: Detection confidence
        - identified_person: (if person_group_id provided) 
            - name: Person name
            - confidence: Identification confidence
    """
    if not os.path.exists(image_path):
        logger.warning(f"Image file not found: {image_path}")
        return []
        
    try:
        # Initialize clients
        face_client = get_face_client(env)
        
        # If person_group_id is provided, verify it exists
        use_identification = False
        if person_group_id:
            use_identification = ensure_person_group_exists(person_group_id, env)
        
        # Detect faces in the image
        with open(image_path, "rb") as image_file:
            file_content = image_file.read()
        
        # Try to detect faces with standard detection first
        logger.info(f"Detecting faces in {image_path}")
        try:
            detected_faces = face_client.detect(
                image_content=file_content,
                detection_model=FaceDetectionModel.DETECTION03,
                recognition_model=FaceRecognitionModel.RECOGNITION04,
                return_face_id=True,
                # Don't request quality assessment to avoid QUALITY_FOR_RECOGNITION error
                return_face_attributes=[]
            )
        except Exception as e:
            logger.warning(f"Face detection failed with error: {e}")
            # If detection fails completely, return empty results
            return []
        
        if not detected_faces:
            logger.info(f"No faces detected in {image_path}")
            return []
            
        results = []
        for face in detected_faces:
            face_info = {
                "face_id": face.face_id,
                "rectangle": {
                    "top": face.face_rectangle.top,
                    "left": face.face_rectangle.left,
                    "width": face.face_rectangle.width,
                    "height": face.face_rectangle.height
                },
                "confidence": getattr(face, "detection_confidence", None)
            }
            results.append(face_info)
            
        # If a person group ID is provided and valid, identify the faces
        if use_identification and results:
            # Use all faces for identification
            face_ids = [face["face_id"] for face in results]
            
            if face_ids:
                logger.info(f"Identifying {len(face_ids)} faces against person group '{person_group_id}'")
                
                # Process in batches of 10 to avoid API limits
                BATCH_SIZE = 10
                all_identify_results = []
                
                # Create batches of face IDs
                for i in range(0, len(face_ids), BATCH_SIZE):
                    batch_face_ids = face_ids[i:i + BATCH_SIZE]
                    logger.info(f"Processing batch of {len(batch_face_ids)} faces (batch {i//BATCH_SIZE + 1}/{(len(face_ids) + BATCH_SIZE - 1)//BATCH_SIZE})")
                    
                    try:
                        batch_results = face_client.identify_from_large_person_group(
                            face_ids=batch_face_ids,
                            large_person_group_id=person_group_id,
                            # Reduce confidence threshold to increase chances of identification
                            confidence_threshold=0.5,
                            max_num_of_candidates_returned=5
                        )
                        all_identify_results.extend(batch_results)
                    except Exception as e:
                        logger.error(f"Error in face identification batch: {e}")
                        # Continue with next batch instead of failing entirely
                        continue
                
                if all_identify_results:
                    # Map person IDs to names
                    admin_client = get_face_admin_client(env)
                    
                    # Process identification results
                    for identify_result in all_identify_results:
                        face_id = identify_result.face_id
                        # Find the corresponding face info
                        for face_info in results:
                            if face_info["face_id"] == face_id:
                                if identify_result.candidates:
                                    # Get the top candidate
                                    candidate = identify_result.candidates[0]
                                    # Get person details
                                    try:
                                        person = admin_client.large_person_group.get_person(
                                            large_person_group_id=person_group_id,
                                            person_id=candidate.person_id
                                        )
                                        face_info["identified_person"] = {
                                            "name": person.name,
                                            "confidence": candidate.confidence
                                        }
                                    except Exception as e:
                                        logger.error(f"Error getting person details: {e}")
                                        face_info["identified_person"] = {
                                            "name": "Unknown",
                                            "confidence": candidate.confidence
                                        }
                                break
                    
        return results
    except Exception as e:
        logger.error(f"Error in face analysis: {e}")
        return []


def process_frame_with_faces(image_path: str, env: CobraEnvironment, person_group_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a single frame, extracting face information.
    Returns the frame with face detection/recognition results.
    """
    try:
        face_results = analyze_faces(image_path, env, person_group_id)
        
        result = {
            "frame_path": image_path,
            "faces": face_results,
            "identified_people": []
        }
        
        # Create a list of identified people for easier access
        if face_results:
            for face in face_results:
                if "identified_person" in face and face["identified_person"]["confidence"] > 0.5:  # Lower threshold for more matches
                    result["identified_people"].append(face["identified_person"]["name"])
        
        return result
    except Exception as e:
        logger.error(f"Error processing frame with faces: {e}")
        # Return a minimal result structure if processing fails
        return {
            "frame_path": image_path,
            "faces": [],
            "identified_people": []
        }


def upload_blob(
    local_file_path: str,
    blob_name: str,
    env: CobraEnvironment,
    overwrite: bool = True,
    read_permission_hours: int = 48 # Default SAS expiry for read access
) -> Optional[str]:
    """
    Uploads a local file to Azure Blob Storage and returns a SAS URL.

    Args:
        local_file_path: Path to the local file to upload.
        blob_name: Name for the blob in Azure Storage.
        env: CobraEnvironment containing Blob Storage config.
        overwrite: Whether to overwrite the blob if it exists.
        read_permission_hours: Duration in hours for which the generated SAS token is valid for reading.

    Returns:
        A SAS URL string for the uploaded blob with read permissions, or None on failure.
    """
    if not env.blob_storage or not env.blob_storage.account_name or not env.blob_storage.container_name:
        logger.error("Blob storage account/container name not configured in environment.")
        return None

    if not env.blob_storage.connection_string and not env.blob_storage.sas_token:
         logger.error("Blob storage connection string or SAS token not configured in environment.")
         return None

    blob_service_client = None
    try:
        if env.blob_storage.connection_string:
            connect_str = env.blob_storage.connection_string.get_secret_value()
            blob_service_client = BlobServiceClient.from_connection_string(connect_str)
            # logger.info(f"Connected to Blob Storage using connection string.")
        elif env.blob_storage.sas_token:
            account_url = f"https://{env.blob_storage.account_name}.blob.core.windows.net"
            sas_token_str = env.blob_storage.sas_token.get_secret_value()
            blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token_str)
            # logger.info(f"Connected to Blob Storage using SAS token.")
        else:
            # This case should be caught by the initial check, but belt-and-suspenders
            logger.error("No valid authentication method found for Blob Storage.")
            return None

        blob_client = blob_service_client.get_blob_client(
            container=env.blob_storage.container_name,
            blob=blob_name
        )

        # logger.info(f"Uploading {local_file_path} to blob: {env.blob_storage.container_name}/{blob_name}")
        with open(local_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=overwrite)
        # logger.info("Upload successful.")

        # Generate SAS token with read permission
        sas_token = generate_blob_sas(
            account_name=blob_client.account_name,
            container_name=blob_client.container_name,
            blob_name=blob_client.blob_name,
            account_key=blob_service_client.credential.account_key if hasattr(blob_service_client.credential, 'account_key') else None, # Needed if using account key via conn string
            user_delegation_key=None, # Not typically used for service SAS
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=read_permission_hours),
            # start=datetime.utcnow() - timedelta(minutes=5) # Optional: Add start time
        )

        sas_url = f"{blob_client.url}?{sas_token}"
        # logger.info(f"Generated SAS URL (valid for {read_permission_hours} hours).")
        return sas_url

    except Exception as e:
        logger.error(f"Failed to upload blob or generate SAS URL: {e}")
        return None


def create_basic_manifest_from_video(video_path: str) -> VideoManifest:
    """Creates a basic VideoManifest object from a video file path, populating source info."""
    manifest = VideoManifest()

    # Check that the video file exists
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    else:
        manifest.name = os.path.basename(video_path)
        manifest.source_video.path = os.path.abspath(video_path)

    # Get video metadata
    file_metadata = get_file_info(video_path)
    if file_metadata is not None:
        duration_sec = file_metadata.get("duration", None) # Get duration early
        duration_iso = seconds_to_iso8601_duration(duration_sec) # Calculate ISO duration

        manifest_source = {
            "path": video_path,
            "video_found": False,
            "size": [],
            "rotation": 0,
            "fps": 0,
            "duration": duration_sec, # Use stored duration_sec
            "duration_iso": duration_iso, # Added ISO duration
            "nframes": 0,
            "audio_found": file_metadata.get("audio_found", False), # Use the flag from get_file_info
            "audio_duration": 0,
            "audio_fps": 0,
            "format_name": file_metadata.get("format_name", None), # Added format name
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
            elif manifest_source["duration"] and manifest_source["fps"] > 0:
                manifest_source["nframes"] = int(manifest_source["duration"] * manifest_source["fps"])

            # Handle rotation
            if "side_data_list" in video_info:
                for side_data in video_info["side_data_list"]:
                    if "rotation" in side_data:
                        manifest_source["rotation"] = int(side_data["rotation"])
                        break

        if manifest_source["audio_found"] and "audio_info" in file_metadata:
            audio_info = file_metadata["audio_info"]
            if "duration" in audio_info:
                manifest_source["audio_duration"] = float(audio_info["duration"])
            
            # Handle audio sample rate (not FPS)
            sample_rate_str = audio_info.get("sample_rate", "0")
            try:
                manifest_source["audio_fps"] = int(sample_rate_str) # Store sample rate here for simplicity
            except ValueError:
                 manifest_source["audio_fps"] = 0

        manifest.source_video = manifest.source_video.model_copy(
            update=manifest_source
        )

    return manifest


# --- Helper function for ISO 8601 Duration ---
def seconds_to_iso8601_duration(seconds: float) -> Optional[str]:
    if seconds is None or seconds < 0:
        return None
    
    total_seconds = seconds
    
    hours = int(total_seconds // 3600)
    total_seconds %= 3600
    minutes = int(total_seconds // 60)
    total_seconds %= 60
    secs = round(total_seconds, 3) # Keep milliseconds
    
    duration_string = "PT"
    if hours > 0:
        duration_string += f"{hours}H"
    if minutes > 0:
        duration_string += f"{minutes}M"
    # Always include seconds, even if 0, unless duration is exactly 0
    if secs > 0 or duration_string == "PT":
         # Format seconds to handle integers and floats cleanly
         if secs == int(secs):
              duration_string += f"{int(secs)}S"
         else:
              duration_string += f"{secs:.3f}S".rstrip('0').rstrip('.') + "S" # Avoid trailing zeros/periods

    # Handle case where duration is exactly 0
    if duration_string == "PT":
        return "PT0S"
        
    return duration_string
# --- End Helper ---

# --- ADDED UTILITY FUNCTIONS (from track_refiner.py) ---
def _pad_bbox(bbox: List[int], frame_shape: Tuple[int, int], pct: float = 0.10) -> Tuple[int, int, int, int]:
    """Pads a bounding box. frame_shape is (height, width)."""
    x1, y1, x2, y2 = bbox
    frame_height, frame_width = frame_shape

    box_h, box_w = y2 - y1, x2 - x1
    pad_w, pad_h = int(box_w * pct), int(box_h * pct)

    x1p = max(0, x1 - pad_w)
    y1p = max(0, y1 - pad_h)
    x2p = min(frame_width - 1, x2 + pad_w)
    y2p = min(frame_height - 1, y2 + pad_h)
    return int(x1p), int(y1p), int(x2p), int(y2p)


def get_frame_crop_base64(video_path: str, frame_number: int, bbox: List[int], crop_padding_percentage: float = 0.10) -> Optional[str]:
    """
    Opens video, seeks to frame_number, reads frame, crops using bbox, encodes to base64.
    bbox: [x1, y1, x2, y2]
    crop_padding_percentage: Percentage to pad the bounding box.
    Returns: base64 encoded string of JPEG image or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Use logger if available and configured, otherwise print
        logger.error(f"get_frame_crop_base64: Cannot open video {video_path}")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_number)) # Frame number should be 0-indexed for seeking
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        logger.error(f"get_frame_crop_base64: Failed to read frame {frame_number} from {video_path}")
        return None

    try:
        # Apply padding to bbox before cropping
        padded_bbox = _pad_bbox(bbox, frame.shape[:2], pct=crop_padding_percentage) # frame.shape is (height, width, channels)
        x1, y1, x2, y2 = padded_bbox
        
        # Ensure coordinates are valid after padding
        if x1 >= x2 or y1 >= y2:
            # Fallback to original if padding makes it invalid
            logger.warning(f"get_frame_crop_base64: Invalid bbox [{x1},{y1},{x2},{y2}] after padding for frame {frame_number}. Using original bbox.")
            x1_orig, y1_orig, x2_orig, y2_orig = bbox
            crop = frame[y1_orig:y2_orig, x1_orig:x2_orig]
        else:
            crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            logger.warning(f"get_frame_crop_base64: Crop is empty for frame {frame_number}, bbox {bbox}. Path: {video_path}")
            return None

        is_success, buffer = cv2.imencode('.jpg', crop)
        if not is_success:
            logger.error(f"get_frame_crop_base64: Failed to encode crop to JPEG for frame {frame_number}. Path: {video_path}")
            return None
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"get_frame_crop_base64: Exception during cropping/encoding frame {frame_number}: {e}", exc_info=True)
        return None
# --- END ADDED UTILITY FUNCTIONS ---
