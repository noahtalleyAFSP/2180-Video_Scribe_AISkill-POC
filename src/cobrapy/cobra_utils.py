import os
import time
from typing import Union, Optional, List, Dict, Any
from .models.video import VideoManifest
from .models.environment import CobraEnvironment
import subprocess
import concurrent.futures
from shutil import rmtree
import azure.cognitiveservices.speech as speechsdk
from azure.ai.vision.face import FaceAdministrationClient, FaceClient
from azure.ai.vision.face.models import FaceDetectionModel, FaceRecognitionModel, QualityForRecognition
from azure.core.credentials import AzureKeyCredential
import json
import logging

logger = logging.getLogger(__name__)


def encode_image_base64(image_path):
    import base64

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_safe_dir_name(name: str) -> str:
    import re

    # Replace unsafe characters with underscores
    return re.sub(r'[<>:"/\\|?*.]', "_", name).replace(" ", "_")


def generate_transcript(audio_file_path: str, env: CobraEnvironment) -> Dict:
    """
    Generate transcript using Azure Speech Services.
    Returns a dictionary with word-level timing information.
    """
    # First convert audio to correct format for Azure Speech
    wav_path = f"{os.path.splitext(audio_file_path)[0]}_azure.wav"
    cmd = [
        "ffmpeg",
        "-i", audio_file_path,
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", "16000",
        wav_path,
        "-y",
        "-hide_banner",
        "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True)
    
    speech_config = speechsdk.SpeechConfig(
        subscription=env.speech.key.get_secret_value(),
        region=env.speech.region
    )
    
    audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    segments = []
    done = False
    error = None

    def handle_result(evt):
        nonlocal segments
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            segment = {
                "text": evt.result.text,
                "offset": evt.result.offset / 10000000,  # Convert ticks to seconds
                "duration": evt.result.duration / 10000000,
                "words": []
            }
            if hasattr(evt.result, "words"):
                for word in evt.result.words:
                    segment["words"].append({
                        "word": word.word,
                        "offset": word.offset / 10000000,
                        "duration": word.duration / 10000000
                    })
            segments.append(segment)
            logger.debug(f"Transcribed segment: {segment}")

    def handle_canceled(evt):
        nonlocal done, error
        if evt.cancellation_details.reason == speechsdk.CancellationReason.EndOfStream:
            logger.info("Speech recognition completed successfully")
        else:
            error = f"Speech recognition canceled: {evt.cancellation_details.reason}"
            if evt.cancellation_details.error_details:
                error += f"\nError details: {evt.cancellation_details.error_details}"
            logger.error(error)
        done = True

    def handle_session_stopped(evt):
        nonlocal done
        logger.info("Speech recognition stopped")
        done = True

    speech_recognizer.recognized.connect(handle_result)
    speech_recognizer.canceled.connect(handle_canceled)
    speech_recognizer.session_stopped.connect(handle_session_stopped)

    speech_recognizer.start_continuous_recognition()
    
    # Wait for transcription to complete with timeout
    max_wait_time = 600  # 10 minutes timeout
    wait_time = 0
    while not done and wait_time < max_wait_time:
        time.sleep(1)
        wait_time += 1
        if wait_time % 10 == 0:
            logger.info(f"Transcription in progress... {wait_time}/{max_wait_time} seconds")
    
    speech_recognizer.stop_continuous_recognition()
    
    # Clean up temporary WAV file
    try:
        os.remove(wav_path)
    except Exception as e:
        logger.warning(f"Failed to remove temporary WAV file: {e}")

    if error:
        raise Exception(error)

    # Merge nearby segments
    merged_segments = []
    current_segment = None
    for segment in sorted(segments, key=lambda x: x["offset"]):
        if current_segment is None:
            current_segment = segment
        elif segment["offset"] - (current_segment["offset"] + current_segment["duration"]) < 0.5:
            current_segment["duration"] = (segment["offset"] + segment["duration"]) - current_segment["offset"]
            current_segment["text"] += " " + segment["text"]
            if "words" in segment:
                current_segment.setdefault("words", []).extend(segment["words"])
        else:
            merged_segments.append(current_segment)
            current_segment = segment
    
    if current_segment:
        merged_segments.append(current_segment)

    return {
        "text": " ".join(seg["text"] for seg in merged_segments),
        "segments": merged_segments
    }


def parse_transcript(transcript_object: Dict, start_time: float, end_time: float) -> str:
    """Parse the transcript object and return text for the specified time range."""
    if not isinstance(transcript_object, dict):
        raise TypeError("The transcript object must be a dictionary.")
    
    relevant_segments = []
    for segment in transcript_object.get("segments", []):
        segment_start = segment["offset"]
        segment_end = segment_start + segment["duration"]
        
        if (segment_start >= start_time and segment_start < end_time) or \
           (segment_end > start_time and segment_end <= end_time):
            relevant_segments.append(segment["text"])
    
    return " ".join(relevant_segments)


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
        video_path
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True)
        # Parse the JSON output
        ffprobe_info = json.loads(result.stdout)
        
        file_info = {}
        if "streams" in ffprobe_info:
            for stream in ffprobe_info["streams"]:
                if stream["codec_type"] == "video":
                    file_info["video_info"] = stream
                if stream["codec_type"] == "audio":
                    file_info["audio_info"] = stream
        
        if "format" in ffprobe_info:
            file_info["format"] = ffprobe_info["format"]
            if "duration" in ffprobe_info["format"]:
                file_info["duration"] = float(ffprobe_info["format"]["duration"])
        
        return file_info
    except subprocess.CalledProcessError as e:
        print(
            f"Failed to get info for file {video_path}\n"
            f"{e.stderr}", end='', flush=True)
        return None
    except json.JSONDecodeError as e:
        print(
            f"Failed to parse ffprobe output for file {video_path}\n"
            f"Error: {str(e)}\nOutput: {result.stdout}", end='', flush=True)
        return None


def segment_and_extract(
    start_time, end_time, input_video_path, segment_path, frames_dir, fps
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

    # Now extract frames from the segment video using the fps filter
    output_pattern = os.path.join(frames_dir, "frame_%05d.jpg")
    cmd_extract_frames = [
        "ffmpeg",
        "-i",
        segment_video_path,
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",  # Adjust quality if needed
        output_pattern,
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    subprocess.run(cmd_extract_frames, check=True)


def extract_base_audio(video_path, audio_path):
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-q:a",
        "0",
        "-map",
        "a",
        audio_path,
        "-y",  # Overwrite output file if it exists
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    subprocess.run(cmd, check=True)


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
    transcript = generate_transcript(audio_file_path=audio_chunk_path, env=env)
    
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
        # check to see if the path is valid
        if os.path.isfile(video_manifest):
            with open(video_manifest, "r", encoding="utf-8") as f:
                video_manifest = VideoManifest.model_validate_json(
                    json_data=f.read())
            return video_manifest
        else:
            raise FileNotFoundError(
                f"video_manifest file not found in {video_manifest}"
            )
    elif isinstance(video_manifest, VideoManifest):
        return video_manifest
    else:
        raise ValueError(
            "video_manifest must be a string or a VideoManifest object")


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
    if not os.path.exists(asset_directory_path):
        os.makedirs(asset_directory_path)
    else:
        if overwrite_output is True:
            # delete the directory and all of its contents
            rmtree(asset_directory_path)
            os.makedirs(asset_directory_path)
        else:
            raise FileExistsError(
                f"Directory already exists: {asset_directory_path}. If you would like to overwrite it, set overwrite_output=True"
            )
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
