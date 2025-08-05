#!/usr/bin/env python3
# -*- coding: cp1252 -*-
import asyncio
import base64
import concurrent.futures
import json
import logging
import os
import shutil
import time
import gc
import torch
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import openai
from cobrapy.local_models.local_phi4_adapter import LocalPhi4Adapter
from cobrapy.local_models.transcriber import LocalTranscriber

from .cobra_utils import (
    extract_base_audio,
    generate_batch_transcript,
    generate_safe_dir_name,
    get_elapsed_time,
    get_file_info,
    prepare_outputs_directory,
    seconds_to_iso8601_duration,
    segment_and_extract,
    validate_video_manifest,
    write_video_manifest,
    estimate_image_tokens,
    upload_blob
)
from .models.environment import CobraEnvironment
from .models.video import Frame, Segment, VideoManifest

logger = logging.getLogger(__name__)

def log_gpu_memory(log_point: str):
    """Logs the current GPU memory usage."""
    if torch.cuda.is_available():
        logger.info(f"--- GPU Memory at: {log_point} ---")
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"  -> Allocated: {allocated:.2f} GiB")
        logger.info(f"  -> Reserved:  {reserved:.2f} GiB")

class ChapterGenerator:
    """
    A streamlined processor to generate transcription and chapters for a video.
    This class combines preprocessing and analysis steps for a synchronous workflow.
    """

    SHOT_TYPES: List[str] = [
        "Establishing Shot", "Extreme Wide Shot (EWS)", "Wide Shot (WS)", "Full Shot (FS)",
        "Medium Wide Shot (MWS)", "Medium Long Shot (MLS)", "Medium Shot (MS)", "Cowboy Shot",
        "Medium Close-Up (MCU)", "Close-Up (CU)", "Extreme Close-Up (ECU)", "Two-Shot",
        "Three-Shot", "Reverse Angle", "Over-the-Shoulder", "Point-of-View (POV) Shot",
        "Reaction Shot", "Insert Shot", "Cutaway", "Dutch Angle", "Tracking/Dolly Shot",
        "Crane/Jib Shot", "Handheld/Steadicam Shot", "Whip-Pan (Swish-Pan)", "Special / Other"
    ]

    system_prompt_vision_chapters: str = (
        """You are VideoAnalyzerGPT, focused on describing video content within specific time ranges. Your goal is to generate a single, detailed description for the provided time range, using precise timestamps based *only* on the frame data provided.


ABSOLUTELY FORBIDDEN WORDS & PHRASES
You must **NEVER** use any of the following terms in your response:
"video", "clip", "footage", "camera", "shot begins/starts/opens", "audio",
"transcription", "segment", "scene", "frame", "image", "viewer",
or meta-phrases such as "in this chapter/segment/clip".

STYLE GUIDELINES
* Write in the PRESENT tense, third-person narrative.
* Begin with the main subject or action (e.g., "A man in a sharp grey suit sits...").
* Blend what is visible and audible into a single cinematic paragraph.
* Vary vocabulary; aim for evocative storytelling, not enumeration.

ENTIMENT / EMOTIONS / THEME
Infer the dominant **sentiment** ("positive", "neutral", "negative")
and up to **3** nuanced **emotions** (e.g., "anticipation", "anxiety").
Pick a concise **theme** (<= 3 words) that captures the underlying idea
(e.g., "professional tension", "nostalgic reunion").
Do **NOT** default to "neutral" unless absolutely nothing suggests otherwise.



**CRITICAL INSTRUCTIONS - READ CAREFULLY:**

1.  **JSON STRUCTURE:** You MUST return a valid JSON object with ONLY the top-level key: "chapters".
    * "chapters": An array containing EXACTLY ONE chapter object describing the content within the specified time range.

2.  **SHOT ANALYSIS:** Within the chapter object, include:
    * `shotType`: Classify the dominant camera shot types observed during this segment. Provide this as a **LIST of strings**. Choose **one or more** from the following list: {shot_types_list}. If the shot changes (e.g., zoom, dolly), include all relevant types observed.
    * `shotDescription`: Concisely describe the shot's composition, camera work, and visual style. Mention key elements like subjects, setting, color grading, and camera movements (e.g., "Dolly In", "Zoom Out"). Avoid starting with "The shot starts as...".

3.  **EXACT OUTPUT FORMAT:** Use the *actual start and end times* for this segment (`{start_time}s` to `{end_time}s`) in the `start` and `end` fields of your JSON output.
    ```json
    {{
      "chapters": [
        {{
          "start": "{start_time}s",
          "end": "{end_time}s",
          "shotType": ["Medium Shot (MS)", "Close-Up (CU)"],
          "shotDescription": "The segment features a transition from a Medium Shot to a Close-Up, focusing on a character's reaction. The camera performs a slow zoom, and the visual style is cinematic with vibrant color grading.",
          "sentiment": "neutral",
          "emotions": ["emotion1", "emotion2"],
          "theme": "short theme",
          "summary": "A character receives some news, their expression shifting from neutral to surprised as the conversation continues in the background..."
        }}
      ]
    }}
    ```

4.  **TIMESTAMP PRECISION & ACCURACY:**
    * Use the absolute start ({start_time}s) and end ({end_time}s) times provided for the current time range.
    * Format: "0.000s".

5.  **SUMMARY CONTENT (NARRATIVE DESCRIPTION):**
    * **Style:** Write a descriptive, present-tense narrative of what is happening in this time range. Describe the scene directly as if the viewer is watching it unfold.
    * **Content:** Weave together the setting, visuals, character actions, and key information from the transcription into a cohesive paragraph. Be detailed and verbose.
    * **CRITICAL: Do not break the fourth wall.** This means you MUST NOT use phrases that refer to the video, the camera, or the analysis process. For example, **strictly avoid** phrases like: "The video begins with...", "In this segment...", "The scene opens with...", "This clip shows...", or "The camera focuses on...".
    * **Continuity:** Write the summary as a continuation of an ongoing story. Do not write it as a standalone description. Assume the reader already has context from the previous moments in the video. Start the description directly with the most important action or visual element of the segment.
    * **Accuracy:** Base your entire description *only* on the provided frame images and transcription text. DO NOT invent details, characters, settings, or events not directly observable in the input.
"""
    )

    system_prompt_text_chapters: str = (
    """You are TextAnalyzerGPT, focused on analyzing transcribed audio content. Your goal is to generate a single, detailed analysis for the provided time range, based *only* on the transcription provided.

ABSOLUTELY FORBIDDEN WORDS & PHRASES
You must **NEVER** use any of the following terms in your response:
"video", "clip", "footage", "camera", "shot", "audio", "frame", "image", "viewer", "scene", "segment",
or meta-phrases such as "in this chapter/segment/clip".

STYLE GUIDELINES
* Write the summary in the PRESENT tense, third-person narrative.
* Blend the key points from the transcription into a cohesive paragraph.
* Vary vocabulary; aim for insightful summarization, not just repetition.

SENTIMENT / EMOTIONS / THEME
Infer the dominant **sentiment** ("positive", "neutral", "negative")
and up to **3** nuanced **emotions** (e.g., "anticipation", "anxiety").
Pick a concise **theme** (<= 3 words) that captures the underlying idea
(e.g., "professional tension", "nostalgic reunion").
Do **NOT** default to "neutral" unless absolutely nothing suggests otherwise.

**CRITICAL INSTRUCTIONS - READ CAREFULLY:**

1.  **JSON STRUCTURE:** You MUST return a valid JSON object with ONLY the top-level key: "chapters".
    * "chapters": An array containing EXACTLY ONE chapter object describing the content within the specified time range.

2.  **EXACT OUTPUT FORMAT:** Use the *actual start and end times* for this segment (`{start_time}s` to `{end_time}s`) in the `start` and `end` fields of your JSON output. The output object MUST NOT contain `shotType` or `shotDescription`.
    ```json
    {{
      "chapters": [
        {{
          "start": "{start_time}s",
          "end": "{end_time}s",
          "sentiment": "neutral",
          "emotions": ["emotion1", "emotion2"],
          "theme": "short theme",
          "summary": "A character receives some news, their expression shifting from neutral to surprised as the conversation continues in the background..."
        }}
      ]
    }}
    ```

3.  **ACCURACY:** Base your entire description *only* on the provided transcription text. DO NOT invent details, characters, settings, or events not directly observable in the input.
"""
    )

    # DEBUGGING PROMPT: Using a simple, open-ended prompt to see if the model can generate ANY text.
    local_system_prompt_vision_chapters: str = (
        """You are a helpful video analysis assistant.
        Based on the provided visual frames and transcription, describe the scene in this video segment.
        Be descriptive and detailed about what you see and what is being said."""
    )

    local_system_prompt_text_chapters: str = (
    """You are an analysis assistant. Your task is to analyze a transcription from a specific time range.

Return a single JSON object with the key "chapters", containing one object with the following fields:
- "start": "{start_time}s"
- "end": "{end_time}s"
- "sentiment": A single word: "positive", "neutral", or "negative".
- "theme": A short, 2-3 word theme.
- "summary": A concise, present-tense summary of the transcription.

Base your analysis ONLY on the text provided.

Example Response:
```json
{{
  "chapters": [
    {{
      "start": "0.000s",
      "end": "30.000s",
      "sentiment": "neutral",
      "theme": "Initial questions",
      "summary": "The conversation begins with one person asking another about their whereabouts on a specific night."
    }}
  ]
}}
```
"""
    )

    results_template_chapters: Dict[str, Any] = {
        "chapters": [
            {
                "start": "0.000s",
                "end": "30.000s",
                "shotType": [],
                "shotDescription": "",
                "sentiment": "neutral",
                "emotions": [],
                "theme": "",
                "summary": ""
            }
        ]
    }


    def __init__(self, video_path: str, env: Optional[CobraEnvironment], output_dir: Optional[str] = None, overwrite_output: bool = False,
                 use_local_llm: bool = False,
                 images_per_segment: int = 4,
                 whisper_model: str = "base",
                 segment_length: float = 30.0,
                 fps: float = 0.2):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if not use_local_llm and env is None:
            raise ValueError("A CobraEnvironment object must be provided for cloud-based analysis.")

        self.video_path = os.path.abspath(video_path)
        self.env = env
        self.images_per_segment = images_per_segment

        self.manifest = self._create_basic_manifest(video_path)
        self.output_dir = prepare_outputs_directory(
            file_name=os.path.basename(video_path),
            segment_length=segment_length,
            frames_per_second=fps,
            output_directory=output_dir,
            overwrite_output=overwrite_output,
            output_directory_prefix="chapters_"
        )
        self.manifest.processing_params.output_directory = self.output_dir

        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "image_tokens": 0, "total_tokens": 0}
        self.prompt_log = []
        self.image_detail_level = "low"

        self.use_local_llm = use_local_llm
        if self.use_local_llm:
            self.local_llm = LocalPhi4Adapter(
                images_per_segment=images_per_segment
            )
            self.local_transcriber = LocalTranscriber(whisper_model)
            log_gpu_memory("After loading local models")
        else:
            self.local_llm = None
            self.local_transcriber = None


    def generate(
        self,
        fps: float = 0.2,
        segment_length: int = 30,
        use_scene_detection: bool = False,
        scene_detection_threshold: float = 30.0,
        downscale_to_max_width: Optional[int] = None,
        downscale_to_max_height: Optional[int] = None,
        enable_language_identification: bool = False,
    ):
        """
        Main method to run the entire chapter generation pipeline.
        """
        logger.info("Starting chapter generation pipeline...")

        # 1. Preprocessing
        self._preprocess_video(
            fps=fps,
            segment_length=segment_length,
            use_scene_detection=use_scene_detection,
            scene_detection_threshold=scene_detection_threshold,
            downscale_to_max_width=downscale_to_max_width,
            downscale_to_max_height=downscale_to_max_height,
            enable_language_identification=enable_language_identification
        )
        logger.info(f"Preprocessing complete. {len(self.manifest.segments)} segments created.")

        # 2. Synchronous Analysis
        logger.info("Starting synchronous analysis of segments for chapters...")
        segment_results = self._analyze_segment_list()
        logger.info("Segment analysis complete.")

        # 3. Aggregation
        final_chapters = self._aggregate_chapters(segment_results)

        # 4. Save results
        self._save_results(final_chapters)

        logger.info(f"Chapter generation complete. Results saved in {self.output_dir}")
        return final_chapters


    def _preprocess_video(self, fps: float, segment_length: int, use_scene_detection: bool, scene_detection_threshold: float, downscale_to_max_width: int, downscale_to_max_height: int, enable_language_identification: bool):
        """
        Handles video preprocessing: segmentation, frame extraction, and transcription.
        """
        self.manifest.processing_params.fps = fps
        self.manifest.processing_params.segment_length = segment_length
        self.manifest.processing_params.use_scene_detection = use_scene_detection

        # Segmentation
        scene_list = None
        if use_scene_detection:
            logger.info(f"Using scene detection with threshold: {scene_detection_threshold}")
            try:
                from scenedetect import detect, ContentDetector, SceneManager
                from scenedetect.video_manager import VideoManager
                video_manager = VideoManager([self.video_path])
                scene_manager = SceneManager()
                scene_manager.add_detector(ContentDetector(threshold=scene_detection_threshold))
                video_manager.set_downscale_factor()
                video_manager.start()
                scene_manager.detect_scenes(frame_source=video_manager)
                scene_list = scene_manager.get_scene_list()
                video_manager.release()

            except ImportError:
                logger.error("PySceneDetect is not installed. Please install it (`pip install scenedetect[opencv]`) to use scene detection.")
                use_scene_detection = False # Fallback
            except Exception as e:
                logger.error(f"Scene detection failed: {e}. Falling back to fixed-length segments.")
                use_scene_detection = False # Fallback

        self._generate_segments(fps, segment_length, scene_list)

        # Audio Transcription
        if self.manifest.source_video.audio_found:
            self._extract_audio_and_transcribe(enable_language_identification)
        else:
            logger.info("No audio stream found, skipping transcription.")

        # Frame extraction for each segment
        logger.info(f"Extracting frames for {len(self.manifest.segments)} segments...")
        for i, segment in enumerate(self.manifest.segments):
            self._preprocess_segment(segment, i, downscale_to_max_width, downscale_to_max_height)

        write_video_manifest(self.manifest)


    def _generate_segments(self, fps, segment_length, scene_list=None):
        duration = self.manifest.source_video.duration
        if not duration:
            raise ValueError("Video duration is unknown, cannot generate segments.")

        if scene_list:
            logger.info(f"Creating segments from {len(scene_list)} detected scenes.")
            for i, (start_time, end_time) in enumerate(scene_list):
                start_sec = start_time.get_seconds()
                end_sec = end_time.get_seconds()
                segment_name = f"scene{i+1}_start{start_sec:.3f}s_end{end_sec:.3f}s"
                seg_obj = self._create_segment_object(segment_name, start_sec, end_sec, end_sec - start_sec, fps)
                self.manifest.segments.append(seg_obj)
        else:
            logger.info(f"Creating fixed-length segments of {segment_length}s.")
            num_segments = int(duration // segment_length)
            for i in range(num_segments):
                start = i * segment_length
                end = start + segment_length
                segment_name = f"seg{i+1}_start{start:.1f}s_end{end:.1f}s"
                seg_obj = self._create_segment_object(segment_name, start, end, segment_length, fps)
                self.manifest.segments.append(seg_obj)

            # Handle remainder
            if duration % segment_length > 1: # Only add if remainder is significant
                start = num_segments * segment_length
                end = duration
                segment_name = f"seg{num_segments+1}_start{start:.1f}s_end{end:.1f}s"
                seg_obj = self._create_segment_object(segment_name, start, end, end - start, fps)
                self.manifest.segments.append(seg_obj)

    def _create_segment_object(self, base_name: str, start_time: float, end_time: float, segment_duration: float, analysis_fps: float):
        segment_path = os.path.join(self.output_dir, base_name)
        references_dir = os.path.join(segment_path, "references")
        os.makedirs(references_dir, exist_ok=True)
        return Segment(
            id=base_name,
            segment_name=base_name,
            start_time=start_time,
            end_time=end_time,
            duration=segment_duration,
            segment_path=segment_path,
            total_frames=int(segment_duration * analysis_fps),
            analyzed_result={},
            segment_frames_file_path=[os.path.join(references_dir, f) for f in sorted(os.listdir(references_dir))] if os.path.exists(references_dir) else []
        )

    def _preprocess_segment(self, segment: Segment, index: int, downscale_to_max_width=None, downscale_to_max_height=None):
        references_dir = os.path.join(segment.segment_path, "references")
        segment_and_extract(
            segment.start_time,
            segment.end_time,
            self.video_path,
            segment.segment_path,
            references_dir,
            self.manifest.processing_params.fps,
            downscale_to_max_width,
            downscale_to_max_height
        )

        frame_files = sorted(os.listdir(references_dir))
        segment.segment_frames_file_path = [os.path.join(references_dir, f) for f in frame_files]

        fps = self.manifest.processing_params.fps
        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(references_dir, frame_file)
            timestamp = segment.start_time + (i / fps)
            segment.frames.append(Frame(frame_path=frame_path, timestamp=timestamp))

        # Assign transcription to segment
        if self.manifest.audio_transcription:
            if self.use_local_llm:
                segment.transcription = self._slice_text_from_segments(
                    self.manifest.audio_transcription,
                    segment.start_time,
                    segment.end_time
                )
            else:
                from .cobra_utils import parse_transcript
                segment.transcription = parse_transcript(self.manifest.audio_transcription, segment.start_time, segment.end_time)

    def _slice_text_from_segments(self, segments: List[Dict], start: float, end: float) -> str:
        """Extracts transcription text for a specific time range from Whisper segments."""
        return "".join(
            s["text"] for s in segments
            if max(start, s["start"]) < min(end, s["end"])
        ).strip()

    def _extract_audio_and_transcribe(self, enable_language_identification: bool):
        """Extracts audio and generates a transcription for the entire video."""
        audio_file_name = f"{os.path.splitext(self.manifest.name)[0]}_full_audio.wav"
        audio_file_path = os.path.join(self.output_dir, audio_file_name)

        try:
            logger.info(f"Extracting full audio to {audio_file_path}...")
            extract_base_audio(self.video_path, audio_file_path)
            self.manifest.source_video.audio_path = audio_file_path
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            self.manifest.source_video.audio_found = False
            return

        if self.use_local_llm:
            logger.info("Starting local transcription using Whisper...")
            try:
                transcript_segments = self.local_transcriber.transcribe(audio_file_path)
                self.manifest.audio_transcription = transcript_segments
                transcript_path = os.path.join(self.output_dir, "full_transcription.json")
                with open(transcript_path, "w", encoding='utf-8') as f:
                    json.dump(transcript_segments, f, indent=4, ensure_ascii=False)
                logger.info(f"Full local transcription saved to {transcript_path}")
                return
            except Exception as e:
                logger.error(f"Local transcription failed: {e}")
                return

        logger.info("Uploading audio to Azure Blob Storage for transcription...")
        blob_name = f"audio-for-transcription/{datetime.utcnow().strftime('%Y%m%d')}/{os.path.basename(audio_file_path)}"

        audio_sas_url = upload_blob(audio_file_path, blob_name, self.env)

        if not audio_sas_url:
            logger.error("Failed to upload audio to blob storage. Cannot generate transcript.")
            return

        logger.info("Submitting audio for batch transcription...")
        transcript = generate_batch_transcript(
            audio_sas_url,
            self.env,
            enable_language_identification=enable_language_identification
        )

        if transcript:
            logger.info("Transcription received successfully.")
            self.manifest.audio_transcription = transcript
            transcript_path = os.path.join(self.output_dir, "full_transcription.json")
            with open(transcript_path, "w", encoding='utf-8') as f:
                json.dump(transcript, f, indent=4, ensure_ascii=False)
            logger.info(f"Full transcription saved to {transcript_path}")
        else:
            logger.error("Transcription failed.")

    def _analyze_segment_list(self):
        """Analyzes all segments synchronously."""
        results_list = []
        for segment in self.manifest.segments:
            parsed_response = self._analyze_segment_for_chapter(segment)
            if parsed_response:
                results_list.append(parsed_response)

            if self.use_local_llm:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log_gpu_memory(f"After cleanup for segment {segment.segment_name}")
        return results_list

    def _analyze_segment_for_chapter(self, segment: Segment) -> Optional[Dict]:
        """Analyzes a single segment to generate a chapter description."""
        if self.use_local_llm:
            prompt_messages = self._generate_segment_prompt(segment)
            if not prompt_messages:
                return None
            try:
                log_gpu_memory(f"Before LLM call for segment {segment.segment_name}")
                # The adapter now handles the full generation and parsing process.
                response_data = self.local_llm.generate(prompt_messages)
                log_gpu_memory(f"After LLM call for segment {segment.segment_name}")

                if not response_data or "error" in response_data:
                    logger.error(f"Failed to get valid response from local LLM for segment {segment.segment_name}. Response: {response_data}")
                    # If there's an error, let's create a placeholder to avoid crashing
                    # and to see how many segments fail.
                    return {
                        "start": f"{segment.start_time:.3f}s",
                        "end": f"{segment.end_time:.3f}s",
                        "summary": f"Error: LLM failed to generate content. Details: {response_data.get('error', 'Unknown error')}",
                        "theme": "Generation Error",
                        "error": True
                    }
                
                # The adapter should return a dict. If it has 'chapters', use that.
                if "chapters" in response_data and isinstance(response_data["chapters"], list) and response_data["chapters"]:
                    return response_data["chapters"][0]
                else:
                    # Fallback for unexpected (but valid) JSON from the LLM
                    logger.warning(f"LLM response for {segment.segment_name} did not contain 'chapters' key. Using raw response.")
                    # We can create a partial result from the raw dict.
                    return {
                        "start": f"{segment.start_time:.3f}s",
                        "end": f"{segment.end_time:.3f}s",
                        "summary": str(response_data),
                        "theme": "Raw Response"
                    }

            except Exception as e:
                logger.error(f"Local LLM call failed on {segment.segment_name}: {e}", exc_info=True)
                return None

        # This is the original cloud-based path
        logger.info(f"Analyzing segment for chapter: {segment.segment_name}")

        prompt_messages = self._generate_segment_prompt(segment)
        if not prompt_messages:
            return None

        try:
            llm_response = self._call_llm(prompt_messages, log_token_category="chapter_analysis")
            response_content = llm_response.choices[0].message.content

            parsed_llm_output = self._parse_llm_json_response(response_content)

            if not parsed_llm_output or "error" in parsed_llm_output:
                logger.error(f"Failed to parse LLM response for segment {segment.segment_name}")
                return None

            if "chapters" in parsed_llm_output and isinstance(parsed_llm_output["chapters"], list) and parsed_llm_output["chapters"]:
                return parsed_llm_output["chapters"][0]
            else:
                logger.warning(f"LLM response for {segment.segment_name} did not contain the expected chapter structure.")
                return None

        except Exception as e:
            logger.error(f"LLM call failed for segment {segment.segment_name}: {e}")
            return None

    def _generate_segment_prompt(self, segment: Segment) -> Optional[List[Dict]]:
        """Generates the prompt for a single segment analysis."""
        # --- BUG FIX: Robust frame selection logic ---
        available_frames = segment.frames
        num_available = len(available_frames)
        num_to_select = self.images_per_segment

        selected_frames = []
        if num_available > 0:
            if num_to_select >= num_available:
                # If we want more or equal frames than are available, take all of them
                selected_frames = available_frames
            elif num_to_select == 1:
                # If we specifically want one, take the middle one for better representation
                selected_frames = [available_frames[num_available // 2]]
            else:
                # This logic now only runs when num_to_select > 1 and < num_available
                # The denominator (num_to_select - 1) is guaranteed to be non-zero
                indices = sorted(list(set([int(i * (num_available - 1) / (num_to_select - 1)) for i in range(num_to_select)])))
                selected_frames = [available_frames[i] for i in indices]

        logger.info(f"Selected {len(selected_frames)} out of {num_available} available frames for segment {segment.segment_name}.")

        is_vision_analysis = len(selected_frames) > 0

        system_prompt_template = ""
        if self.use_local_llm:
            # For local LLM, use the simpler, more direct prompts.
            if is_vision_analysis:
                system_prompt_template = self.local_system_prompt_vision_chapters
            else:
                system_prompt_template = self.local_system_prompt_text_chapters
        else:
            # For cloud LLM, we use the detailed vision prompt or the detailed text prompt
            if is_vision_analysis:
                system_prompt_template = self.system_prompt_vision_chapters
            else:
                system_prompt_template = self.system_prompt_text_chapters
        
        # Format the selected prompt
        if is_vision_analysis and not self.use_local_llm:
            shot_types_str = ", ".join([f'"{s}"' for s in self.SHOT_TYPES])
            system_prompt = system_prompt_template.format(
                shot_types_list=shot_types_str,
                start_time=f"{segment.start_time:.3f}",
                end_time=f"{segment.end_time:.3f}"
            )
        else:
            system_prompt = system_prompt_template.format(
                start_time=f"{segment.start_time:.3f}",
                end_time=f"{segment.end_time:.3f}"
            )

        messages = [{"role": "system", "content": system_prompt}]
        user_content_parts = []
        transcription_context = segment.transcription if segment.transcription else "No transcription available."
        
        user_text = (
            f"Context from {segment.start_time:.3f}s–{segment.end_time:.3f}s:\n\n"
            f"{transcription_context}"
        )

        if is_vision_analysis:
            user_text += "\n\nVisual references:"

        user_content_parts.append({"type": "text", "text": user_text})

        for frame in selected_frames:
            try:
                user_content_parts.append({
                    "type": "image",
                    "image": frame.frame_path
                })
            except Exception as e:
                logger.error(f"Error adding frame {frame.frame_path} to prompt: {e}")

        # If we requested images but couldn't add any, abort.
        # This is different from the case where images_per_segment is 0.
        if self.images_per_segment > 0 and not is_vision_analysis:
            logger.warning(
                f"Requested {self.images_per_segment} images but none were available/added for "
                f"{segment.segment_name}; skipping analysis for this segment."
            )
            return None

        messages.append({"role": "user", "content": user_content_parts})
        return messages

    def _call_llm(self, messages, model="gpt-4o", log_token_category=None):
        if self.use_local_llm:
            raise RuntimeError("_call_llm called while use_local_llm=True")
        self._log_prompt(log_token_category or "unknown", {}, messages)

        client = openai.AzureOpenAI(
            api_key=self.env.vision.api_key.get_secret_value(),
            api_version=self.env.vision.api_version,
            azure_endpoint=self.env.vision.endpoint,
        )

        response = client.chat.completions.create(
            model=self.env.vision.deployment,
            messages=messages,
            max_tokens=2048
        )

        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        image_tokens = estimate_image_tokens(messages)

        if not self.use_local_llm:
            self.token_usage["prompt_tokens"] += prompt_tokens
            self.token_usage["completion_tokens"] += completion_tokens
            self.token_usage["image_tokens"] += image_tokens
            self.token_usage["total_tokens"] += prompt_tokens + completion_tokens

        logger.debug(f"[TOKENS] {log_token_category}: P={prompt_tokens}, C={completion_tokens}, I(est)={image_tokens}")
        return response

    def _parse_llm_json_response(self, raw_content_str: str):
        try:
            code_to_parse = raw_content_str.strip()
            if code_to_parse.startswith("```json"):
                code_to_parse = code_to_parse.split("```json", 1)[1].strip()
            if code_to_parse.endswith("```"):
                code_to_parse = code_to_parse.rsplit("```", 1)[0].strip()

            return json.loads(code_to_parse)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON: {e}\nContent: {raw_content_str[:500]}...")
            return {"error": f"JSONDecodeError: {e}"}

    def _aggregate_chapters(self, segment_results: List[Dict]) -> Dict:
        """Aggregates chapter results from all segments."""
        valid_results = [res for res in segment_results if res is not None]
        final_result = {
            "chapters": sorted(valid_results, key=lambda x: float(str(x.get("start", "inf")).rstrip("s")))
        }
        return final_result

    def _save_results(self, final_chapters: Dict):
        """Saves the final chapter JSON and logs."""
        chapters_path = os.path.join(self.output_dir, "chapters.json")
        with open(chapters_path, "w", encoding="utf-8") as f:
            json.dump(final_chapters, f, indent=4, ensure_ascii=False)

        if not self.use_local_llm:
            prompt_log_path = os.path.join(self.output_dir, "run_prompts.md")
            with open(prompt_log_path, "w", encoding="utf-8") as f:
                f.write(f"# LLM Prompts Log\n\n")
                for i, entry in enumerate(self.prompt_log):
                    f.write(f"## Prompt {i+1}: {entry['type']}\n")
                    f.write("```json\n")
                    f.write(json.dumps(entry['messages'], indent=2))
                    f.write("\n```\n\n")

        write_video_manifest(self.manifest)

    def _log_prompt(self, prompt_type, context, messages):
        self.prompt_log.append({
            "type": prompt_type,
            "context": context,
            "messages": messages
        })

    def _create_basic_manifest(self, video_path: str) -> VideoManifest:
        """Creates a basic VideoManifest object from a video file path."""
        manifest = VideoManifest()
        manifest.name = os.path.basename(video_path)
        manifest.source_video.path = os.path.abspath(video_path)

        file_metadata = get_file_info(video_path)
        if file_metadata:
            duration = file_metadata.get("duration", 0.0)
            manifest.source_video.duration = duration
            manifest.source_video.audio_found = file_metadata.get("audio_found", False)
            manifest.processing_params.run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return manifest