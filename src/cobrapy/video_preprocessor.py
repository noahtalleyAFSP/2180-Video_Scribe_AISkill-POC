import os
import time
import math
import numpy as np
import psutil
from typing import Union, Type, Optional

import concurrent.futures
import requests
from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas
from datetime import datetime, timedelta

from .models.video import VideoManifest, Segment
from .models.environment import CobraEnvironment
from .cobra_utils import (
    get_elapsed_time,
    parse_transcript,
    write_video_manifest,
    extract_base_audio,
    segment_and_extract,
    parallelize_audio,
    parallelize_transcription,
    prepare_outputs_directory,
)
from .cobra_utils import generate_batch_transcript


class VideoPreProcessor:
    # take either a video manifest object or a path to a video manifest file
    def __init__(
        self, video_manifest: Union[str, VideoManifest], env: CobraEnvironment
    ):
        self.manifest = video_manifest
        self.env = env

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
        use_speech_based_segments=False  # New parameter to enable speech-based segments
    ) -> str:
        start_time = time.time()
        print(
            f"({get_elapsed_time(start_time)}s) Preprocessing video {self.manifest.name}"
        )

        # Validate video manifest
        if not isinstance(self.manifest, VideoManifest):
            raise ValueError(
                "Video manifest is not defined. Be sure you have initialized the VideoClient object with a valid video_path or manifest parameter."
            )

        # Validate processing parameters
        if fps is None or fps <= 0:
            raise ValueError("'fps' must be a positive number")

        if segment_length is None or segment_length <= 0:
            raise ValueError("'segment_length' must be a positive number")
        elif segment_length > self.manifest.source_video.duration:
            print(
                "Segment length is longer than the video duration. Setting the segment length to the video duration."
            )
            segment_length = self.manifest.source_video.duration

        # Set processing parameters
        print(f"({get_elapsed_time(start_time)}) Setting processing parameters...")
        self.manifest.processing_params.fps = fps
        self.manifest.processing_params.segment_length = segment_length
        self.manifest.processing_params.use_speech_based_segments = use_speech_based_segments  # Save this setting
        
        if self.manifest.source_video.audio_found is False:
            self.manifest.processing_params.generate_transcript_flag = False
            if use_speech_based_segments:
                print("Warning: No audio found in video, falling back to time-based segmentation")
                self.manifest.processing_params.use_speech_based_segments = False
        else:
            self.manifest.processing_params.generate_transcript_flag = generate_transcripts_flag

        self.manifest.processing_params.trim_to_nearest_second = trim_to_nearest_second
        self.manifest.processing_params.allow_partial_segments = allow_partial_segments

        # Prepare the output directory
        print(f"({get_elapsed_time(start_time)}s) Preparing output directory")

        if output_directory is not None:
            self.manifest.processing_params.output_directory = (
                prepare_outputs_directory(
                    file_name=self.manifest.name,
                    output_directory=output_directory,
                    frames_per_second=fps,
                    segment_length=segment_length,
                    overwrite_output=overwrite_output,
                )
            )
        else:
            self.manifest.processing_params.output_directory = (
                prepare_outputs_directory(
                    file_name=self.manifest.name,
                    segment_length=segment_length,
                    frames_per_second=fps,
                    overwrite_output=overwrite_output,
                )
            )

        # Extract the audio using FFmpeg first if needed for speech-based segmentation
        if (
            self.manifest.source_video.audio_found
            and self.manifest.processing_params.generate_transcript_flag
        ):
            print(f"({get_elapsed_time(start_time)}s) Extracting audio and initiating Batch Transcription...")
            try:
                # _extract_audio_and_transcribe will now handle upload and wait for Batch API
                self._extract_audio_and_transcribe(max_workers)
                if not self.manifest.audio_transcription:
                     print(f"({get_elapsed_time(start_time)}s) Warning: Batch transcription did not produce results.")
                else:
                     print(f"({get_elapsed_time(start_time)}s) Batch transcription completed.")
            except Exception as e:
                print(f"({get_elapsed_time(start_time)}s) ERROR during audio extraction or transcription: {e}")
                # Decide: stop or continue without transcription?
                print(f"({get_elapsed_time(start_time)}s) Warning: Continuing preprocessing without transcription.")
                self.manifest.processing_params.generate_transcript_flag = False # Ensure flag is off

        # Generate the segments based on speech or time
        print(f"({get_elapsed_time(start_time)}s) Generating time-based segments...")
        self._generate_segments()
        
        # Configure thread pool
        if max_workers is None:
            # Number of physical cores
            cpu_count = psutil.cpu_count(logical=False) or 1
            memory = psutil.virtual_memory().total / (1024**3)  # Total memory in GB
            max_workers = min(cpu_count, int(memory // 2))
        else:
            max_workers = max_workers

        # Process the segments
        print(f"({get_elapsed_time(start_time)}s) Processing segments (extracting frames)...")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = []
            # Submit the segments as tasks to the executor
            for i, segment in enumerate(self.manifest.segments):
                # Skip segments that have already been processed
                if segment.processed:
                    continue
                # Pass the full transcription object if available for parsing segment text
                segment_transcription_text = None
                if self.manifest.audio_transcription:
                     # Parse relevant text for this segment's time window
                     segment_transcription_text = parse_transcript(
                          self.manifest.audio_transcription, # Pass the Batch API result dict
                          segment.start_time,
                          segment.end_time
                     )

                futures.append(
                     executor.submit(self._preprocess_segment,
                                      segment=segment,
                                      index=i,
                                      transcription_text=segment_transcription_text # Pass parsed text
                                      )
                )

            # As tasks are completed, update the video manifest
            for future in concurrent.futures.as_completed(futures):
                i, updated_segment, res = future.result()
                self.manifest.segments[i] = updated_segment
                # No need to set .processed=res here, _preprocess_segment should handle state

        print(f"({get_elapsed_time(start_time)}s) All segments pre-processed")

        # Check to make sure the frame intervals in the manifest and the frame file paths in the manfest match.
        # Not sure why there is a possibility of a mismatch, but it has definitely been observed in testing.
        for segment in self.manifest.segments:
            frame_file_paths = segment.segment_frames_file_path
            frame_intervals = segment.segment_frame_time_intervals
            if len(frame_file_paths) != len(frame_intervals):
                print(
                    f"Segment {segment.segment_name}: Frame file paths and frame intervals do not match. Adjusting..."
                )
                min_list_length = min(
                    len(frame_file_paths), len(frame_intervals))
                segment.segment_frames_file_path = frame_file_paths[:min_list_length]
                segment.segment_frame_time_intervals = frame_intervals[:min_list_length]

        write_video_manifest(self.manifest)

        return self.manifest.video_manifest_path

    def _generate_segments(self):
        video_duration = self.manifest.source_video.duration
        segment_length = self.manifest.processing_params.segment_length
        analysis_fps = self.manifest.processing_params.fps
        trim_to_nearest_second = self.manifest.processing_params.trim_to_nearest_second
        allow_partial_segments = self.manifest.processing_params.allow_partial_segments

        # Calculate the effective duration and number of segments to create
        if trim_to_nearest_second:
            effective_duration = math.floor(video_duration)
        else:
            effective_duration = video_duration

        if allow_partial_segments:
            num_segments = math.ceil(effective_duration / segment_length)
        else:
            num_segments = math.floor(effective_duration / segment_length)

        num_segments = int(num_segments)

        self.manifest.segment_metadata = self.manifest.segment_metadata.model_copy(
            update={
                "effective_duration": effective_duration,
                "num_segments": num_segments,
            }
        )

        # Define each segment and add to the video manifest
        for i in range(num_segments):
            start_time = i * segment_length
            end_time = min((i + 1) * segment_length, effective_duration)

            # Determine how many frames should be in the segment and what time they would be at.
            segment_duration = end_time - start_time

            number_of_frames_in_segment = math.ceil(
                segment_duration * analysis_fps)

            segment_frames_times = np.linspace(
                start_time, end_time, number_of_frames_in_segment, endpoint=False
            )

            segment_frames_times = [round(x, 2) for x in segment_frames_times]

            # Create a segment name and folder path
            segment_name = f"seg{i+1}_start{start_time}s_end{end_time}s"
            output_directory = self.manifest.processing_params.output_directory
            segment_folder_path = os.path.join(output_directory, segment_name)

            os.makedirs(segment_folder_path, exist_ok=True)

            self.manifest.segments.append(
                Segment(
                    segment_name=segment_name,
                    segment_folder_path=segment_folder_path,
                    start_time=start_time,
                    end_time=end_time,
                    segment_duration=segment_duration,
                    number_of_frames=number_of_frames_in_segment,
                    segment_frame_time_intervals=segment_frames_times,
                    processed=False,
                )
            )

    def _preprocess_segment(self, segment: Segment, index: int, transcription_text: Optional[str]):
        stop_watch_time = time.time()
        print(
            f"**Segment {index} {segment.segment_name} - beginning frame extraction."
        )
        try:
            input_video_path = self.manifest.source_video.path
            segment_path = segment.segment_folder_path
            start_time = segment.start_time
            end_time = segment.end_time
            fps = self.manifest.processing_params.fps

            frames_dir = os.path.join(segment_path, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            segment_video_path = os.path.join(segment_path, "segment.mp4")
            segment_and_extract(start_time, end_time, input_video_path, segment_path, frames_dir, fps)

            frame_files = sorted(os.listdir(frames_dir))
            number_of_frames = len(frame_files)

            segment_duration = end_time - start_time
            frame_times = np.linspace(start_time, end_time, number_of_frames, endpoint=False)
            frame_times = [round(t, 3) for t in frame_times] # Keep 3 decimals for consistency

            segment.segment_frame_time_intervals = frame_times
            segment.segment_frames_file_path = []

            for i, (frame_file, frame_time) in enumerate(zip(frame_files, frame_times)):
                old_frame_path = os.path.join(frames_dir, frame_file)
                # Use 3 decimal places in filename
                new_frame_filename = f"frame_{i}_{frame_time:.3f}s.jpg"
                new_frame_path = os.path.join(frames_dir, new_frame_filename)
                os.rename(old_frame_path, new_frame_path)
                segment.segment_frames_file_path.append(new_frame_path)

            os.remove(segment_video_path)
            print(f"**Segment {index} {segment.segment_name} - extracted {number_of_frames} frames in {get_elapsed_time(stop_watch_time)}s")

            # Set the pre-parsed transcription text
            segment.transcription = transcription_text if transcription_text else "No transcription for this segment."

            segment.number_of_frames = len(segment.segment_frames_file_path)
            segment.processed = True # Mark as processed (frames extracted)
            return index, segment, True # Return success status

        except Exception as e:
            print(f"Error processing segment {segment.segment_name} frames: {e}")
            segment.processed = False # Mark as failed
            return index, segment, False # Return failure status

    # Rename to reflect combined action
    def _extract_audio_and_transcribe(self, max_workers: int):
        """Extracts audio, uploads to blob, runs Batch Transcription, and waits for results."""
        start_t = time.time()
        base_audio_name = f"{os.path.splitext(self.manifest.name)[0]}.wav"
        local_audio_path = os.path.join(
            self.manifest.processing_params.output_directory,
            base_audio_name
        )

        # --- 1. Extract Full Audio Locally (calls the updated extract_base_audio) ---
        print(f"({get_elapsed_time(start_t)}s) Extracting full audio to {local_audio_path} (Format: WAV)...")
        try:
             extract_base_audio(self.manifest.source_video.path, local_audio_path)
             self.manifest.source_audio.path = local_audio_path
             self.manifest.source_audio.file_size_mb = os.path.getsize(local_audio_path) / (1024 * 1024)
             print(f"({get_elapsed_time(start_t)}s) Audio extracted ({self.manifest.source_audio.file_size_mb:.2f} MB).")
        except Exception as e:
             print(f"({get_elapsed_time(start_t)}s) Failed to extract audio: {e}")
             raise # Re-raise exception to stop processing

        # --- 2. Upload Audio to Azure Blob Storage ---
        print(f"({get_elapsed_time(start_t)}s) Uploading audio to Azure Blob Storage...")
        blob_service_client = None
        blob_config = self.env.blob_storage
        blob_name = f"audio_uploads/{base_audio_name}"

        try:
            if blob_config.connection_string:
                 blob_service_client = BlobServiceClient.from_connection_string(
                      blob_config.connection_string.get_secret_value()
                 )
            elif blob_config.account_name and blob_config.sas_token:
                 account_url = f"https://{blob_config.account_name}.blob.core.windows.net"
                 blob_service_client = BlobServiceClient(
                      account_url=account_url,
                      credential=blob_config.sas_token.get_secret_value()
                 )
            else:
                 raise ValueError("Missing Blob Storage credentials (Connection String or SAS Token).")

            blob_client = blob_service_client.get_blob_client(
                 container=blob_config.container_name,
                 blob=blob_name
            )

            with open(local_audio_path, "rb") as data:
                 blob_client.upload_blob(data, overwrite=True)
            print(f"({get_elapsed_time(start_t)}s) Audio uploaded to {blob_config.container_name}/{blob_name}")

            # --- 3. Generate SAS Token for the uploaded blob ---
            print(f"({get_elapsed_time(start_t)}s) Generating SAS token for transcription...")
            sas_token = generate_blob_sas(
                 account_name=blob_config.account_name,
                 container_name=blob_config.container_name,
                 blob_name=blob_name,
                 account_key=blob_service_client.credential.account_key if hasattr(blob_service_client.credential, 'account_key') else None,
                 user_delegation_key=None,
                 permission=BlobSasPermissions(read=True),
                 expiry=datetime.utcnow() + timedelta(hours=24)
            )
            blob_sas_url = f"{blob_client.url}?{sas_token}"
            print(f"({get_elapsed_time(start_t)}s) SAS URL generated.")

        except Exception as e:
             print(f"({get_elapsed_time(start_t)}s) Failed to upload audio or generate SAS: {e}")
             raise

        # --- 4. Call Batch Transcription (and wait) ---
        print(f"({get_elapsed_time(start_t)}s) Submitting Batch Transcription job...")
        try:
             transcript_result = generate_batch_transcript(
                  audio_blob_sas_url=blob_sas_url,
                  env=self.env,
                  candidate_locales=["en-US"]
             )
             self.manifest.audio_transcription = transcript_result
             print(f"({get_elapsed_time(start_t)}s) Batch transcription job finished.")
        except Exception as e:
             print(f"({get_elapsed_time(start_t)}s) Batch transcription call failed: {e}")
             self.manifest.audio_transcription = None

        # --- 5. Optional: Clean up local audio ---
        try:
            print(f"({get_elapsed_time(start_t)}s) Removing local audio file: {local_audio_path}")
            os.remove(local_audio_path)
        except Exception as e:
            print(f"({get_elapsed_time(start_t)}s) Warning: Could not remove local audio file {local_audio_path}: {e}")
