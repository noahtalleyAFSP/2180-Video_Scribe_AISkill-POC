import os
import time
import math
import numpy as np
import psutil
from typing import Union, Type

import concurrent.futures

from .models.video import VideoManifest, Segment
from .models.environment import CobraEnvironment
from .cobra_utils import (
    get_elapsed_time,
    generate_transcript,
    parse_transcript,
    get_elapsed_time,
    write_video_manifest,
    extract_base_audio,
    segment_and_extract,
    parallelize_audio,
    parallelize_transcription,
    prepare_outputs_directory,
)


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
            print(f"({get_elapsed_time(start_time)}s) Extracting audio...")
            self._extract_audio(max_workers)
        
        # Generate the segments based on speech or time
        print(f"({get_elapsed_time(start_time)}s) Generating segments...")
        if use_speech_based_segments and self.manifest.audio_transcription:
            self._generate_speech_based_segments()
        else:
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
        print(f"({get_elapsed_time(start_time)}s) Processing segments...")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = []
            # Submit the segments as tasks to the executor
            for i, segment in enumerate(self.manifest.segments):
                # Skip segments that have already been processed
                if segment.processed:
                    continue
                futures.append(
                    executor.submit(self._preprocess_segment,
                                    segment=segment, index=i)
                )

            # As tasks are completed, update the video manifest
            for future in concurrent.futures.as_completed(futures):
                i, updated_segment, res = future.result()
                self.manifest.segments[i] = updated_segment
                self.manifest.segments[i].processed = res

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

    def _preprocess_segment(self, segment: Segment, index: int):
        stop_watch_time = time.time()

        create_transcript_flag = (
            self.manifest.processing_params.generate_transcript_flag
        )

        print(
            f"**Segment {index} {segment.segment_name} - beginning processing. Transcripts: {create_transcript_flag}"
        )

        try:
            input_video_path = self.manifest.source_video.path
            segment_path = segment.segment_folder_path
            start_time = segment.start_time
            end_time = segment.end_time
            fps = self.manifest.processing_params.fps  # Desired analysis FPS

            frames_dir = os.path.join(segment_path, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            segment_video_path = os.path.join(segment_path, "segment.mp4")
            segment_and_extract(
                start_time, end_time, input_video_path, segment_path, frames_dir, fps
            )

            # Calculate frame times based on actual extracted frames
            frame_files = sorted(os.listdir(frames_dir))
            number_of_frames = len(frame_files)
            
            # Recalculate intervals based on actual frame count
            segment_duration = end_time - start_time
            frame_times = np.linspace(start_time, end_time, number_of_frames, endpoint=False)
            frame_times = [round(t, 2) for t in frame_times]
            
            segment.segment_frame_time_intervals = frame_times
            segment.segment_frames_file_path = []
            
            # Rename frames with matching intervals
            for i, (frame_file, frame_time) in enumerate(zip(frame_files, frame_times)):
                old_frame_path = os.path.join(frames_dir, frame_file)
                new_frame_filename = f"frame_{i}_{frame_time}s.jpg"
                new_frame_path = os.path.join(frames_dir, new_frame_filename)
                os.rename(old_frame_path, new_frame_path)
                segment.segment_frames_file_path.append(new_frame_path)

            # Remove the temporary segment video file
            os.remove(segment_video_path)

            print(
                f"**Segment {index} {segment.segment_name} - extracted and renamed frames in {get_elapsed_time(stop_watch_time)}"
            )

            # Process transcription if needed
            if create_transcript_flag and not segment.transcription:
                # Only extract transcription if it wasn't already provided
                # (which happens with speech-based segments)
                transcript = self.manifest.audio_transcription
                segment.transcription = parse_transcript(
                    transcript, start_time, end_time
                )

            # Update segment frame count
            segment.number_of_frames = len(segment.segment_frames_file_path)

            return index, segment, True
        except Exception as e:
            print(f"Error processing segment {segment.segment_name}: {e}")
            return index, segment, False

    # Define the audio output path
    def _extract_audio(self, max_workers: int):
        audio_path = os.path.join(
            self.manifest.processing_params.output_directory,
            f"{os.path.splitext(self.manifest.name)[0]}.mp3",
        )

        # Use FFmpeg to extract audio
        extract_base_audio(self.manifest.source_video.path, audio_path)

        audio_file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)

        # Process audio based on file size
        if audio_file_size_mb <= 25.0:
            # For small audio files, process directly
            self.manifest.source_audio.path = audio_path
            self.manifest.source_audio.file_size_mb = audio_file_size_mb
            transcript = generate_transcript(
                audio_file_path=audio_path, env=self.env)
            self.manifest.audio_transcription = transcript
        else:
            # For large audio files, split into chunks and process in parallel
            print(
                f"Audio file size is {audio_file_size_mb:.2f}MB; splitting into chunks..."
            )

            # Calculate number of chunks
            splitting_value = int(audio_file_size_mb / 20)
            duration = float(self.manifest.source_video.duration)
            chunk_size = duration / splitting_value

            # Prepare arguments for parallel extraction
            extract_args_list = []
            for counter in range(splitting_value):
                start = chunk_size * counter
                end = min(chunk_size * (counter + 1), duration)
                audio_chunk_path = os.path.join(
                    self.manifest.processing_params.output_directory,
                    f"{os.path.splitext(self.manifest.name)[0]}_{counter + 1}.mp3",
                )
                extract_args_list.append(
                    (self.manifest.source_video.path, start, end, audio_chunk_path)
                )
            # Parallelize audio chunk extraction
            extracted_chunks = parallelize_audio(
                extract_args_list, max_workers)

            # Prepare arguments for parallel transcription
            process_args_list = [
                (chunk_path, start) for chunk_path, start in extracted_chunks
            ]
            combined_transcript = parallelize_transcription(process_args_list)

            self.manifest.source_audio.path = audio_path
            self.manifest.source_audio.file_size_mb = audio_file_size_mb
            self.manifest.audio_transcription = combined_transcript

    def _generate_speech_based_segments(self):
        """Generate segments based on natural speech breaks in the transcript."""
        print("Generating speech-based segments...")
        
        # Check if we have a transcript
        if not hasattr(self.manifest, 'audio_transcription') or not self.manifest.audio_transcription:
            print("No transcription available, falling back to time-based segments")
            return self._generate_segments()
        
        # Extract speech segments from transcript
        segments_data = self.manifest.audio_transcription.get("segments", [])
        
        if not segments_data:
            print("No speech segments found, falling back to time-based segments")
            return self._generate_segments()
        
        print(f"Found {len(segments_data)} speech segments")
        
        # Group speech segments into logical groups
        grouped_segments = []
        current_group = []
        current_duration = 0
        max_duration = min(60, self.manifest.processing_params.segment_length * 3)  # Cap at 60s or 3x segment_length
        min_duration = max(5, self.manifest.processing_params.segment_length / 2)    # Min 5s or half segment_length
        
        # Sort segments by start time
        sorted_segments = sorted(segments_data, key=lambda x: x.get("offset", 0))
        
        for speech in sorted_segments:
            speech_start = speech.get("offset", 0)
            speech_duration = speech.get("duration", 0)
            
            # If this would make the segment too long, start a new one
            if current_duration + speech_duration > max_duration and current_duration >= min_duration:
                if current_group:
                    grouped_segments.append(current_group)
                current_group = [speech]
                current_duration = speech_duration
            else:
                current_group.append(speech)
                current_duration += speech_duration
                
            # If there's a significant pause after this speech (> 2 seconds)
            # and we have enough content, start a new segment
            if len(sorted_segments) > 1:
                idx = sorted_segments.index(speech)
                if idx < len(sorted_segments) - 1:
                    next_speech = sorted_segments[idx + 1]
                    next_start = next_speech.get("offset", 0)
                    pause_duration = next_start - (speech_start + speech_duration)
                    
                    if pause_duration > 2 and current_duration >= min_duration:
                        grouped_segments.append(current_group)
                        current_group = []
                        current_duration = 0
        
        # Add the last group if it's not empty
        if current_group:
            grouped_segments.append(current_group)
        
        # Create segments from the groups
        segments = []
        for i, group in enumerate(grouped_segments):
            if not group:
                continue
            
            # Calculate start and end times
            start_time = min(speech.get("offset", 0) for speech in group)
            end_time = max(speech.get("offset", 0) + speech.get("duration", 0) for speech in group)
            
            # Add small buffer at beginning and end
            buffer = 0.5  # half second buffer
            start_time = max(0, start_time - buffer)
            end_time = min(self.manifest.source_video.duration, end_time + buffer)
            
            segment_duration = end_time - start_time
            
            # Create and set up the segment
            segment = Segment()
            segment.segment_name = f"segment_{i+1:04d}"
            segment.segment_folder_path = os.path.join(
                self.manifest.processing_params.output_directory, segment.segment_name
            )
            segment.start_time = start_time
            segment.end_time = end_time
            segment.segment_duration = segment_duration
            
            # Create segment folder
            os.makedirs(segment.segment_folder_path, exist_ok=True)
            
            # Collect transcription for this segment
            segment_text = " ".join(speech.get("text", "") for speech in group)
            segment.transcription = segment_text
            
            # Mark the segment as speech based
            segment.is_speech_based = True
            
            segments.append(segment)
        
        # If no valid groups were created, fall back to time-based segments
        if not segments:
            print("Could not create valid speech-based segments, falling back to time-based segments")
            return self._generate_segments()
        
        print(f"Created {len(segments)} speech-based segments")
        
        # Update the manifest with the new segments
        self.manifest.segments = segments
        self.manifest.segment_metadata.num_segments = len(segments)
        self.manifest.segment_metadata.effective_duration = sum(
            segment.segment_duration for segment in segments
        )
        
        return segments
