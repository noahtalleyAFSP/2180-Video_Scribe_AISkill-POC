import os
import json
import time
import asyncio
import nest_asyncio
from typing import Union, Type, Optional, List, Dict
from openai import AzureOpenAI, AsyncAzureOpenAI

from .models.video import VideoManifest, Segment
from .models.environment import CobraEnvironment
from .analysis import AnalysisConfig
from .analysis.base_analysis_config import SequentialAnalysisConfig
from .cobra_utils import (
    encode_image_base64,
    validate_video_manifest,
    write_video_manifest,
    ensure_person_group_exists,
    process_frame_with_faces,
)


class VideoAnalyzer:
    manifest: VideoManifest
    env: CobraEnvironment
    reprocess_segments: bool
    person_group_id: str
    peoples_list_path: Optional[str]
    peoples_list: Optional[dict]
    emotions_list_path: Optional[str]
    emotions_list: Optional[dict]
    objects_list_path: Optional[str]
    objects_list: Optional[dict]
    themes_list_path: Optional[str]
    themes_list: Optional[dict]
    actions_list_path: Optional[str]
    actions_list: Optional[dict]
    MAX_FRAMES_PER_PROMPT: int = 45  # Maximum number of frames to send in a single prompt

    # take either a video manifest object or a path to a video manifest file
    def __init__(
        self,
        video_manifest: Union[str, VideoManifest],
        env: CobraEnvironment,
        person_group_id: Optional[str] = None,
        peoples_list_path: Optional[str] = None,
        emotions_list_path: Optional[str] = None,
        objects_list_path: Optional[str] = None,
        themes_list_path: Optional[str] = None,
        actions_list_path: Optional[str] = None,
    ):
        # get and validate video manifest
        self.manifest = validate_video_manifest(video_manifest)
        self.env = env
        self.person_group_id = person_group_id
        self.identified_people_in_segment = {}
        
        # Load peoples list if provided
        self.peoples_list = self._load_json_list(peoples_list_path, "peoples")
        
        # Load emotions list if provided
        self.emotions_list = self._load_json_list(emotions_list_path, "emotions")
        
        # Load objects list if provided
        self.objects_list = self._load_json_list(objects_list_path, "objects")
        
        # Load themes list if provided
        self.themes_list = self._load_json_list(themes_list_path, "themes")
        
        # Load actions list if provided
        self.actions_list = self._load_json_list(actions_list_path, "actions")
    
    def _load_json_list(self, file_path, expected_key):
        """Helper method to load and validate JSON list files."""
        if not file_path or not os.path.exists(file_path):
            return None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Check if the file has the expected structure
                if isinstance(data, dict) and expected_key in data:
                    return data
                else:
                    print(f"Warning: {expected_key} list file doesn't contain expected format. Expected '{expected_key}' key.")
                    return None
        except Exception as e:
            print(f"Error loading {expected_key} list: {str(e)}")
            return None

    # Primary method to analyze the video
    def analyze_video(
        self,
        analysis_config: Type[AnalysisConfig],
        run_async=False,
        max_concurrent_tasks=None,
        reprocess_segments=False,
        person_group_id=None,
        **kwargs,
    ):

        self.reprocess_segments = reprocess_segments
        self.person_group_id = person_group_id

        stopwatch_start_time = time.time()

        print(
            f'Starting video analysis: "{analysis_config.name}" for {self.manifest.name}'
        )

        # If person_group_id is provided, verify it exists
        if self.person_group_id:
            if ensure_person_group_exists(self.person_group_id, self.env):
                print(f"Using face recognition with person group: {self.person_group_id}")
            else:
                print(f"Warning: Person group {self.person_group_id} not found or not accessible")
                self.person_group_id = None

        # Analyze videos using the mapreduce sequence
        if analysis_config.analysis_sequence == "mapreduce":
            print(f"Populating prompts for each segment")

            self.generate_segment_prompts(analysis_config)

            if run_async:
                print("Running analysis asynchronously")
                nest_asyncio.apply()
                results_list = asyncio.run(
                    self._analyze_segment_list_async(
                        analysis_config, max_concurrent_tasks=max_concurrent_tasks
                    )
                )
            else:
                print("Running analysis.")
                results_list = self._analyze_segment_list(analysis_config)

        # For refine-style analyses that need to be run sequentially
        elif analysis_config.analysis_sequence == "refine":
            print(f"Analyzing segments sequentially with refinement")
            results_list = self._analyze_segment_list_sequentially(analysis_config)
        else:
            raise ValueError(
                f"You have provided an AnalyisConfig with a analysis_sequence that has not yet been implmented: {analysis_config.analysis_sequence}"
            )

        ## collapse the segment lists into one large list of segments. (Needed for expected ActionSummary format for UI)
        try:
            # Initialize combined results structure
            all_chapters = []
            global_tags = {
                "persons": {},  # Use dict for easier merging
                "actions": {},
                "objects": {}
            }

            # Process each segment's results
            for segment_response in results_list:
                for item in segment_response:
                    if not isinstance(item, dict):
                        continue
                        
                    # Add chapters
                    if "chapters" in item:
                        chapters = item["chapters"]
                        if isinstance(chapters, list):
                            all_chapters.extend(chapters)
                        else:
                            all_chapters.append(chapters)
                    
                    # Merge global tags
                    if "global_tags" in item:
                        tags = item["global_tags"]
                        for category in ["persons", "actions", "objects"]:
                            if category not in tags:
                                continue
                                
                            # Process each tag in the category
                            for tag in tags[category]:
                                name = tag.get("name", "")
                                if not name:
                                    continue
                                    
                                # Initialize if new
                                if name not in global_tags[category]:
                                    global_tags[category][name] = {
                                        "name": name,
                                        "timecodes": []
                                    }
                                
                                # Add timecodes
                                timecodes = tag.get("timecodes", [])
                                if isinstance(timecodes, list):
                                    global_tags[category][name]["timecodes"].extend(timecodes)
                                else:
                                    global_tags[category][name]["timecodes"].append(timecodes)

            # Convert dict back to lists and clean up timecodes
            for category in global_tags:
                global_tags[category] = list(global_tags[category].values())
                for tag in global_tags[category]:
                    # Remove duplicates and sort timecodes
                    tag["timecodes"] = sorted(set(tag["timecodes"]), key=lambda x: float(x.rstrip("s")))

            # Create final results
            final_results = {
                "chapters": all_chapters,
                "global_tags": global_tags
            }

            # Update manifest's global tags
            self.manifest.global_tags = global_tags

            # Generate final summary if enabled
            if hasattr(analysis_config, "run_final_summary") and analysis_config.run_final_summary:
                print(f"Generating final summary for {self.manifest.name}")
                summary_prompt = self.generate_summary_prompt(analysis_config, final_results)
                summary_results = self._call_llm(summary_prompt)
                self.manifest.final_summary = summary_results.choices[0].message.content
                final_results["final_summary"] = self.manifest.final_summary

            # Save results
            final_results_output_path = os.path.join(
                self.manifest.processing_params.output_directory,
                f"_{analysis_config.name}.json",
            )
            
            print(f"Writing results to {final_results_output_path}")
            with open(final_results_output_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(final_results, indent=4, ensure_ascii=False))

        except:
            print(results_list)
            # Build the full path for the final results file.
            final_results_output_path = os.path.join(
                self.manifest.processing_params.output_directory,
                f"_{analysis_config.name}.json",
            )

            # **Ensure the directory exists before writing:**
            os.makedirs(os.path.dirname(final_results_output_path), exist_ok=True)

            try:
                with open(final_results_output_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(final_results, indent=4, ensure_ascii=False))
            except:
                # In the error branch, similarly ensure the directory exists:
                final_results_output_path = os.path.join(
                    self.manifest.processing_params.output_directory,
                    f"_video_analysis_results_{analysis_config.name}_errors.json",
                )
                os.makedirs(os.path.dirname(final_results_output_path), exist_ok=True)
                with open(final_results_output_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(results_list, indent=4, ensure_ascii=False))
                raise ValueError(
                    f"Bad data generated by model. Check the output at {final_results_output_path}"
                )


        stopwatch_end_time = time.time()

        elapsed_time = stopwatch_end_time - stopwatch_start_time

        print(
            f'Video analysis completed in {round(elapsed_time, 3)}: "{analysis_config.name}" for {self.manifest.name}'
        )
        # write the video manifest to the output directory
        write_video_manifest(self.manifest)
        return final_results

    def generate_segment_prompts(self, analysis_config: Type[AnalysisConfig]):
        for segment in self.manifest.segments:
            self._generate_segment_prompt(segment, analysis_config)

    def generate_summary_prompt(
        self, analysis_config: Type[AnalysisConfig], final_results
    ):
        messages = [
            {"role": "system", "content": analysis_config.summary_prompt},
            {"role": "user", "content": json.dumps(final_results)},
        ]
        return messages

    def _analyze_segment_list(
        self,
        analysis_config: Type[AnalysisConfig],
    ):
        results_list = []
        for segment in self.manifest.segments:
            parsed_response = self._analyze_segment(
                segment=segment, analysis_config=analysis_config
            )

            results_list.append(parsed_response)

        # After all segments have been analyzed
        if hasattr(analysis_config, 'process_segment_results'):
            # Use the custom processing method if available
            final_results = analysis_config.process_segment_results(results_list)
        else:
            # Otherwise use the default processing
            # Initialize combined results structure
            all_chapters = []
            global_tags = {
                "persons": {},  # Use dict for easier merging
                "actions": {},
                "objects": {}
            }

            # Process each segment's results
            for segment_response in results_list:
                for item in segment_response:
                    if not isinstance(item, dict):
                        continue
                        
                    # Add chapters
                    if "chapters" in item:
                        chapters = item["chapters"]
                        if isinstance(chapters, list):
                            all_chapters.extend(chapters)
                        else:
                            all_chapters.append(chapters)
                    
                    # Merge global tags
                    if "global_tags" in item:
                        tags = item["global_tags"]
                        for category in ["persons", "actions", "objects"]:
                            if category not in tags:
                                continue
                                
                            # Process each tag in the category
                            for tag in tags[category]:
                                name = tag.get("name", "")
                                if not name:
                                    continue
                                    
                                # Initialize if new
                                if name not in global_tags[category]:
                                    global_tags[category][name] = {
                                        "name": name,
                                        "timecodes": []
                                    }
                                
                                # Add timecodes
                                timecodes = tag.get("timecodes", [])
                                if isinstance(timecodes, list):
                                    global_tags[category][name]["timecodes"].extend(timecodes)
                                else:
                                    global_tags[category][name]["timecodes"].append(timecodes)

            # Convert dict back to lists and clean up timecodes
            for category in global_tags:
                global_tags[category] = list(global_tags[category].values())
                for tag in global_tags[category]:
                    # Remove duplicates and sort timecodes
                    tag["timecodes"] = sorted(set(tag["timecodes"]), key=lambda x: float(x.rstrip("s")))

            # Create final results
            final_results = {
                "chapters": all_chapters,
                "global_tags": global_tags
            }

            # Update manifest's global tags
            self.manifest.global_tags = global_tags

            # Generate final summary if enabled
            if hasattr(analysis_config, "run_final_summary") and analysis_config.run_final_summary:
                print(f"Generating final summary for {self.manifest.name}")
                summary_prompt = self.generate_summary_prompt(analysis_config, final_results)
                summary_results = self._call_llm(summary_prompt)
                self.manifest.final_summary = summary_results.choices[0].message.content
                final_results["final_summary"] = self.manifest.final_summary

        return final_results

    def _analyze_segment_list_sequentially(
        self, analysis_config: Type[SequentialAnalysisConfig]
    ):
        # if the analysis config is not a SequentialAnalysisConfig, raise an error
        if not isinstance(analysis_config, SequentialAnalysisConfig):
            raise ValueError(
                f"Sequential analysis can only be run with an obect that is a subclass of SequentialAnalysisConfig. You have provided an object of type {type(analysis_config)}"
            )

        # Start the timer
        stopwatch_start_time = time.time()

        results_list = []

        for i, segment in enumerate(self.manifest.segments):
            # check if the segment has already been analyzed, if so, skip it
            if (
                self.reprocess_segments is False
                and analysis_config.name in segment.analysis_completed
            ):
                print(
                    f"Segment {segment.segment_name} has already been analyzed, loading the stored value."
                )
                results_list.append(segment.analyzed_result[analysis_config.name])
                continue
            else:
                print(f"Analyzing segment {segment.segment_name}")

            messages = []
            number_of_previous_results_to_refine = (
                analysis_config.number_of_previous_results_to_refine
            )
            # generate the prompt for the segment
            # include the right number of previous results to refine and generate the prompt
            if len(results_list) == 0:
                result_list_subset = None
            if len(results_list) <= number_of_previous_results_to_refine:
                result_list_subset = results_list[: len(results_list)]
            else:
                result_list_subset = results_list[:number_of_previous_results_to_refine]

            result_list_subset_string = json.dumps(result_list_subset)

            # if it's the first segment, generate without the refine prompt; if it is not the first segment, generate with the refine prompt
            if i == 0:
                system_prompt_template = (
                    analysis_config.generate_system_prompt_template(
                        is_refine_step=False
                    )
                )

                system_prompt = system_prompt_template.format(
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    segment_duration=segment.segment_duration,
                    number_of_frames=segment.number_of_frames,
                    number_of_previous_results_to_refine=number_of_previous_results_to_refine,
                    video_duration=self.manifest.source_video.duration,
                    analysis_lens=analysis_config.lens_prompt,
                    results_template=analysis_config.results_template,
                    current_summary=result_list_subset_string,
                )
            else:
                system_prompt_template = (
                    analysis_config.generate_system_prompt_template(is_refine_step=True)
                )

                system_prompt = system_prompt_template.format(
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    segment_duration=segment.segment_duration,
                    number_of_frames=segment.number_of_frames,
                    number_of_previous_results_to_refine=number_of_previous_results_to_refine,
                    video_duration=self.manifest.source_video.duration,
                    analysis_lens=analysis_config.lens_prompt,
                    results_template=analysis_config.results_template,
                    current_summary=result_list_subset_string,
                )

            messages.append({"role": "system", "content": system_prompt})

            # Form the user prompt with the refine prompt, the audio transcription (if available), and the video frames
            user_content = []
            if segment.transcription is not None:
                user_content.append(
                    {
                        "type": "text",
                        "text": f"Audio Transcription for the next {segment.segment_duration} seconds: {segment.transcription}",
                    }
                )
            user_content.append(
                {
                    "type": "text",
                    "text": f"Next are the {segment.number_of_frames} frames from the next {segment.segment_duration} seconds of the video:",
                }
            )
            # Include the frames
            for i, frame in enumerate(segment.segment_frames_file_path):
                frame_time = segment.segment_frame_time_intervals[i]
                base64_image = encode_image_base64(frame)
                user_content.append(
                    {
                        "type": "text",
                        "text": f"Below is the frame at start_time {frame_time} seconds. Use this to provide timestamps and understand time.",
                    }
                )
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low",
                        },
                    }
                )

            # add user content to the messages
            messages.append({"role": "user", "content": user_content})

            # write the prompt to the manifest
            prompt_output_path = os.path.join(
                segment.segment_folder_path, f"{segment.segment_name}_prompt.json"
            )

            with open(prompt_output_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(messages, indent=4))

            segment.segment_prompt_path = prompt_output_path

            # call the LLM to analyze the segment
            response = self._call_llm(messages)
            parsed_response = self._parse_llm_json_response(response)

            # append the result to the results list
            results_list.append(parsed_response)
            elapsed_time = time.time() - stopwatch_start_time
            print(
                f"Segment {segment.segment_name} analyzed in {round(elapsed_time, 2)} seconds."
            )

            # update the segment object with the analyzed results
            segment.analyzed_result[analysis_config.name] = parsed_response
            segment.analysis_completed.append(analysis_config.name)

            # update the manifest on disk (allows for checkpointing)
            write_video_manifest(self.manifest)

        elapsed_time = time.time() - stopwatch_start_time
        print(f"Analysis completed in {round(elapsed_time,2)} seconds.")

        return results_list

    async def _analyze_segment_list_async(
        self, analysis_config: Type[AnalysisConfig], max_concurrent_tasks=None
    ):
        if max_concurrent_tasks is None:
            max_concurrent_tasks = len(self.manifest.segments)
        else:
            max_concurrent_tasks = min(
                int(max_concurrent_tasks), len(self.manifest.segments)
            )

        sempahore = asyncio.Semaphore(max_concurrent_tasks)

        async def sem_task(segment):
            async with sempahore:
                return await self._analyze_segment_async(segment, analysis_config)

        async def return_value_task(segment):
            return segment.analyzed_result[analysis_config.name]

        segment_task_list = []

        for segment in self.manifest.segments:
            if (
                self.reprocess_segments is False
                and analysis_config.name in segment.analysis_completed
            ):
                print(
                    f"Segment {segment.segment_name} has already been analyzed, loading the stored value."
                )
                segment_task_list.append(return_value_task(segment))
                continue
            else:
                segment_task_list.append(sem_task(segment))

        results_list = await asyncio.gather(*segment_task_list)

        return results_list

    def _analyze_segment(
        self,
        segment: Segment,
        analysis_config: Type[AnalysisConfig]
    ):
        # Face recognition as before (no change)
        if not hasattr(self, 'identified_people_in_segment'):
            self.identified_people_in_segment = {}
        if self.person_group_id:
            for i, frame_path in enumerate(segment.segment_frames_file_path):
                timestamp = round(
                    float(segment.start_time)
                    + float(segment.segment_frame_time_intervals[i]),
                    2
                )
                timestamp_key = f"{timestamp}s"
                face_results = process_frame_with_faces(frame_path, self.env, self.person_group_id)
                if face_results.get("identified_people"):
                    self.identified_people_in_segment[timestamp_key] = ", ".join(face_results["identified_people"])

        # If already analyzed and not reprocessing, skip
        if (not self.reprocess_segments 
            and analysis_config.name in segment.analysis_completed):
            return segment.analyzed_result[analysis_config.name]

        # CHUNKING LOGIC:
        frames = segment.segment_frames_file_path
        times  = segment.segment_frame_time_intervals

        # If this segment has 50 or fewer frames, do it the normal way
        if len(frames) <= self.MAX_FRAMES_PER_PROMPT:
            prompt = self._generate_segment_prompt(
                segment,
                analysis_config,
                frames_subset=frames,
                times_subset=times
            )
            llm_response = self._call_llm(prompt)
            parsed_response = self._parse_llm_json_response(llm_response)
        else:
            # We have more than 50 frames, so break them into partial chunks
            chunked_responses = []
            for start_idx in range(0, len(frames), self.MAX_FRAMES_PER_PROMPT):
                end_idx = start_idx + self.MAX_FRAMES_PER_PROMPT
                frames_chunk = frames[start_idx:end_idx]
                times_chunk  = times[start_idx:end_idx]

                # Generate partial prompt for just these frames
                prompt_chunk = self._generate_segment_prompt(
                    segment,
                    analysis_config,
                    frames_subset=frames_chunk,
                    times_subset=times_chunk,
                    is_partial_chunk=True
                )

                # Call LLM on this chunk
                response_chunk = self._call_llm(prompt_chunk)
                parsed_chunk   = self._parse_llm_json_response(response_chunk)
                chunked_responses.append(parsed_chunk)

            # Merge all partial chunk responses into a single dict
            parsed_response = self._merge_chunked_responses(chunked_responses)

        # Save final merged result for this segment
        segment.analyzed_result[analysis_config.name] = parsed_response
        segment.analysis_completed.append(analysis_config.name)

        # Optionally save to disk
        segment_result_path = os.path.join(segment.segment_folder_path, "_segment_analyzed_result.json")
        with open(segment_result_path, "w", encoding="utf-8") as f:
            json.dump(parsed_response, f, indent=4, ensure_ascii=False)

        # Write updated manifest
        write_video_manifest(self.manifest)
        return parsed_response

    async def _analyze_segment_async(
        self,
        segment: Segment,
        analysis_config: AnalysisConfig,
    ):

        start_time = time.time()
        print(f"Starting analysis for segment {segment.segment_name}")

        # Generate the prompt
        segment_prompt = self._generate_segment_prompt(segment, analysis_config)

        # submit call the LLM to analyze the segment
        response = await self._call_llm_async(segment_prompt)

        # parse the response and update the segment object
        parsed_response = self._parse_llm_json_response(response)
        segment.analyzed_result[analysis_config.name] = parsed_response
        segment.analysis_completed.append(analysis_config.name)

        # write the raw response outputs
        llm_response_output_path = os.path.join(
            segment.segment_folder_path, f"_segment_llm_response.json"
        )
        with open(llm_response_output_path, "w", encoding="utf-8") as f:
            f.write(response.model_dump_json(indent=4))

        # write the LLM generated analysis
        parsed_response_output_path = os.path.join(
            segment.segment_folder_path, f"_segment_analyzed_result.json"
        )
        with open(parsed_response_output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(parsed_response))

        endtime = time.time()
        elapsed_time = endtime - start_time
        print(
            f"Segment {segment.segment_name} analyzed in {round(elapsed_time, 3)} seconds"
        )

        return parsed_response

    def _generate_segment_refine_prompts(self, Segment, AnalysisConfig):
        pass

    def _generate_segment_prompt(
        self,
        segment: Segment,
        analysis_config: AnalysisConfig,
        frames_subset: List[str] = None,
        times_subset: List[float] = None,
        is_partial_chunk: bool = False
    ):
        """
        Generate a prompt for a segment.
        
        Args:
            segment: Segment object containing video segment info
            analysis_config: Analysis configuration
            frames_subset: Optional subset of frames to process
            times_subset: Optional subset of frame timestamps
            is_partial_chunk: Whether this is a partial chunk of a larger segment
        """
        # Use provided frames/times or fall back to full segment
        frames_to_process = frames_subset if frames_subset is not None else segment.segment_frames_file_path
        times_to_process = times_subset if times_subset is not None else segment.segment_frame_time_intervals
        
        # Initialize list to track identified faces
        identified_people = set()
        
        # Generate system prompt
        system_prompt_template = analysis_config.generate_system_prompt_template()
        system_prompt = system_prompt_template.format(
            # Format seconds with millisecond precision
            start_time=f"{float(segment.start_time):.3f}",
            end_time=f"{float(segment.end_time):.3f}",
            segment_duration=segment.segment_duration,
            number_of_frames=segment.number_of_frames,
            video_duration=f"{float(self.manifest.source_video.duration):.3f}",
            analysis_lens=getattr(analysis_config, "system_prompt_lens", "")
            if hasattr(analysis_config, "system_prompt_lens")
            else "",
            results_template=analysis_config.results_template,
        )

        # Prepare user content
        user_content = []

        # Add transcription if available
        if segment.transcription is not None:
            transcription_text = segment.transcription
        else:
            transcription_text = "No transcription available"
        
        user_content.append(
            {
                "type": "text",
                "text": f"Audio Transcription for the next {segment.segment_duration} seconds: {transcription_text}",
            }
        )

        # Add peoples list information if available
        if self.peoples_list:
            user_content.append(self._generate_list_prompt("peoples", self.peoples_list))
        
        # Add emotions list information if available
        if self.emotions_list:
            user_content.append(self._generate_list_prompt("emotions", self.emotions_list))
        
        # Add objects list information if available
        if self.objects_list:
            user_content.append(self._generate_list_prompt("objects", self.objects_list))
        
        # Add themes list information if available
        if self.themes_list:
            user_content.append(self._generate_list_prompt("themes", self.themes_list))
        
        # Add actions list information if available
        if self.actions_list:
            user_content.append(self._generate_list_prompt("actions", self.actions_list))
        
        # Add frame analysis instructions
        user_content.append({
            "type": "text",
            "text": """IMPORTANT: For each frame, analyze and note:
1. VISUAL ELEMENTS:
   - All persons visible (describe appearance, clothing, position)
   - Actions being performed
   - Objects and items in the scene
   - Setting and environment details
   - Body language and visual emotions

2. TEMPORAL TRACKING:
   - Note exact timestamps for all observations
   - Track when elements appear/disappear
   - Note any changes between frames

3. INTEGRATION:
   - Combine visual observations with audio context
   - Ensure all persons, actions, and objects are tagged with precise timecodes
   - Cross-reference visual and audio information"""
        })

        # Add frames with timestamps
        for i, frame_path in enumerate(frames_to_process):
            timestamp = round(float(segment.start_time) + float(times_to_process[i]), 3)
            
            # Encode the image
            image_content = encode_image_base64(frame_path)
            
            # Add frame timestamp first
            user_content.append({
                "type": "text",
                "text": f"\nFrame at {timestamp:.3f}s:"
            })
            
            # Add the image
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_content}",
                    "detail": "high"
                }
            })

        # Add summary of identified people across all frames if any were found
        if self.person_group_id and self.identified_people_in_segment:
            user_content.insert(0, {
                "type": "text",
                "text": f"Identified people in this segment: {', '.join(sorted(self.identified_people_in_segment.values()))}"
            })
            
            # Add instructions about personNames field and prohibit using names in other fields
            user_content.insert(1, {
                "type": "text",
                "text": f"IMPORTANT INSTRUCTIONS ABOUT NAMES:\n\n1. For the 'personNames' field in your output, ONLY include names of people that have been identified via face recognition. If no people were identified, leave the 'personNames' field blank.\n\n2. You CAN naturally mention recognizable public figures (celebrities, politicians, athletes, etc.) in your description fields if you recognize them visually, but do NOT mention the specific names that appear in the 'personNames' field in your other fields.\n\n3. The 'personNames' field is strictly for storing face-recognized identities only, while descriptions should use your own visual recognition abilities for public figures and generic terms for non-public individuals.\n\n4. For the 'peoples' field, identify any people from the provided list who appear in the images, based on their jersey numbers or other identifying features."
            })

        # Save the prompt in the segment
        if segment.segment_prompt_path is None:
            segment.segment_prompt_path = os.path.join(
                segment.segment_folder_path, "segment_prompt.json"
            )

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        with open(segment.segment_prompt_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(prompt, ensure_ascii=False))

        return prompt

    def _generate_list_prompt(self, list_type, list_data):
        """Helper method to generate prompt content for each list type."""
        prompt = {
            "type": "text",
            "text": f"IMPORTANT: Analyze for the following {list_type} in this segment. "
        }
        
        if "instructions" in list_data:
            prompt["text"] += list_data["instructions"] + "\n\n"
        
        prompt["text"] += f"{list_type.upper()} TO IDENTIFY:\n"
        
        # Get items from the list
        items = list_data.get(list_type, [])
        
        for item in items:
            name = item.get("name", "Unknown")
            description = item.get("description", "")
            prompt["text"] += f"- {name}: {description}\n"
        
        prompt["text"] += f"\nGUIDELINES FOR {list_type.upper()} IDENTIFICATION:\n"
        prompt["text"] += f"1. List any identified {list_type} in the '{list_type}' field of your response.\n"
        prompt["text"] += f"2. Only include {list_type} that are clearly present in the segment.\n"
        
        if list_type == "themes":
            prompt["text"] += "3. Choose the most appropriate theme that best represents the segment.\n"
        else:
            prompt["text"] += f"3. For each {list_type[:-1]}, note its context and relevance where possible.\n"
        
        return prompt

    def _call_llm(self, messages_list: list):
        client = AzureOpenAI(
            api_key=self.env.vision.api_key.get_secret_value(),
            api_version=self.env.vision.api_version,
            azure_endpoint=self.env.vision.endpoint,
        )

        response = client.chat.completions.create(
            model=self.env.vision.deployment,
            messages=messages_list,
            max_tokens=2000,
        )

        return response

    async def _call_llm_async(self, messages_list: list):
        client = AsyncAzureOpenAI(
            api_key=self.env.vision.api_key.get_secret_value(),
            api_version=self.env.vision.api_version,
            azure_endpoint=self.env.vision.endpoint,
        )

        response = await client.chat.completions.create(
            model=self.env.vision.deployment,
            messages=messages_list,
            max_tokens=2000,
        )

        return response

    def _parse_llm_json_response(self, response):
        try:
            # Extract the content
            content = response.choices[0].message.content
            
            # Check if the content is wrapped in markdown code blocks
            if "```json" in content:
                # Extract the JSON from the markdown code blocks
                content = content.split("```json")[1].split("```")[0].strip()
            
            # Parse the JSON content
            parsed_content = json.loads(content)
            
            # Get all identified people across all timestamps if we used face recognition
            all_identified_names = set()
            if self.person_group_id and hasattr(self, 'identified_people_in_segment'):
                for people_str in self.identified_people_in_segment.values():
                    for name in [name.strip() for name in people_str.split(",")]:
                        if name:
                            all_identified_names.add(name)
            
            # Process each item in the response
            if isinstance(parsed_content, list):
                for item in parsed_content:
                    # Handle field name conversions
                    if "start_timestamp" in item and "start" not in item:
                        item["start"] = item.pop("start_timestamp")
                    if "end_timestamp" in item and "end" not in item:
                        item["end"] = item.pop("end_timestamp")
                    if "scene_theme" in item and "theme" not in item:
                        item["theme"] = item.pop("scene_theme")
                    if "key_objects" in item and "objects" not in item:
                        item["objects"] = item.pop("key_objects")
                    
                    # Convert timestamp formats to consistent decimal seconds format
                    for time_field in ["start", "end"]:
                        if time_field in item and isinstance(item[time_field], str):
                            try:
                                # Handle millisecond format (e.g., "16000ms")
                                if item[time_field].endswith("ms"):
                                    ms_value = float(item[time_field].rstrip("ms"))
                                    item[time_field] = f"{(ms_value / 1000):.3f}s"
                                # Handle existing seconds format but ensure consistent precision
                                elif item[time_field].endswith("s"):
                                    seconds_value = float(item[time_field].rstrip("s"))
                                    item[time_field] = f"{seconds_value:.3f}s"
                                # Handle raw number (assume seconds)
                                else:
                                    seconds_value = float(item[time_field])
                                    item[time_field] = f"{seconds_value:.3f}s"
                            except ValueError:
                                # If conversion fails, keep the original value
                                pass
                    
                    # Handle personNames field - should ONLY contain faces identified by the Azure Face API
                    if "characters" in item:
                        # Remove characters field and replace with personNames
                        item.pop("characters", None)
                        # Add personNames based on face recognition only
                        item["personNames"] = ""
                    elif "personNames" not in item:
                        # Ensure it exists
                        item["personNames"] = ""
                    
                    # Now populate personNames from identified faces
                    if self.person_group_id and hasattr(self, 'identified_people_in_segment'):
                        # Find the closest timestamp to use for matching faces
                        # This addresses slight variations in formatting that might cause exact matches to fail
                        start_timestamp = item["start"]
                        if start_timestamp in self.identified_people_in_segment:
                            item["personNames"] = self.identified_people_in_segment[start_timestamp]
                        else:
                            # Try to find a close match within 0.1 seconds
                            try:
                                start_time = float(start_timestamp.rstrip("s"))
                                for ts_key, people in self.identified_people_in_segment.items():
                                    ts_value = float(ts_key.rstrip("s"))
                                    if abs(ts_value - start_time) < 0.1:  # Within 0.1 seconds
                                        item["personNames"] = people
                                        break
                            except (ValueError, AttributeError):
                                # If conversion fails, keep empty personNames
                                pass
                    elif not self.person_group_id and item["personNames"]:
                        # If personNames exists but we didn't use face recognition, it should be empty
                        item["personNames"] = ""
                    
                    # Handle peoples field - ensure it only contains names, not other attributes
                    if "peoples" in item:
                        # If peoples is a list of objects or complex items, extract just the names
                        if isinstance(item["peoples"], list):
                            cleaned_peoples = []
                            for person in item["peoples"]:
                                if isinstance(person, dict) and "name" in person:
                                    # Extract only the name from objects
                                    cleaned_peoples.append(person["name"])
                                elif isinstance(person, str):
                                    # Already a name
                                    cleaned_peoples.append(person)
                            item["peoples"] = cleaned_peoples
                        elif isinstance(item["peoples"], str):
                            # Convert comma-separated string to list
                            item["peoples"] = [name.strip() for name in item["peoples"].split(",") if name.strip()]
                    else:
                        # Initialize empty peoples array if it doesn't exist
                        item["peoples"] = []
                        
                    # Handle emotions field - ensure it exists
                    if "emotions" not in item:
                        item["emotions"] = []
                    elif isinstance(item["emotions"], str):
                        # Convert comma-separated string to list
                        item["emotions"] = [emotion.strip() for emotion in item["emotions"].split(",") if emotion.strip()]
                    
                    # Sanitize other fields to remove identified names from face recognition
                    if all_identified_names and self.person_group_id:
                        for field in ["summary", "actions", "objects"]:
                            if field in item and item[field]:
                                sanitized_text = item[field]
                                
                                # Only remove the custom-recognized face names
                                # Keep any public figure names the model identified naturally
                                for name in all_identified_names:
                                    # Don't automatically replace common celebrity names that the model might recognize visually
                                    # Only replace the exact names found in face recognition
                                    if name in sanitized_text and name in self.identified_people_in_segment.get(item["start"], ""):
                                        # Replace the name with an appropriate generic term
                                        if "he " in sanitized_text.lower() or "his " in sanitized_text.lower():
                                            replacement = "a man"
                                        elif "she " in sanitized_text.lower() or "her " in sanitized_text.lower():
                                            replacement = "a woman" 
                                        else:
                                            replacement = "a person"
                                            
                                        sanitized_text = sanitized_text.replace(name, replacement)
                                
                                item[field] = sanitized_text
            
            return parsed_content
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response: {content}")
            return content

    def _merge_chunked_responses(self, chunked_responses: List[Dict]) -> Dict:
        """
        Merge multiple chunked responses into a single response.
        
        Args:
            chunked_responses: List of response dictionaries from different chunks
            
        Returns:
            Merged response dictionary
        """
        if not chunked_responses:
            return {}
        
        merged = {
            "chapters": [],
            "global_tags": {
                "persons": [],
                "actions": [],
                "objects": []
            }
        }
        
        # Merge all chunks
        for response in chunked_responses:
            if not isinstance(response, dict):
                continue
            
            # Merge chapters
            if "chapters" in response:
                merged["chapters"].extend(response["chapters"])
            
            # Merge global tags
            if "global_tags" in response:
                for tag_type in ["persons", "actions", "objects"]:
                    if tag_type in response["global_tags"]:
                        merged["global_tags"][tag_type].extend(
                            response["global_tags"][tag_type]
                        )
        
        # Deduplicate global tags by name and combine their timecodes
        for tag_type in merged["global_tags"]:
            tag_dict = {}
            for tag in merged["global_tags"][tag_type]:
                name = tag.get("name")
                if not name:
                    continue
                
                if name not in tag_dict:
                    tag_dict[name] = {"name": name, "timecodes": []}
                tag_dict[name]["timecodes"].extend(tag.get("timecodes", []))
            
            # Convert back to list and sort timecodes
            merged["global_tags"][tag_type] = list(tag_dict.values())
            for tag in merged["global_tags"][tag_type]:
                tag["timecodes"] = sorted(tag["timecodes"], 
                                        key=lambda x: float(x["start"].rstrip("s")))
        
        return merged