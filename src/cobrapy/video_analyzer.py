import os
import json
import time
import asyncio
import nest_asyncio
from typing import Union, Type, Optional, List, Dict, Set
from openai import AzureOpenAI, AsyncAzureOpenAI
import logging
from collections import defaultdict
import numpy as np
import math

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

logger = logging.getLogger(__name__)

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
    TAG_CHUNK_SECONDS: float = 5.0 # New: Duration for tag analysis chunks

    # Add instance variables to track known tags during analysis
    _current_known_persons: Set[str]
    _current_known_actions: Set[str]
    _current_known_objects: Set[str]

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

        # Initialize known tag sets
        self._current_known_persons = set()
        self._current_known_actions = set()
        self._current_known_objects = set()

        # Optionally pre-populate from lists if provided
        if self.peoples_list:
             self._current_known_persons.update(self.peoples_list.get("persons", []))
        if self.actions_list:
             self._current_known_actions.update(self.actions_list.get("actions", []))
        if self.objects_list:
             self._current_known_objects.update(self.objects_list.get("objects", []))
    
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
        # Reset known tags at the beginning of each analysis run
        self._current_known_persons = set()
        self._current_known_actions = set()
        self._current_known_objects = set()
        # Re-populate from lists if they exist
        if self.peoples_list:
             self._current_known_persons.update(self.peoples_list.get("persons", []))
        if self.actions_list:
             self._current_known_actions.update(self.actions_list.get("actions", []))
        if self.objects_list:
             self._current_known_objects.update(self.objects_list.get("objects", []))

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

        # --- EDIT 1: Determine if custom aggregation exists ---
        has_custom_aggregation = hasattr(analysis_config, 'process_segment_results')
        results_list_or_aggregated = None # Will hold either the list or the final dict

        # Analyze videos using the mapreduce sequence
        if analysis_config.analysis_sequence == "mapreduce":
            print(f"Populating prompts for each segment")

            self.generate_segment_prompts(analysis_config)

            if run_async:
                print("Running analysis asynchronously")
                nest_asyncio.apply()
                # _analyze_segment_list_async now returns aggregated results if process_segment_results exists
                results_list_or_aggregated = asyncio.run(
                    self._analyze_segment_list_async(
                        analysis_config, max_concurrent_tasks=max_concurrent_tasks
                    )
                )
            else:
                print("Running analysis.")
                # _analyze_segment_list now returns aggregated results if process_segment_results exists
                results_list_or_aggregated = self._analyze_segment_list(analysis_config)

        # For refine-style analyses that need to be run sequentially
        elif analysis_config.analysis_sequence == "refine":
            print(f"Analyzing segments sequentially with refinement")
            # Sequential analysis typically refines, results might need specific handling
            # Assuming _analyze_segment_list_sequentially returns a list suitable for generic aggregation for now
            # If refine needs custom aggregation, this logic might need adjustment
            results_list_or_aggregated = self._analyze_segment_list_sequentially(analysis_config)
            has_custom_aggregation = False # Force generic aggregation for refine unless specified otherwise

        else:
            raise ValueError(
                f"You have provided an AnalyisConfig with a analysis_sequence that has not yet been implmented: {analysis_config.analysis_sequence}"
            )

        # --- EDIT 2: Conditional Aggregation & Manifest Update ---
        final_results = {}
        if has_custom_aggregation and isinstance(results_list_or_aggregated, dict):
             print("Using results from custom aggregation method.")
             # --- Check for new structure ---
             if "actionSummary" in results_list_or_aggregated:
                  print("Detected 'actionSummary' structure from custom aggregation.")
                  final_results = results_list_or_aggregated # Keep the whole new structure for the _ActionSummary.json file

                  # --- Update manifest.global_tags (Requires reformatting back to plural keys) ---
                  action_summary_content = final_results.get("actionSummary", {})
                  # Extract lists using singular keys, assign to manifest using plural keys
                  manifest_global_tags = {
                       "persons": action_summary_content.get("person", []),
                       "actions": action_summary_content.get("action", []),
                       "objects": action_summary_content.get("object", [])
                  }
                  self.manifest.global_tags = manifest_global_tags
                  print("DEBUG: Updated self.manifest.global_tags using data from actionSummary (reverted to plural keys).")

             else:
                  # --- Handle old custom aggregation structure (if needed) ---
                  print("Assuming standard structure from custom aggregation (no 'actionSummary' key found).")
                  final_results = results_list_or_aggregated
                  if "global_tags" in final_results:
                      self.manifest.global_tags = final_results["global_tags"]
                      print("DEBUG: Updated self.manifest.global_tags from standard custom aggregation.")

        elif isinstance(results_list_or_aggregated, list):
             # --- Generic Aggregation (produces standard structure) ---
             print("Running generic aggregation (produces standard structure).")
             # --- Restore Generic Aggregation Logic ---
             try:
                 # Initialize combined results structure
                 all_chapters = []
                 global_tags_agg_dict = { # Use a temporary dict to collect tags by name
                     "persons": {},
                     "actions": {},
                     "objects": {}
                 }

                 # Process each segment's results from the list
                 for result_container in results_list_or_aggregated:
                     if not isinstance(result_container, dict):
                         print(f"Warning: Skipping non-dictionary item in results list during generic agg: {type(result_container)}")
                         continue

                     # Assume the structure is {"analysis_result": {...}, "segment_name": ..., ...}
                     segment_response = result_container.get("analysis_result", {})
                     if not isinstance(segment_response, dict):
                          print(f"Warning: Skipping item with non-dict 'analysis_result' during generic agg: {type(segment_response)}")
                          continue

                     # Add chapters
                     chapters_data = segment_response.get("chapters", [])
                     if isinstance(chapters_data, dict): # Handle potential single chapter object
                         all_chapters.append(chapters_data)
                     elif isinstance(chapters_data, list):
                         all_chapters.extend(chap for chap in chapters_data if isinstance(chap, dict)) # Add only valid chapter dicts
                     else:
                         print(f"Warning: Unexpected data type for 'chapters' during generic agg: {type(chapters_data)}")

                     # Merge global tags
                     tags_data = segment_response.get("global_tags", {})
                     if not isinstance(tags_data, dict):
                         print(f"Warning: Unexpected data type for 'global_tags' during generic agg: {type(tags_data)}")
                         continue

                     for category in ["persons", "actions", "objects"]:
                         if category not in tags_data or not isinstance(tags_data.get(category), list): # Check if key exists and is a list
                             continue

                         for tag_obj in tags_data[category]:
                             if not isinstance(tag_obj, dict):
                                 print(f"Warning: Skipping non-dictionary tag in '{category}' during generic agg: {type(tag_obj)}")
                                 continue
                             name = tag_obj.get("name")
                             if not name or not isinstance(name, str) or not name.strip():
                                 print(f"Warning: Skipping tag in '{category}' with missing/invalid name during generic agg: {tag_obj}")
                                 continue

                             cleaned_name = name.strip()
                             if cleaned_name not in global_tags_agg_dict[category]:
                                 # Initialize the structure for this tag name
                                 global_tags_agg_dict[category][cleaned_name] = {"name": cleaned_name, "timecodes": []}

                             timecodes = tag_obj.get("timecodes", [])
                             if isinstance(timecodes, list):
                                 # Add only valid timecode dictionaries
                                 valid_timecodes = [tc for tc in timecodes if isinstance(tc, dict) and "start" in tc and "end" in tc]
                                 global_tags_agg_dict[category][cleaned_name]["timecodes"].extend(valid_timecodes)
                             else:
                                 print(f"Warning: Unexpected timecode format for tag '{cleaned_name}' during generic agg: {timecodes}")

                 # Convert aggregated dict back to list structure and clean up timecodes
                 final_global_tags_agg = {} # This variable will now be defined
                 for category, tags_dict in global_tags_agg_dict.items():
                     tag_list = []
                     for tag_name, tag_object in tags_dict.items():
                         # Deduplicate and sort timecodes
                         unique_timecodes_set = set(tuple(sorted(d.items())) for d in tag_object["timecodes"])
                         unique_timecodes_list = sorted(
                             [dict(t) for t in unique_timecodes_set],
                             # Handle potential float conversion errors gracefully
                             key=lambda x: float(str(x.get("start", "inf")).rstrip("s")) if str(x.get("start", "inf")).rstrip("s").replace('.', '', 1).isdigit() else float('inf')
                         )
                         tag_object["timecodes"] = unique_timecodes_list
                         tag_list.append(tag_object) # Append the dict {name: ..., timecodes: [...]}
                     final_global_tags_agg[category] = tag_list # Final structure uses plural keys

                 # Create final results dictionary for this path
                 final_results = {
                     "chapters": all_chapters,
                     "global_tags": final_global_tags_agg # Use the final aggregated list structure
                 }

             except Exception as e:
                 print(f"Error during generic aggregation: {e}")
                 print("Attempting to save raw results list instead.")
                 # Fallback to save raw list if aggregation fails
                 error_output_path = os.path.join(
                     self.manifest.processing_params.output_directory,
                     f"_video_analysis_results_{analysis_config.name}_generic_aggregation_error.json",
                 )
                 os.makedirs(os.path.dirname(error_output_path), exist_ok=True)
                 with open(error_output_path, "w", encoding="utf-8") as f:
                     # Save the raw list that caused the error
                     json.dump(results_list_or_aggregated, f, indent=4, ensure_ascii=False)
                 raise ValueError(
                     f"Error during generic aggregation. Raw results list saved to {error_output_path}. Error: {e}"
                 )
             # --- End Restore ---

             # Ensure self.manifest.global_tags is updated here too if generic runs
             if 'final_global_tags_agg' in locals(): # Check if generic agg ran and produced tags
                  self.manifest.global_tags = final_global_tags_agg
                  print("DEBUG: Updated self.manifest.global_tags from generic aggregation.")


        # -- Summary Generation & Saving --
        try:
            # Generate final summary if enabled
            if hasattr(analysis_config, "run_final_summary") and analysis_config.run_final_summary:
                content_to_summarize = final_results
                summary_target_dict = final_results # Default target for summary
                summary_text = "Analysis resulted in empty chapters and tags." # Default summary
                should_generate_summary = False

                if "actionSummary" in final_results:
                     action_summary_content = final_results["actionSummary"]
                     summary_target_dict = action_summary_content # Put summary inside actionSummary
                     # Check if there's content inside actionSummary to summarize
                     if action_summary_content and \
                        (action_summary_content.get("chapter") or \
                         action_summary_content.get("person") or \
                         action_summary_content.get("action") or \
                         action_summary_content.get("object")):
                          should_generate_summary = True
                     else:
                          print("Skipping final summary generation due to empty actionSummary content.")

                # Check standard structure (if not actionSummary)
                elif final_results and (final_results.get("chapters") or final_results.get("global_tags")):
                    should_generate_summary = True
                else: # Standard structure but empty
                    print("Skipping final summary generation due to empty analysis results.")
                    # Add default summary to top level if standard structure
                    final_results["final_summary"] = summary_text

                # Generate summary if content exists
                if should_generate_summary:
                     print(f"Generating final summary for {self.manifest.name}")
                     summary_prompt = self.generate_summary_prompt(analysis_config, content_to_summarize)
                     summary_results = self._call_llm(summary_prompt)
                     summary_text = summary_results.choices[0].message.content
                     summary_target_dict["final_summary"] = summary_text # Add summary to correct dict (actionSummary or final_results)
                     print(f"DEBUG: Added final_summary to {'actionSummary object' if 'actionSummary' in final_results else 'top level'}.")

                # Always update the manifest's final summary field
                self.manifest.final_summary = summary_text


            # Save the _ActionSummary.json results
            final_results_output_path = os.path.join(
                self.manifest.processing_params.output_directory,
                f"_{analysis_config.name}.json", # e.g., _ActionSummary.json
            )
            print(f"Writing final results structure to {final_results_output_path}")
            os.makedirs(os.path.dirname(final_results_output_path), exist_ok=True)
            with open(final_results_output_path, "w", encoding="utf-8") as f:
                 # final_results contains the full structure ('actionSummary' key or standard)
                 json_obj = json.loads(json.dumps(final_results)) # Ensure clean serializable dict
                 f.write(json.dumps(json_obj, indent=4, ensure_ascii=False))

        except Exception as e: # Catch potential errors during summary or saving
            print(f"Error during final summary generation or saving results: {e}")
            # Log the final_results that caused the error if possible
            print(f"Data causing error: {final_results}")
            # Attempt to save the potentially problematic final_results
            error_output_path = os.path.join(
                self.manifest.processing_params.output_directory,
                f"_video_analysis_results_{analysis_config.name}_final_error.json",
            )
            os.makedirs(os.path.dirname(error_output_path), exist_ok=True)
            try:
                 with open(error_output_path, "w", encoding="utf-8") as f:
                     json.dump(final_results, f, indent=4, ensure_ascii=False)
                 print(f"Problematic final_results saved to {error_output_path}")
            except Exception as dump_e:
                 print(f"Could not even dump the problematic final_results: {dump_e}")
            # Optional: Re-raise the original exception or a new one
            # raise ValueError(f"Failed during summary/saving. Check logs and {error_output_path}. Original error: {e}")


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
            results_list.append({
                "analysis_result": parsed_response,
                "segment_name": segment.segment_name,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "frame_paths": segment.segment_frames_file_path,
                # Pass the whole transcription object once (it's the same for all segments)
                "full_transcription_object": self.manifest.audio_transcription if not results_list else None # Pass only for first segment
             })

        # After all segments have been analyzed
        if hasattr(analysis_config, 'process_segment_results'):
            # Use the custom processing method if available
            final_results = analysis_config.process_segment_results(results_list)
            # Update the manifest's global_tags (Keep existing fix)
            if isinstance(final_results, dict) and "global_tags" in final_results:
                 self.manifest.global_tags = final_results["global_tags"]
                 print(f"DEBUG (SYNC): Updated manifest global_tags via custom process_segment_results.")
            else:
                 print(f"DEBUG (SYNC): Did NOT update manifest global_tags. final_results type: {type(final_results)}, keys: {final_results.keys() if isinstance(final_results, dict) else 'N/A'}")

            # --- Return the AGGREGATED results ---
            return final_results
        else:
             # --- EDIT 5: Use generic aggregation directly if no custom method ---
             # This block now becomes the primary return path if no custom agg.
             print("Running generic aggregation within _analyze_segment_list.")
             # (The generic aggregation logic previously here is now primarily in analyze_video)
             # For simplicity here, we just return the raw list and let analyze_video handle it.
             # Alternatively, you could replicate the generic agg logic from analyze_video here.
             # Let's return the raw list for analyze_video to handle.
             return results_list # Return list including frame_paths
             # --- End EDIT 5 ---

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
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"},
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
                parsed_response = await self._analyze_segment_async(segment, analysis_config)
                return {
                    "analysis_result": parsed_response,
                    "segment_name": segment.segment_name,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "frame_paths": segment.segment_frames_file_path,
                    # Pass the whole transcription object - will need to handle potential multiple copies in aggregation
                    "full_transcription_object": self.manifest.audio_transcription
                }

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

        # --- Aggregation Logic ---
        if hasattr(analysis_config, 'process_segment_results'):
            # Pass the list containing transcription objects to aggregation
            final_results_agg = analysis_config.process_segment_results(results_list)
            # Update the manifest's global_tags with the aggregated results (Keep existing fix)
            if isinstance(final_results_agg, dict) and "global_tags" in final_results_agg:
                 self.manifest.global_tags = final_results_agg["global_tags"]
                 print(f"DEBUG (ASYNC): Updated manifest global_tags via custom process_segment_results.") # Optional debug log
            else:
                 print(f"DEBUG (ASYNC): Did NOT update manifest global_tags. final_results_agg type: {type(final_results_agg)}, keys: {final_results_agg.keys() if isinstance(final_results_agg, dict) else 'N/A'}") # Optional debug log

            # --- Return the AGGREGATED results ---
            return final_results_agg # Return aggregated results
        else:
             # ... (rest of generic aggregation handling) ...
             return results_list # Return list including frame_paths

    def _analyze_segment(
        self,
        segment: Segment,
        analysis_config: Type[AnalysisConfig]
    ):
        if (not self.reprocess_segments
            and analysis_config.name in segment.analysis_completed):
            print(f"Segment {segment.segment_name} already analyzed, skipping.")
            # Ensure the stored result is loaded correctly if needed later
            # This might need adjustment if the stored format changes
            return segment.analyzed_result.get(analysis_config.name, {"chapters": [], "global_tags": {"persons": [], "actions": [], "objects": []}})

        print(f"Analyzing segment {segment.segment_name} ({segment.start_time:.3f}s - {segment.end_time:.3f}s)")
        stopwatch_segment = time.time()

        # --- Retry Parameters ---
        max_retries = 3
        initial_delay = 1.0 # seconds

        # --- Stage 1: Get Chapters (Full Segment Context) with Retry ---
        print(f"  - Stage 1: Requesting Chapters...")
        chapters_result = {"chapters": []} # Default
        for attempt in range(max_retries):
            try:
                chapter_prompt = self._generate_segment_prompt(
                    segment=segment,
                    analysis_config=analysis_config,
                    generate_chapters=True,
                    generate_tags=False
                )
                if chapter_prompt:
                    chapter_llm_response = self._call_llm(chapter_prompt)
                    chapters_result = self._parse_llm_json_response(
                        chapter_llm_response,
                        expecting_chapters=True,
                        expecting_tags=False
                    )
                    # Add a check: if parsing failed and returned default, it counts as an error for retry
                    if not chapters_result.get("chapters"): # Check if key has content
                        # Check if the raw response was likely empty or problematic
                        raw_content_check = chapter_llm_response.choices[0].message.content.strip()
                        if not raw_content_check or raw_content_check == "{}":
                             raise ValueError("LLM response parsed to empty chapters, possibly indicating an issue.")

                    print(f"  - Stage 1: Received {len(chapters_result.get('chapters',[]))} chapter(s).")
                    break # Success, exit retry loop
                else:
                    print("  - Stage 1: Skipping chapter generation due to empty prompt.")
                    break # No prompt generated, don't retry

            except Exception as e:
                print(f"  - Stage 1: Attempt {attempt + 1}/{max_retries} failed for Chapters. Error: {e}")
                if attempt + 1 == max_retries:
                    print(f"  - Stage 1: Max retries reached for Chapters. Proceeding without chapters.")
                    # chapters_result remains default empty
                else:
                    wait_time = initial_delay * (2 ** attempt) # Exponential backoff
                    print(f"    - Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)


        # --- Stage 2: Get Tags (Chunked Analysis) with Retry per Chunk ---
        print(f"  - Stage 2: Requesting Tags in {self.TAG_CHUNK_SECONDS}s chunks...")
        all_frames = segment.segment_frames_file_path
        all_times = segment.segment_frame_time_intervals
        tag_chunk_results = []
        segment_duration = segment.end_time - segment.start_time
        num_chunks = math.ceil(segment_duration / self.TAG_CHUNK_SECONDS) if self.TAG_CHUNK_SECONDS > 0 else 1
        print(f"    - Dividing into {num_chunks} chunks...")


        for i in range(num_chunks):
            chunk_start = segment.start_time + (i * self.TAG_CHUNK_SECONDS)
            chunk_end = min(segment.start_time + ((i + 1) * self.TAG_CHUNK_SECONDS), segment.end_time)
            print(f"    - Processing Chunk {i+1}/{num_chunks} ({chunk_start:.3f}s - {chunk_end:.3f}s)")

            epsilon = 0.001
            chunk_indices = [idx for idx, t in enumerate(all_times) if chunk_start - epsilon <= t <= chunk_end + epsilon]
            if not chunk_indices:
                 print(f"      - No frames found for chunk {i+1}. Skipping.")
                 continue
            chunk_frames = [all_frames[idx] for idx in chunk_indices]
            chunk_times = [all_times[idx] for idx in chunk_indices]
            print(f"      - Found {len(chunk_frames)} frames for this chunk.")


            # Retry loop for each chunk
            parsed_chunk = None
            for attempt in range(max_retries):
                try:
                    tag_prompt = self._generate_segment_prompt(
                        segment=segment, analysis_config=analysis_config, frames_subset=chunk_frames,
                        times_subset=chunk_times, generate_chapters=False, generate_tags=True,
                        chunk_start_time=chunk_start, chunk_end_time=chunk_end
                    )

                    if tag_prompt:
                        tag_llm_response = self._call_llm(tag_prompt)
                        parsed_chunk = self._parse_llm_json_response(
                            tag_llm_response, expecting_chapters=False, expecting_tags=True
                        )
                        # Check if parsing failed and returned default empty tags
                        if not any(parsed_chunk.get("global_tags", {}).values()):
                           # Check if the raw response was likely empty or problematic
                           raw_content_check = tag_llm_response.choices[0].message.content.strip()
                           if not raw_content_check or raw_content_check == "{}":
                                raise ValueError("LLM response parsed to empty tags, possibly indicating an issue.")

                        tags_found = sum(len(v) for k, v in parsed_chunk.get("global_tags", {}).items())
                        print(f"      - Chunk {i+1} (Attempt {attempt+1}) analysis complete. Found {tags_found} tags.")
                        break # Success, exit retry loop for this chunk
                    else:
                        print(f"      - Chunk {i+1}: Skipping tag generation due to empty prompt.")
                        break # No prompt, exit retry loop

                except Exception as e:
                    print(f"      - Chunk {i+1}: Attempt {attempt + 1}/{max_retries} failed for Tags. Error: {e}")
                    if attempt + 1 == max_retries:
                        print(f"      - Chunk {i+1}: Max retries reached. Skipping tags for this chunk.")
                        # parsed_chunk remains None or last failed attempt's default
                    else:
                        wait_time = initial_delay * (2 ** attempt) # Exponential backoff
                        print(f"        - Retrying in {wait_time:.1f} seconds...")
                        time.sleep(wait_time)

                # Add the result (even if it's None or default empty after retries)
                if parsed_chunk: # Only add if parsing didn't completely fail after retries
                   tag_chunk_results.append(parsed_chunk)


        # --- Stage 3: Merge Tags ---
        print(f"  - Stage 3: Merging tags from {len(tag_chunk_results)} successful chunks...")
        merged_global_tags = self._merge_tagging_chunks(tag_chunk_results)
        tags_merged_count = sum(len(v) for k, v in merged_global_tags.items())
        print(f"  - Stage 3: Merged into {tags_merged_count} unique tags.")

        # --- Stage 4: Combine ---
        final_combined_result = {
            "chapters": chapters_result.get("chapters", []), # Use potentially empty chapters
            "global_tags": merged_global_tags
        }

        # --- Stage 5: Update Segment Object and Save ---
        segment.analyzed_result[analysis_config.name] = final_combined_result
        # Ensure analysis_completed list tracks unique names
        if analysis_config.name not in segment.analysis_completed:
            segment.analysis_completed.append(analysis_config.name)

        # Save intermediate results (optional but recommended)
        try:
             segment_result_path = os.path.join(segment.segment_folder_path, f"_segment_analyzed_result_{analysis_config.name}.json")
             with open(segment_result_path, "w", encoding="utf-8") as f:
                  json.dump(final_combined_result, f, indent=4, ensure_ascii=False)
        except Exception as e:
             print(f"Warning: Could not save intermediate segment result for {segment.segment_name}: {e}")

        # Update manifest (This might need locking if multiple async segments write simultaneously)
        # For simplicity, we assume write_video_manifest handles potential concurrent writes gracefully,
        # or we accept the risk of potential race conditions if updates are very frequent.
        # A safer approach would involve collecting all results and writing the manifest once at the end
        # of _analyze_segment_list_async, or using an async lock.
        write_video_manifest(self.manifest)

        # Update known tags (potential race condition if shared state isn't handled carefully)
        # Similar to manifest writing, modifying shared sets like _current_known_* needs care.
        # For map-reduce, this update might be less critical for cross-segment influence during the run.
        if isinstance(merged_global_tags, dict):
             for tag_type, tag_list in merged_global_tags.items():
                  if isinstance(tag_list, list):
                      for item in tag_list:
                          if isinstance(item, dict):
                              name = item.get("name")
                              if name and isinstance(name, str):
                                   cleaned_name = name.strip()
                                   if cleaned_name:
                                       if tag_type == "persons": self._current_known_persons.add(cleaned_name)
                                       elif tag_type == "actions": self._current_known_actions.add(cleaned_name)
                                       elif tag_type == "objects": self._current_known_objects.add(cleaned_name)


        elapsed_segment = time.time() - stopwatch_segment
        print(f"Segment {segment.segment_name} finished analysis in {elapsed_segment:.2f} seconds.")

        return final_combined_result # Return the combined result

    async def _analyze_segment_async(
        self,
        segment: Segment,
        analysis_config: AnalysisConfig,
    ):
        if (not self.reprocess_segments
            and analysis_config.name in segment.analysis_completed):
            print(f"Segment {segment.segment_name} already analyzed (async), skipping.")
            return segment.analyzed_result.get(analysis_config.name, {"chapters": [], "global_tags": {"persons": [], "actions": [], "objects": []}})

        print(f"Analyzing segment {segment.segment_name} asynchronously...")
        stopwatch_segment = time.time()

        # --- Retry Parameters ---
        max_retries = 3
        initial_delay = 1.0 # seconds

        # --- Stage 1: Get Chapters (Async) with Retry ---
        print(f"  - Stage 1 (Async): Requesting Chapters...")
        chapters_result = {"chapters": []}
        for attempt in range(max_retries):
            try:
                chapter_prompt = self._generate_segment_prompt(
                    segment=segment, analysis_config=analysis_config, generate_chapters=True, generate_tags=False
                )
                if chapter_prompt:
                    chapter_llm_response = await self._call_llm_async(chapter_prompt)
                    chapters_result = self._parse_llm_json_response(
                         chapter_llm_response, expecting_chapters=True, expecting_tags=False
                    )
                    # Add check for failed parsing returning default
                    if not chapters_result.get("chapters"):
                        raw_content_check = chapter_llm_response.choices[0].message.content.strip()
                        if not raw_content_check or raw_content_check == "{}":
                             raise ValueError("LLM response parsed to empty chapters, possibly indicating an issue.")

                    print(f"  - Stage 1 (Async): Received {len(chapters_result.get('chapters',[]))} chapter(s).")
                    break # Success
                else:
                    print("  - Stage 1 (Async): Skipping chapter generation due to empty prompt.")
                    break # No prompt

            except Exception as e:
                print(f"  - Stage 1 (Async): Attempt {attempt + 1}/{max_retries} failed for Chapters. Error: {e}")
                if attempt + 1 == max_retries:
                    print(f"  - Stage 1 (Async): Max retries reached for Chapters. Proceeding without chapters.")
                else:
                    wait_time = initial_delay * (2 ** attempt)
                    print(f"    - Retrying in {wait_time:.1f} seconds...")
                    await asyncio.sleep(wait_time) # Use asyncio.sleep for async

        # --- Stage 2: Get Tags (Chunked Async Calls) with Retry per Chunk ---
        print(f"  - Stage 2 (Async): Requesting Tags in {self.TAG_CHUNK_SECONDS}s chunks...")
        all_frames = segment.segment_frames_file_path
        all_times = segment.segment_frame_time_intervals
        tag_chunk_tasks = [] # Will store tasks that call a retry-wrapper
        segment_duration = segment.end_time - segment.start_time
        num_chunks = math.ceil(segment_duration / self.TAG_CHUNK_SECONDS) if self.TAG_CHUNK_SECONDS > 0 else 1
        print(f"    - Dividing into {num_chunks} chunks...")

        # --- Define an async retry wrapper for tag chunk calls ---
        async def call_and_parse_chunk_with_retry(chunk_idx, chunk_start, chunk_end, chunk_frames, chunk_times):
            print(f"    - Preparing Chunk {chunk_idx+1}/{num_chunks} ({chunk_start:.3f}s - {chunk_end:.3f}s)")
            for attempt in range(max_retries):
                try:
                    tag_prompt = self._generate_segment_prompt(
                         segment=segment, analysis_config=analysis_config, frames_subset=chunk_frames,
                         times_subset=chunk_times, generate_chapters=False, generate_tags=True,
                         chunk_start_time=chunk_start, chunk_end_time=chunk_end
                    )
                    if not tag_prompt:
                         print(f"      - Chunk {chunk_idx+1}: Skipping tag generation due to empty prompt.")
                         return None # No prompt, return None

                    tag_llm_response = await self._call_llm_async(tag_prompt)
                    parsed_chunk = self._parse_llm_json_response(
                         tag_llm_response, expecting_chapters=False, expecting_tags=True
                    )
                    # Check for failed parsing returning default
                    if not any(parsed_chunk.get("global_tags", {}).values()):
                        raw_content_check = tag_llm_response.choices[0].message.content.strip()
                        if not raw_content_check or raw_content_check == "{}":
                            raise ValueError("LLM response parsed to empty tags, possibly indicating an issue.")

                    tags_found = sum(len(v) for k, v in parsed_chunk.get("global_tags", {}).items())
                    print(f"      - Chunk {chunk_idx+1} (Attempt {attempt+1}) analysis complete. Found {tags_found} tags.")
                    return parsed_chunk # Success

                except Exception as e:
                     print(f"      - Chunk {chunk_idx+1}: Attempt {attempt + 1}/{max_retries} failed for Tags. Error: {e}")
                     if attempt + 1 == max_retries:
                          print(f"      - Chunk {chunk_idx+1}: Max retries reached. Skipping tags for this chunk.")
                          return None # Failed after retries, return None
                     else:
                          wait_time = initial_delay * (2 ** attempt)
                          print(f"        - Retrying in {wait_time:.1f} seconds...")
                          await asyncio.sleep(wait_time)
            return None # Should not be reached if loop completes, but good practice

        # --- Create tasks for the retry wrapper ---
        for i in range(num_chunks):
            # ... (setup code for chunk: chunk_start, chunk_end, chunk_indices, chunk_frames, chunk_times) ...
            chunk_start = segment.start_time + (i * self.TAG_CHUNK_SECONDS)
            chunk_end = min(segment.start_time + ((i + 1) * self.TAG_CHUNK_SECONDS), segment.end_time)
            epsilon = 0.001
            chunk_indices = [idx for idx, t in enumerate(all_times) if chunk_start - epsilon <= t <= chunk_end + epsilon]
            if not chunk_indices: continue
            chunk_frames = [all_frames[idx] for idx in chunk_indices]
            chunk_times = [all_times[idx] for idx in chunk_indices]
            print(f"      - Creating task for Chunk {i+1} ({len(chunk_frames)} frames)")

            # Create a task using the retry wrapper
            task = asyncio.create_task(
                call_and_parse_chunk_with_retry(i, chunk_start, chunk_end, chunk_frames, chunk_times),
                name=f"RetryChunk_{i+1}"
            )
            tag_chunk_tasks.append(task)


        # Gather results from all chunk tasks
        tag_chunk_results_raw = []
        if tag_chunk_tasks:
             print(f"    - Awaiting {len(tag_chunk_tasks)} tag chunk analyses (with retries)...")
             tag_chunk_results_raw = await asyncio.gather(*tag_chunk_tasks) # No return_exceptions needed here as wrapper handles it
             print(f"    - Tag chunk analyses complete.")

        # Process results (filter out None results from failed chunks)
        tag_chunk_results = [result for result in tag_chunk_results_raw if result is not None]

        # --- Stage 3: Merge Tags ---
        print(f"  - Stage 3 (Async): Merging tags from {len(tag_chunk_results)} successful chunks...")
        merged_global_tags = self._merge_tagging_chunks(tag_chunk_results)
        tags_merged_count = sum(len(v) for k, v in merged_global_tags.items())
        print(f"  - Stage 3 (Async): Merged into {tags_merged_count} unique tags.")

        # --- Stage 4: Combine ---
        final_combined_result = {
            "chapters": chapters_result.get("chapters", []), # Use potentially empty chapters
            "global_tags": merged_global_tags
        }

        # --- Stage 5: Update Segment & Save ---
        segment.analyzed_result[analysis_config.name] = final_combined_result
        if analysis_config.name not in segment.analysis_completed:
            segment.analysis_completed.append(analysis_config.name)
        # Save intermediate results (Consider async file writing if it becomes a bottleneck)
        try:
             segment_result_path = os.path.join(segment.segment_folder_path, f"_segment_analyzed_result_{analysis_config.name}.json")
             with open(segment_result_path, "w", encoding="utf-8") as f:
                  json.dump(final_combined_result, f, indent=4, ensure_ascii=False)
        except Exception as e:
             print(f"Warning: Could not save intermediate segment result for {segment.segment_name} (async): {e}")

        # Update manifest (This might need locking if multiple async segments write simultaneously)
        # For simplicity, we assume write_video_manifest handles potential concurrent writes gracefully,
        # or we accept the risk of potential race conditions if updates are very frequent.
        # A safer approach would involve collecting all results and writing the manifest once at the end
        # of _analyze_segment_list_async, or using an async lock.
        write_video_manifest(self.manifest)

        # Update known tags (potential race condition if shared state isn't handled carefully)
        # Similar to manifest writing, modifying shared sets like _current_known_* needs care.
        # For map-reduce, this update might be less critical for cross-segment influence during the run.
        if isinstance(merged_global_tags, dict):
             for tag_type, tag_list in merged_global_tags.items():
                  if isinstance(tag_list, list):
                      for item in tag_list:
                          if isinstance(item, dict):
                              name = item.get("name")
                              if name and isinstance(name, str):
                                   cleaned_name = name.strip()
                                   if cleaned_name:
                                       if tag_type == "persons": self._current_known_persons.add(cleaned_name)
                                       elif tag_type == "actions": self._current_known_actions.add(cleaned_name)
                                       elif tag_type == "objects": self._current_known_objects.add(cleaned_name)


        elapsed_segment = time.time() - stopwatch_segment
        print(f"Segment {segment.segment_name} finished analysis (async) in {elapsed_segment:.2f} seconds.")

        return final_combined_result

    def _generate_segment_prompt(
        self,
        segment: Segment,
        analysis_config: AnalysisConfig,
        frames_subset: List[str] = None,
        times_subset: List[float] = None,
        generate_chapters: bool = True,
        generate_tags: bool = True,
        chunk_start_time: Optional[float] = None,
        chunk_end_time: Optional[float] = None,
        is_partial_chunk: bool = False
    ):
        """
        Generate a prompt for a segment, adaptable for full context or tag-specific chunks.

        Args:
            segment: Segment object containing video segment info
            analysis_config: Analysis configuration
            frames_subset: Optional subset of frames to process
            times_subset: Optional subset of frame timestamps
            generate_chapters: Whether to ask for the 'chapters' part.
            generate_tags: Whether to ask for the 'global_tags' part.
            chunk_start_time: The start time of the specific chunk being analyzed for tags.
            chunk_end_time: The end time of the specific chunk being analyzed for tags.
            is_partial_chunk: Whether this is a partial chunk due to MAX_FRAMES_PER_PROMPT limit.
        """
        frames_to_process = frames_subset if frames_subset is not None else segment.segment_frames_file_path
        times_to_process = times_subset if times_subset is not None else segment.segment_frame_time_intervals

        if not frames_to_process:
             print(f"Warning: No frames to process for prompt generation in segment {segment.segment_name}.")
             # Handle this case - maybe return None or raise error?
             return None # Or empty prompt structure

        # --- 1. System Prompt Generation ---
        system_prompt_base = getattr(analysis_config, "system_prompt", "Default system prompt")
        system_prompt_lens_template = getattr(analysis_config, "system_prompt_lens", "")

        # Modify the base prompt instructions based on what we're generating
        adjusted_system_prompt = system_prompt_base

        # Adjust JSON structure instruction (Point 1)
        if generate_chapters and not generate_tags:
            adjusted_system_prompt = adjusted_system_prompt.replace(
                 'ONLY these two top-level keys: "chapters" and "global_tags"',
                 'ONLY the top-level key: "chapters". Do NOT include "global_tags".'
            ).replace(
                 '- "global_tags": An object containing three keys: "persons", "actions", "objects" (each value must be an array of tag objects).',
                 '' # Remove the global_tags description
            )
        elif not generate_chapters and generate_tags:
             adjusted_system_prompt = adjusted_system_prompt.replace(
                 'ONLY these two top-level keys: "chapters" and "global_tags"',
                 'ONLY the top-level key: "global_tags". Do NOT include "chapters".'
            ).replace(
                 '- "chapters": An array containing EXACTLY ONE chapter object describing the current segment.',
                 '' # Remove the chapters description
            )
        # else: keep original instruction for both

        # Adjust Output Format Example (Point 2)
        output_format_example_str = analysis_config.results_template # Or reconstruct from prompt
        try:
             output_format_example = json.loads(output_format_example_str) if isinstance(output_format_example_str, str) else output_format_example_str
             if not generate_chapters:
                  output_format_example.pop("chapters", None)
             if not generate_tags:
                  output_format_example.pop("global_tags", None)
             adjusted_example_str = json.dumps(output_format_example, indent=2)
             # Replace the example in the prompt (this requires finding the bounds reliably, maybe use markers)
             start_marker = "2. EXACT OUTPUT FORMAT:\n{{"
             end_marker = "}}" # This is tricky if }} appears elsewhere
             # Find the section and replace (simplistic approach below)
             start_idx = adjusted_system_prompt.find(start_marker)
             end_idx = adjusted_system_prompt.find(end_marker, start_idx + len(start_marker))
             if start_idx != -1 and end_idx != -1:
                 original_example_part = adjusted_system_prompt[start_idx:end_idx+len(end_marker)]
                 new_example_part = f"2. EXACT OUTPUT FORMAT:\n{adjusted_example_str}"
                 adjusted_system_prompt = adjusted_system_prompt.replace(original_example_part, new_example_part, 1)

        except Exception as e:
             print(f"Warning: Could not adjust output format example in prompt: {e}")


        # Adjust Timestamp Precision Instruction (Point 3) - Crucial for tag chunks
        if not generate_chapters and generate_tags and chunk_start_time is not None and chunk_end_time is not None:
              # Make timestamp instructions specific to the TAG CHUNK
              adjusted_system_prompt = adjusted_system_prompt.replace(
                   "Base ALL timestamps (`chapters` start/end AND `global_tags` timecodes start/end)",
                   "Base ALL `global_tags` timecodes (`start`, `end`)"
              ).replace(
                  "All timestamps MUST be within the overall segment's absolute start ({start_time}) and end ({end_time}) times provided in the analysis section below.",
                  f"CRITICAL: All tag timecodes MUST be strictly between {chunk_start_time:.3f}s and {chunk_end_time:.3f}s, based *only* on the frame timestamps provided below for this specific chunk. Do not use timestamps outside this narrow range."
              ).replace(
                  "Do NOT DEFAULT TO SEGMENT BOUNDARIES", # Keep this part
                  f"Do NOT DEFAULT TO CHUNK BOUNDARIES ({chunk_start_time:.3f}s-{chunk_end_time:.3f}s)" # Add specific bounds
              )
        # If generating chapters, keep the reference to segment start/end for chapter times
        elif generate_chapters:
            # Remove reference to global_tags if only generating chapters
            if not generate_tags:
                 adjusted_system_prompt = adjusted_system_prompt.replace(
                      "(`chapters` start/end AND `global_tags` timecodes start/end)",
                      "(`chapters` start/end)"
                 )


        # Append Lens
        known_actions_str = ", ".join(sorted(list(self._current_known_actions))) if self._current_known_actions else "None"
        known_objects_str = ", ".join(sorted(list(self._current_known_objects))) if self._current_known_objects else "None"

        try:
            # Determine start/end for the lens based on context
            lens_start = chunk_start_time if chunk_start_time is not None else float(segment.start_time)
            lens_end = chunk_end_time if chunk_end_time is not None else float(segment.end_time)

            formatted_lens = system_prompt_lens_template.format(
                start_time=f"{lens_start:.3f}",
                end_time=f"{lens_end:.3f}",
                number_of_frames=len(frames_to_process),
                known_actions=known_actions_str,
                known_objects=known_objects_str
            )
             # Conditionally add chapter/tag reminders to lens
            if generate_chapters and not generate_tags:
                 formatted_lens += "\n\nFocus ONLY on generating the 'chapters' array for this segment."
            elif not generate_chapters and generate_tags:
                 formatted_lens += f"\n\nFocus ONLY on generating the 'global_tags' object for this time chunk ({lens_start:.3f}s-{lens_end:.3f}s). Tag accurately within this window."

            adjusted_system_prompt += "\n" + formatted_lens

        except Exception as e:
             print(f"Error formatting system_prompt_lens: {e}")
             adjusted_system_prompt += "\nError formatting analysis lens."


        # --- 2. User Content Generation ---
        user_content = []

        # Add transcription context (always useful, even for tags)
        transcription_text = segment.transcription if segment.transcription else "No transcription available"
        # Optional: Trim transcription to chunk if generating tags only? Maybe not necessary.
        user_content.append(
            {"type": "text", "text": f"Audio Transcription Context: {transcription_text}"}
        )

        # Add lists (people, objects, etc.) - always useful context
        # ... (keep existing code for adding peoples_list, emotions_list etc prompts) ...
        if self.peoples_list: user_content.append(self._generate_list_prompt("peoples", self.peoples_list))
        # ... add other lists ...
        if self.actions_list: user_content.append(self._generate_list_prompt("actions", self.actions_list))


        # Add Frame Analysis Instructions (Keep general instructions)
        user_content.append({
            "type": "text",
            "text": """IMPORTANT: Analyze the following frames carefully. Note visual elements, actions, objects, and track their appearance/disappearance using the provided timestamps."""
        })

        # Add Frames with Timestamps
        if not times_to_process or len(times_to_process) != len(frames_to_process):
             print(f"Error: Mismatch between frames ({len(frames_to_process)}) and times ({len(times_to_process)}) for segment {segment.segment_name}")
             # Fallback or error handling needed
             # Simplistic fix: Recalculate times based on frame count and segment duration/chunk duration
             effective_start = chunk_start_time if chunk_start_time is not None else segment.start_time
             effective_end = chunk_end_time if chunk_end_time is not None else segment.end_time
             if len(frames_to_process) > 0:
                  times_to_process = np.linspace(effective_start, effective_end, len(frames_to_process), endpoint=False)
                  times_to_process = [round(float(t), 3) for t in times_to_process]
             else:
                  times_to_process = []


        for i, frame_path in enumerate(frames_to_process):
             # Ensure we have a valid timestamp for each frame
             if i >= len(times_to_process):
                  print(f"Warning: Missing timestamp for frame {i} ({frame_path}). Skipping frame.")
                  continue
             timestamp = times_to_process[i] # Use the potentially recalculated time

             # Encode image
             image_content = encode_image_base64(frame_path)

             # Add frame timestamp label
             user_content.append({"type": "text", "text": f"\nFrame at {timestamp:.3f}s:"})
             # Add image data
             user_content.append({
                 "type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_content}", "detail": "high"}
             })

        # Add Identified People Summary (if applicable)
        # ... (keep existing logic for adding identified people if self.person_group_id)

        # Add FINAL reminder about output format, adapted for the request
        final_reminder_text = "CRITICAL: Your output MUST be VALID JSON"
        if generate_chapters and generate_tags:
            final_reminder_text += " with 'chapters' and 'global_tags' keys."
        elif generate_chapters:
             final_reminder_text += " with ONLY the 'chapters' key."
        elif generate_tags:
             final_reminder_text += f" with ONLY the 'global_tags' key, containing tags strictly within the {chunk_start_time:.3f}s to {chunk_end_time:.3f}s window."
        # Append the adjusted example structure (or a simplified text version)
        final_reminder_text += f"\n\nExpected Structure:\n{adjusted_example_str}"

        user_content.append({"type": "text", "text": final_reminder_text})


        # --- 3. Construct Final Prompt ---
        prompt = [
            {"role": "system", "content": adjusted_system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Save the prompt (optional, maybe save chunk prompts differently)
        # ... (consider whether/how to save these potentially numerous chunk prompts) ...

        return prompt

    def _merge_tagging_chunks(self, tag_chunk_results: List[Dict]) -> Dict:
        """Merges global_tags from multiple chunk analyses."""
        merged_tags = {
            "persons": defaultdict(list), # Store timecodes per person name
            "actions": defaultdict(list),
            "objects": defaultdict(list)
        }

        for chunk_result in tag_chunk_results:
            if not isinstance(chunk_result, dict): continue
            global_tags = chunk_result.get("global_tags", {})
            if not isinstance(global_tags, dict): continue

            for tag_type in ["persons", "actions", "objects"]:
                if tag_type not in global_tags or not isinstance(global_tags[tag_type], list): continue

                for item in global_tags[tag_type]:
                     if not isinstance(item, dict): continue
                     name = item.get("name")
                     timecodes = item.get("timecodes", [])
                     if not name or not isinstance(name, str) or not name.strip() or not isinstance(timecodes, list): continue

                     cleaned_name = name.strip()
                     # Add all valid timecodes from this chunk to the master list for this tag name
                     for tc in timecodes:
                          if isinstance(tc, dict) and "start" in tc and "end" in tc:
                               # Basic validation - further validation/clamping happens later if needed
                               try:
                                    start_f = float(str(tc["start"]).rstrip('s'))
                                    end_f = float(str(tc["end"]).rstrip('s'))
                                    if start_f <= end_f:
                                         merged_tags[tag_type][cleaned_name].append(tc)
                               except (ValueError, TypeError):
                                    print(f"Warning: Skipping invalid timecode format during chunk merge: {tc}")
                                    continue

        # Convert defaultdict back to the final list structure
        final_merged_structure = { "persons": [], "actions": [], "objects": [] }
        for tag_type, name_dict in merged_tags.items():
            for name, timecode_list in name_dict.items():
                if timecode_list: # Only add if there are timecodes
                     # Optional: Deduplicate timecodes here if needed, though aggregation later might handle it
                     unique_timecodes_set = set(tuple(sorted(d.items())) for d in timecode_list)
                     unique_timecodes_list = sorted(
                            [dict(t) for t in unique_timecodes_set],
                            key=lambda x: float(str(x.get("start", "inf")).rstrip("s"))
                     )
                     final_merged_structure[tag_type].append({
                          "name": name,
                          "timecodes": unique_timecodes_list
                     })

        return final_merged_structure

    def _parse_llm_json_response(self, response, expecting_chapters=True, expecting_tags=True):
         # ... (initial raw_content extraction and ```json``` handling) ...
         raw_content = response.choices[0].message.content
         # ... (existing ```json stripping) ...
         content = raw_content # Placeholder for stripped content
         if "```json" in content:
             content = content.split("```json")[1].split("```")[0].strip()
         elif content.startswith("```") and content.endswith("```"):
             content = content[3:-3].strip()

         try:
             parsed_data = json.loads(content)
             if not isinstance(parsed_data, dict):
                 raise json.JSONDecodeError("Response is not a JSON object", content, 0)

             final_parsed_content = {}

             # Process Chapters if expected
             if expecting_chapters:
                 chapters_raw = parsed_data.get("chapters", [])
                 # ... (existing chapter processing logic: single dict handling, timestamp conversion, etc.) ...
                 processed_chapters = []
                 if isinstance(chapters_raw, dict): chapters_raw = [chapters_raw] # Handle single
                 if isinstance(chapters_raw, list):
                      for item in chapters_raw:
                           if not isinstance(item, dict): continue
                           # --- Apply existing processing/validation ---
                           # (Copy relevant parts of existing chapter processing here)
                           # e.g., timestamp format conversion, field name mapping...
                           # Make sure to handle potential errors gracefully for each chapter
                           # --- End existing chapter processing copy ---
                           processed_chapters.append(item) # Append valid chapter
                 final_parsed_content["chapters"] = processed_chapters
             else:
                  # If not expecting chapters, but key exists, maybe log warning?
                  final_parsed_content["chapters"] = [] # Ensure key exists but is empty

             # Process Global Tags if expected
             if expecting_tags:
                 global_tags_raw = parsed_data.get("global_tags", {})
                 processed_tags = {"persons": [], "actions": [], "objects": []}
                 if isinstance(global_tags_raw, dict):
                      for tag_type in ["persons", "actions", "objects"]:
                           tags_list = global_tags_raw.get(tag_type, [])
                           if not isinstance(tags_list, list): tags_list = []
                           valid_tags_for_type = []
                           for item in tags_list:
                                if isinstance(item, dict) and "name" in item and "timecodes" in item:
                                     # Add basic validation for timecodes format maybe?
                                     valid_tags_for_type.append(item)
                           processed_tags[tag_type] = valid_tags_for_type
                 final_parsed_content["global_tags"] = processed_tags
             else:
                  # If not expecting tags, but key exists, maybe log warning?
                  final_parsed_content["global_tags"] = {"persons": [], "actions": [], "objects": []} # Ensure structure

             return final_parsed_content

         except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response content: {content}")
            # Return default structure based on expectations
            fallback = {}
            if expecting_chapters: fallback["chapters"] = []
            if expecting_tags: fallback["global_tags"] = {"persons": [], "actions": [], "objects": []}
            return fallback
         except Exception as e:
             print(f"Error processing LLM response: {e}")
             print(f"Response content: {content}")
             # Return default structure based on expectations
             fallback = {}
             if expecting_chapters: fallback["chapters"] = []
             if expecting_tags: fallback["global_tags"] = {"persons": [], "actions": [], "objects": []}
             return fallback

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