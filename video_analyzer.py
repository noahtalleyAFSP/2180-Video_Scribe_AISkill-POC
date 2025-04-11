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
import re

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
    TAG_CHUNK_SECONDS: float = 1.0 # New: Duration for tag analysis chunks

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
        copyright_json_str: Optional[str] = None,
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

        # --- Determine if custom aggregation exists ---
        has_custom_aggregation = hasattr(analysis_config, 'process_segment_results')
        # --- EDIT: Rename to reflect potential structure ---
        processed_results = None # Will hold the dict from process_segment_results

        # Analyze videos using the mapreduce sequence
        if analysis_config.analysis_sequence == "mapreduce":
            print(f"Populating prompts for each segment")

            if run_async:
                print("Running analysis asynchronously")
                nest_asyncio.apply()
                processed_results = asyncio.run(
                    self._analyze_segment_list_async(
                        analysis_config,
                        max_concurrent_tasks=max_concurrent_tasks,
                        copyright_json_str=copyright_json_str
                    )
                )
            else:
                print("Running analysis.")
                processed_results = self._analyze_segment_list(analysis_config, copyright_json_str=copyright_json_str) # Pass copyright str

        # For refine-style analyses that need to be run sequentially
        elif analysis_config.analysis_sequence == "refine":
            print(f"Analyzing segments sequentially with refinement")
            # Assuming _analyze_segment_list_sequentially returns a list suitable for generic aggregation
            segment_results_list = self._analyze_segment_list_sequentially(analysis_config)
            has_custom_aggregation = False # Force generic aggregation for refine unless specified otherwise
            # --- EDIT: Need to run generic aggregation if refine doesn't have custom ---
            # The structure returned by _analyze_segment_list_sequentially is assumed to be
            # a list of results per segment, similar to what generic aggregation expects.
            # We will apply generic aggregation to this list later.
            # For now, store the list.
            processed_results = segment_results_list # Store list for later generic agg

        else:
            raise ValueError(
                f"You have provided an AnalyisConfig with a analysis_sequence that has not yet been implmented: {analysis_config.analysis_sequence}"
            )

        # --- EDIT: Standardize final_results structure & Update Manifest ---
        final_results = {} # Initialize the dictionary to be saved as JSON

        if has_custom_aggregation and isinstance(processed_results, dict):
            print("Using results from custom aggregation method.")
            # Assume the custom method returns the final structure directly
            # (e.g., {'actionSummary': {...}, 'transcriptionDetails': {...}})
            final_results = processed_results

            # --- Update manifest.global_tags from the actionSummary structure ---
            if "actionSummary" in final_results:
                action_summary_content = final_results.get("actionSummary", {})
                # Construct the global_tags structure expected by the manifest
                # Use camelCase keys from action_summary_content, map to plural keys for manifest
                manifest_global_tags = {
                     "persons": action_summary_content.get("person", []), # person -> persons
                     "actions": action_summary_content.get("action", []), # action -> actions
                     "objects": action_summary_content.get("object", [])  # object -> objects
                }
                self.manifest.global_tags = manifest_global_tags
                print("DEBUG: Updated self.manifest.global_tags using data from actionSummary (mapped to plural keys).")
            else:
                # Handle old custom structure (if needed) - unlikely with ActionSummary config
                print("Warning: Custom aggregation result missing 'actionSummary'. Manifest global_tags not updated.")

        elif isinstance(processed_results, list):
            # --- Apply Generic Aggregation to the list of segment results ---
            # This path handles non-custom aggregation and 'refine' sequence outputs.
            print("Running generic aggregation (produces standard structure).")
            try:
                all_chapters_agg = []
                global_tags_agg_dict = { "persons": {}, "actions": {}, "objects": {} }

                for result_container in processed_results:
                     if not isinstance(result_container, dict):
                         print(f"Warning: Skipping non-dict item in results list during generic agg: {type(result_container)}")
                         continue
                     # --- EDIT: Check camelCase 'analysisResult' ---
                     segment_response = result_container.get("analysisResult", {})
                     if not isinstance(segment_response, dict):
                          print(f"Warning: Skipping item with non-dict 'analysisResult' during generic agg: {type(segment_response)}")
                          continue

                     # Add chapters ('chapters' key is fine)
                     chapters_data = segment_response.get("chapters", [])
                     if isinstance(chapters_data, dict): all_chapters_agg.append(chapters_data)
                     elif isinstance(chapters_data, list): all_chapters_agg.extend(chap for chap in chapters_data if isinstance(chap, dict))
                     else: print(f"Warning: Unexpected data type for 'chapters' during generic agg: {type(chapters_data)}")

                     # Merge global tags ('globalTags' key is camelCase)
                     # --- EDIT: Check camelCase 'globalTags' ---
                     tags_data = segment_response.get("globalTags", {})
                     if not isinstance(tags_data, dict):
                         print(f"Warning: Unexpected data type for 'globalTags' during generic agg: {type(tags_data)}")
                         continue

                     for category in ["persons", "actions", "objects"]: # Internal keys 'persons', etc. are fine
                         if category not in tags_data or not isinstance(tags_data.get(category), list): continue

                         for tag_obj in tags_data[category]:
                             if not isinstance(tag_obj, dict):
                                 print(f"Warning: Skipping non-dictionary tag in '{category}' during generic agg: {type(tag_obj)}")
                                 continue
                             name = tag_obj.get("name") # 'name' fine
                             if not name or not isinstance(name, str) or not name.strip():
                                 print(f"Warning: Skipping tag in '{category}' with missing/invalid name during generic agg: {tag_obj}")
                                 continue

                             cleaned_name = name.strip()
                             if cleaned_name not in global_tags_agg_dict[category]:
                                 global_tags_agg_dict[category][cleaned_name] = {"name": cleaned_name, "timecodes": []} # 'name', 'timecodes' fine

                             timecodes = tag_obj.get("timecodes", []) # 'timecodes' fine
                             if isinstance(timecodes, list):
                                 valid_timecodes = [tc for tc in timecodes if isinstance(tc, dict) and "start" in tc and "end" in tc] # 'start', 'end' fine
                                 global_tags_agg_dict[category][cleaned_name]["timecodes"].extend(valid_timecodes)
                             else: print(f"Warning: Unexpected timecode format for tag '{cleaned_name}' during generic agg: {timecodes}")

                # Convert aggregated dict back to list structure and clean up timecodes
                final_global_tags_agg_list = {} # Use plural keys for final structure
                for category, tags_dict in global_tags_agg_dict.items():
                     tag_list_agg = []
                     for tag_name, tag_object in tags_dict.items():
                         unique_timecodes_set = set(tuple(sorted(d.items())) for d in tag_object["timecodes"])
                         unique_timecodes_list = sorted(
                             [dict(t) for t in unique_timecodes_set],
                             key=lambda x: float(str(x.get("start", "inf")).rstrip("s")) if str(x.get("start", "inf")).rstrip("s").replace('.', '', 1).isdigit() else float('inf')
                         )
                         tag_object["timecodes"] = unique_timecodes_list
                         tag_list_agg.append(tag_object)
                     final_global_tags_agg_list[category] = tag_list_agg # Use plural key

                # --- Assemble the standard structure for generic aggregation ---
                # This structure will be used for summary generation if needed,
                # but the final saved file will likely be the ActionSummary structure
                # if the ActionSummary config was used.
                final_results_standard = {
                    "chapters": all_chapters_agg, # 'chapters' fine
                    "globalTags": final_global_tags_agg_list # Use camelCase 'globalTags' containing plural keys internally
                }
                # --- Store this standard structure for potential summary use ---
                # The final file saved will depend on whether ActionSummary processing ran.
                # For now, let's assume ActionSummary always runs for this config.
                # We'll set final_results later. If ActionSummary fails, we might need this.
                _intermediate_generic_results = final_results_standard

                # Update manifest global tags from generic aggregation
                self.manifest.global_tags = final_global_tags_agg_list
                print("DEBUG: Updated self.manifest.global_tags from generic aggregation.")

            except Exception as e:
                print(f"Error during generic aggregation: {e}")
                # ... (existing error handling for generic aggregation failure) ...
                raise ValueError(f"Error during generic aggregation. Error: {e}")

            # --- EDIT: If generic aggregation ran BUT ActionSummary is the config,
            # we still expect the final result to be from ActionSummary's processing.
            # The generic aggregation path primarily serves non-ActionSummary mapreduce
            # or refine sequences. Let's check if ActionSummary processing needs to run
            # on the `processed_results` list. This happens automatically if `has_custom_aggregation` is true.
            # If `has_custom_aggregation` was false (e.g., refine), we need to handle it.
            # Let's assume `ActionSummary` processing IS the custom aggregation and already ran if `has_custom_aggregation` is true.
            # If it was false (refine sequence), `final_results` is still empty.
            # We need a way to potentially save the generic results if no ActionSummary exists.
            # Given the current structure, let's prioritize the `ActionSummary` output.
            # If `has_custom_aggregation` was true, `final_results` already holds the ActionSummary dict.
            # If it was false (refine), we need to decide what to save. The prompt asks for ActionSummary format.
            # Let's assume for 'refine' we might save the `_intermediate_generic_results` if needed.
            # But for the user's request (ActionSummary config), `has_custom_aggregation` should be true.
            if not final_results: # If custom aggregation didn't run (e.g., refine)
                print("Warning: No custom aggregation ran. Final results might not match ActionSummary format.")
                final_results = _intermediate_generic_results # Use generic result as fallback


        # -- Summary Generation & Saving --
        try:
            if hasattr(analysis_config, "run_final_summary") and analysis_config.run_final_summary:
                # --- EDIT: Target the correct dictionary based on structure ---
                content_to_summarize = final_results # Default: summarize the whole thing
                summary_target_dict = final_results # Default target
                summary_text = "Analysis resulted in empty content."
                description_text = "No description generated."
                should_generate_summary = False

                if "actionSummary" in final_results:
                     action_summary_content = final_results["actionSummary"]
                     content_to_summarize = action_summary_content # Summarize content inside actionSummary
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
                          # Add default summary/desc to actionSummary
                          summary_target_dict["description"] = description_text
                          summary_target_dict["summary"] = summary_text

                # Check standard structure (if not actionSummary - e.g., generic fallback)
                elif final_results and (final_results.get("chapters") or final_results.get("globalTags")):
                    should_generate_summary = True
                else: # Standard structure but empty
                    print("Skipping final summary generation due to empty analysis results.")
                    # Add default summary/desc to top level
                    final_results["description"] = description_text
                    final_results["summary"] = summary_text


                if should_generate_summary:
                    print(f"Generating summary and description for {self.manifest.name}")
                    summary_prompt_messages = self.generate_summary_prompt(analysis_config, content_to_summarize)
                    # Defaults reset here
                    description_text = "Failed to generate description."
                    summary_text = "Failed to generate summary."
                    try:
                        summary_results_llm = self._call_llm(summary_prompt_messages) # Renamed variable
                        raw_response_content = summary_results_llm.choices[0].message.content.strip()

                        try:
                            # ... (existing JSON parsing for description/summary) ...
                            if raw_response_content.startswith("```json"):
                                raw_response_content = raw_response_content.split("```json", 1)[1]
                            if raw_response_content.endswith("```"):
                                raw_response_content = raw_response_content.rsplit("```", 1)[0]

                            parsed_summaries = json.loads(raw_response_content.strip())
                            if isinstance(parsed_summaries, dict):
                                # --- EDIT: Use camelCase keys 'description', 'summary' ---
                                description_text = parsed_summaries.get("description", description_text).strip()
                                summary_text = parsed_summaries.get("summary", summary_text).strip()
                                print("Successfully parsed description and summary from LLM response.")
                            else:
                                print("Warning: LLM response for summaries was not a JSON dictionary.")
                                summary_text = raw_response_content

                        except json.JSONDecodeError as json_e:
                            print(f"Warning: Failed to parse JSON summary/description response: {json_e}")
                            print(f"Raw response: {raw_response_content}")
                            summary_text = raw_response_content

                    except Exception as llm_e:
                        print(f"Warning: LLM call for summary/description failed: {llm_e}")

                    # --- ADD keys to the target dictionary ---
                    # --- EDIT: Use camelCase keys 'description', 'summary' ---
                    summary_target_dict["description"] = description_text
                    summary_target_dict["summary"] = summary_text
                    print(f"DEBUG: Added description and summary (camelCase) to {'actionSummary object' if 'actionSummary' in final_results else 'top level'}.")

                # Update the manifest's final summary field (use the main summary text)
                # The manifest field itself is final_summary (snake_case)
                self.manifest.final_summary = summary_text


            # Save the results
            final_results_output_path = os.path.join(
                self.manifest.processing_params.output_directory,
                f"_{analysis_config.name}.json",
            )
            print(f"Writing final results structure to {final_results_output_path}")
            os.makedirs(os.path.dirname(final_results_output_path), exist_ok=True)
            with open(final_results_output_path, "w", encoding="utf-8") as f:
                 json_obj = json.loads(json.dumps(final_results))
                 f.write(json.dumps(json_obj, indent=4, ensure_ascii=False))

        except Exception as e:
            print(f"Error during final summary generation or saving results: {e}")
            # ... (existing error handling for final step) ...
            raise ValueError(f"Failed during summary/saving. Original error: {e}")


        stopwatch_end_time = time.time()

        elapsed_time = stopwatch_end_time - stopwatch_start_time

        print(
            f'Video analysis completed in {round(elapsed_time, 3)}: "{analysis_config.name}" for {self.manifest.name}'
        )
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
        copyright_json_str: Optional[str] = None # Added copyright string param
    ):
        results_list = []
        for segment in self.manifest.segments:
            parsed_response = self._analyze_segment(
                segment=segment, analysis_config=analysis_config
            )
            # --- EDIT: Use camelCase keys for the container dict ---
            results_list.append({
                "analysisResult": parsed_response, # camelCase
                "segmentName": segment.segment_name, # camelCase
                "startTime": segment.start_time, # camelCase
                "endTime": segment.end_time,   # camelCase
                "framePaths": segment.segment_frames_file_path, # camelCase
                "fullTranscriptionObject": self.manifest.audio_transcription if not results_list else None # camelCase
             })

        # After all segments have been analyzed
        if hasattr(analysis_config, 'process_segment_results'):
            parsed_copyright = None
            # --- EDIT: Use passed copyright_json_str ---
            # copyright_json_str = kwargs.get('copyright_json_str') # No longer needed from kwargs
            if copyright_json_str:
                try:
                    parsed_copyright = json.loads(copyright_json_str)
                    print("DEBUG (SYNC): Successfully parsed copyright JSON string.")
                except json.JSONDecodeError as e:
                    print(f"Warning (SYNC): Failed to parse copyright JSON string: {e}. Content: {copyright_json_str[:100]}...")
            # --- End Parse ---

            # Pass the list with camelCase keys
            final_results = analysis_config.process_segment_results(
                results_list,
                self.manifest,
                self.env,
                parsed_copyright_info=parsed_copyright
            )
            # --- EDIT: Check for 'actionSummary' and update manifest tags ---
            # The logic to update manifest.global_tags was moved to analyze_video
            # as it needs the final processed result, which is returned here.
            print(f"DEBUG (SYNC): Returning results from custom process_segment_results.")
            return final_results
        else:
             # Return the raw list (with camelCase keys) for analyze_video to handle generic aggregation
             return results_list

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
        self,
        analysis_config: Type[AnalysisConfig],
        max_concurrent_tasks=None,
        copyright_json_str: Optional[str] = None
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
                # --- EDIT: Use camelCase keys for the container dict ---
                return {
                    "analysisResult": parsed_response, # camelCase
                    "segmentName": segment.segment_name, # camelCase
                    "startTime": segment.start_time, # camelCase
                    "endTime": segment.end_time,   # camelCase
                    "framePaths": segment.segment_frames_file_path, # camelCase
                    "fullTranscriptionObject": self.manifest.audio_transcription # camelCase
                }

        async def return_value_task(segment):
            # --- EDIT: Return the analysis result directly ---
            # Assuming the structure stored in segment.analyzed_result is already correct (or will be fixed by analyze_segment)
            return segment.analyzed_result.get(analysis_config.name)

        segment_task_list = []
        needs_processing = False # Flag to check if any segment actually needs processing
        for segment in self.manifest.segments:
            if (
                self.reprocess_segments is False
                and analysis_config.name in segment.analysis_completed
            ):
                print(
                    f"Segment {segment.segment_name} has already been analyzed, creating value task."
                )
                # --- EDIT: Append a task that returns the structured result, not just the value ---
                # We need the surrounding keys like 'segmentName' etc., if generic aggregation runs later.
                # Let's make it return the same structure as sem_task for consistency.
                # Assume the stored result under analysis_config.name is the 'analysisResult' part.
                stored_result = segment.analyzed_result.get(analysis_config.name, {})
                segment_task_list.append(asyncio.create_task(asyncio.sleep(0, result={
                    "analysisResult": stored_result,
                    "segmentName": segment.segment_name,
                    "startTime": segment.start_time,
                    "endTime": segment.end_time,
                    "framePaths": segment.segment_frames_file_path,
                    "fullTranscriptionObject": self.manifest.audio_transcription
                })))
            else:
                needs_processing = True
                segment_task_list.append(sem_task(segment))

        # --- EDIT: Gather results into a list ---
        results_list = await asyncio.gather(*segment_task_list)
        # Filter out potential None results if tasks failed unexpectedly (though gather should raise)
        results_list = [res for res in results_list if res is not None]


        # --- Aggregation Logic ---
        if hasattr(analysis_config, 'process_segment_results'):
            parsed_copyright_agg = None
            if copyright_json_str:
                try:
                    parsed_copyright_agg = json.loads(copyright_json_str)
                    print("DEBUG (ASYNC): Successfully parsed copyright JSON string.")
                except json.JSONDecodeError as e:
                    print(f"Warning (ASYNC): Failed to parse copyright JSON string: {e}. Content: {copyright_json_str[:100]}...")
            # --- End Parse ---

            # Pass the list with camelCase keys
            final_results_agg = analysis_config.process_segment_results(
                results_list,
                self.manifest,
                self.env,
                parsed_copyright_info=parsed_copyright_agg
            )
            # --- EDIT: Manifest update logic moved to analyze_video ---
            print(f"DEBUG (ASYNC): Returning results from custom process_segment_results.")
            return final_results_agg # Return aggregated results
        else:
             # Return the raw list (with camelCase keys) for analyze_video generic aggregation
             return results_list

    def _analyze_segment(
        self,
        segment: Segment,
        analysis_config: Type[AnalysisConfig]
    ):
        if (not self.reprocess_segments
            and analysis_config.name in segment.analysis_completed):
            print(f"Segment {segment.segment_name} already analyzed, skipping.")
            # --- EDIT: Return structure consistent with camelCase ---
            return segment.analyzed_result.get(analysis_config.name, {"chapters": [], "globalTags": {"persons": [], "actions": [], "objects": []}}) # Use globalTags

        print(f"Analyzing segment {segment.segment_name} ({segment.start_time:.3f}s - {segment.end_time:.3f}s)")
        stopwatch_segment = time.time()

        max_retries = 3
        initial_delay = 1.0

        # --- Stage 1: Get Chapters ---
        print(f"  - Stage 1: Requesting Chapters...")
        chapters_result = {"chapters": []}
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
                    # --- EDIT: _parse_llm_json_response returns camelCase keys internally ---
                    chapters_result = self._parse_llm_json_response(
                        chapter_llm_response,
                        expecting_chapters=True,
                        expecting_tags=False
                    ) # Expects {"chapters": ...}
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
                # ... (retry logic) ...
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
            parsed_chunk = None # Holds result of the CURRENT attempt
            successful_parsed_chunk = None # Holds the LAST SUCCESSFUL result for this chunk

            for attempt in range(max_retries):
                try:
                    tag_prompt = self._generate_segment_prompt(
                        segment=segment, analysis_config=analysis_config, frames_subset=chunk_frames,
                        times_subset=chunk_times, generate_chapters=False, generate_tags=True,
                        chunk_start_time=chunk_start, chunk_end_time=chunk_end
                    )

                    if tag_prompt:
                        tag_llm_response = self._call_llm(tag_prompt)
                        # --- EDIT: _parse_llm_json_response returns camelCase keys internally ---
                        parsed_chunk = self._parse_llm_json_response(
                            tag_llm_response, expecting_chapters=False, expecting_tags=True
                        ) # Expects {"globalTags": {"persons": ..., "actions": ..., "objects": ...}}
                        print(f"      - DEBUG: Chunk {i+1} (Attempt {attempt+1}) Parsed Result: {json.dumps(parsed_chunk)}")

                        # Check if parsing was successful AND contains tags
                        # --- EDIT: Check camelCase 'globalTags' ---
                        if parsed_chunk is not None and any(parsed_chunk.get("globalTags", {}).values()):
                            # --- EDIT: Access camelCase 'globalTags' ---
                            tags_found = sum(len(v) for k, v in parsed_chunk.get("globalTags", {}).items())
                            print(f"      - Chunk {i+1} (Attempt {attempt+1}) analysis complete. Found {tags_found} tags.")
                            # --- Store the successful result ---
                            successful_parsed_chunk = parsed_chunk # Store the successful result
                            # --- END Store ---
                            break # Success, exit retry loop for this chunk
                        else:
                            # Parsing succeeded but returned no tags, or returned default empty.
                            # Check if the raw response was likely empty or problematic before deciding it's an issue
                            raw_content_check = tag_llm_response.choices[0].message.content.strip()
                            if not raw_content_check or raw_content_check == "{}":
                                 # This indicates a likely LLM issue, raise error to trigger retry
                                 raise ValueError("LLM response parsed to empty tags/structure, possibly indicating an issue.")
                            else:
                                 # Parsing ok, but no tags found by LLM for this attempt. Log it.
                                 # If this was the only successful parse, successful_parsed_chunk will be this empty dict.
                                 print(f"      - INFO: Chunk {i+1} (Attempt {attempt+1}) parsed successfully but LLM found no tags.")
                                 successful_parsed_chunk = parsed_chunk # Store the empty-tag result
                                 break # Treat as success (LLM just found nothing), exit retry loop

                    else:
                        print(f"      - Chunk {i+1}: Skipping tag generation due to empty prompt.")
                        break # No prompt, exit retry loop

                except Exception as e:
                    # ... (retry logic) ...
                    print(f"      - Chunk {i+1}: Attempt {attempt + 1}/{max_retries} failed for Tags. Error: {e}")
                    parsed_chunk = None # Reset current attempt parse result

                    if attempt + 1 == max_retries:
                        print(f"      - Chunk {i+1}: Max retries reached. Skipping tags for this chunk (unless a previous attempt succeeded).")
                    else:
                        wait_time = initial_delay * (2 ** attempt) # Exponential backoff
                        print(f"        - Retrying in {wait_time:.1f} seconds...")
                        time.sleep(wait_time)

            # --- Append the LAST SUCCESSFUL result (if any) ---
            # --- More DEBUGGING: Check parsed_chunk and append ---
            print(f"      - DEBUG: Chunk {i+1} retry loop finished. Checking successful_parsed_chunk before append.")
            # Use the variable that holds the last successful result
            if successful_parsed_chunk is not None:
               print(f"      - DEBUG: Chunk {i+1} - A successful parse was found ({type(successful_parsed_chunk)}). Appending to tag_chunk_results.")
               tag_chunk_results.append(successful_parsed_chunk)
               print(f"      - DEBUG: Chunk {i+1} - tag_chunk_results now has {len(tag_chunk_results)} items.")
            else:
               print(f"      - DEBUG: Chunk {i+1} - No successful parse result found for this chunk; not appending.")
            # --- END Append Logic ---


        # --- DEBUGGING: Check list before merging ---
        print(f"  - DEBUG: Finished processing all chunks for segment. Checking tag_chunk_results before merge.")
        print(f"  - DEBUG: Collected tag_chunk_results count: {len(tag_chunk_results)}")
        print(f"  - DEBUG: Collected tag_chunk_results content (first 2): {json.dumps(tag_chunk_results[:2], indent=2)}")
        # --- END DEBUGGING ---

        # --- Stage 3: Merge Tags ---
        print(f"  - Stage 3: Merging tags from {len(tag_chunk_results)} successful chunks...")
        # --- EDIT: _merge_tagging_chunks expects list of dicts with 'globalTags' key ---
        merged_global_tags = self._merge_tagging_chunks(tag_chunk_results) # Returns {"persons": ..., "actions": ..., "objects": ...}
        tags_merged_count = sum(len(v) for k, v in merged_global_tags.items())
        print(f"  - Stage 3: Merged into {tags_merged_count} unique tags.")

        # --- Stage 4: Combine ---
        # --- EDIT: Use camelCase 'globalTags' for the final structure ---
        final_combined_result = {
            "chapters": chapters_result.get("chapters", []), # Use potentially empty chapters
            "globalTags": merged_global_tags # Use camelCase key
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
            # --- EDIT: Return structure consistent with camelCase ---
            return segment.analyzed_result.get(analysis_config.name, {"chapters": [], "globalTags": {"persons": [], "actions": [], "objects": []}}) # Use globalTags

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
                    # --- EDIT: _parse_llm_json_response returns camelCase keys internally ---
                    chapters_result = self._parse_llm_json_response(
                         chapter_llm_response, expecting_chapters=True, expecting_tags=False
                    ) # Expects {"chapters": ...}
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
                # ... (retry logic) ...
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
                    # --- EDIT: _parse_llm_json_response returns camelCase keys internally ---
                    parsed_chunk = self._parse_llm_json_response(
                         tag_llm_response, expecting_chapters=False, expecting_tags=True
                    ) # Expects {"globalTags": ...}
                    # --- EDIT: Check camelCase 'globalTags' ---
                    if not any(parsed_chunk.get("globalTags", {}).values()):
                        raw_content_check = tag_llm_response.choices[0].message.content.strip()
                        if not raw_content_check or raw_content_check == "{}":
                            raise ValueError("LLM response parsed to empty tags, possibly indicating an issue.")
                    # --- EDIT: Access camelCase 'globalTags' ---
                    tags_found = sum(len(v) for k, v in parsed_chunk.get("globalTags", {}).items())
                    print(f"      - Chunk {chunk_idx+1} (Attempt {attempt+1}) analysis complete. Found {tags_found} tags.")
                    return parsed_chunk # Success

                except Exception as e:
                     # ... (retry logic) ...
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
        # --- EDIT: _merge_tagging_chunks expects list of dicts with 'globalTags' key ---
        merged_global_tags = self._merge_tagging_chunks(tag_chunk_results) # Returns {"persons": ..., "actions": ..., "objects": ...}
        tags_merged_count = sum(len(v) for k, v in merged_global_tags.items())
        print(f"  - Stage 3 (Async): Merged into {tags_merged_count} unique tags.")

        # --- Stage 4: Combine ---
        # --- EDIT: Use camelCase 'globalTags' ---
        final_combined_result = {
            "chapters": chapters_result.get("chapters", []), # Use potentially empty chapters
            "globalTags": merged_global_tags # Use camelCase
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

        return final_combined_result # Return camelCase structure

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
        is_partial_chunk: bool = False # This argument seems unused, but kept for consistency
    ):
        """
        Generate a prompt for a segment, adaptable for full context or tag-specific chunks.
        Handles chapters OR tags, not both simultaneously.
        Provides robust fallbacks for missing config attributes and formatting errors.
        """
        frames_to_process = frames_subset if frames_subset is not None else segment.segment_frames_file_path
        times_to_process = times_subset if times_subset is not None else segment.segment_frame_time_intervals
        transcription_context = segment.transcription if segment.transcription else "No transcription available"
        messages = []
        system_prompt = "Default system prompt." # Initialize with a default
        lens_prompt = "" # Initialize lens template
        # Variable to hold results template string for final reminder
        results_template_str_for_reminder = "{}"

        if not frames_to_process:
             print(f"Warning: No frames to process for prompt generation in segment {segment.segment_name}.")
             return None # Cannot generate prompt without frames

        # Ensure frame times match frames
        if not times_to_process or len(times_to_process) != len(frames_to_process):
             print(f"Warning: Mismatch or missing times for frames in segment {segment.segment_name}. Recalculating...")
             effective_start = chunk_start_time if chunk_start_time is not None else segment.start_time
             effective_end = chunk_end_time if chunk_end_time is not None else segment.end_time
             if len(frames_to_process) > 0:
                  times_to_process = np.linspace(effective_start, effective_end, len(frames_to_process), endpoint=False)
                  times_to_process = [round(float(t), 3) for t in times_to_process]
             else:
                  times_to_process = []

        # --- Determine which prompt components to use ---
        if generate_chapters and generate_tags:
            print("Error: _generate_segment_prompt called requesting both chapters and tags simultaneously.")
            raise ValueError("Cannot generate chapters and tags in the same prompt.")
        elif generate_chapters:
            if not hasattr(analysis_config, 'system_prompt_chapters'):
                 print("Warning: AnalysisConfig missing 'system_prompt_chapters'. Falling back to default.")
                 system_prompt_template = "Default system prompt for chapters. Please provide chapter summary."
            else:
                 system_prompt_template = getattr(analysis_config, 'system_prompt_chapters', "Default system prompt for chapters.")

            if not hasattr(analysis_config, 'system_prompt_lens_chapters'):
                 print("Warning: AnalysisConfig missing 'system_prompt_lens_chapters'. Falling back to default.")
                 lens_prompt = getattr(analysis_config, 'lens_prompt', "Default lens prompt for chapters.")
            else:
                 lens_prompt = getattr(analysis_config, 'system_prompt_lens_chapters', "Default lens prompt for chapters.")

            # Get results template string for the final reminder, but don't format it into system prompt
            if not hasattr(analysis_config, 'results_template_chapters'):
                 print("Warning: AnalysisConfig missing 'results_template_chapters'. Falling back to default.")
                 results_template_str_for_reminder = json.dumps({"chapters": []})
            else:
                 results_template_obj = getattr(analysis_config, 'results_template_chapters', {"chapters": []})
                 results_template_str_for_reminder = json.dumps(results_template_obj, indent=2)

            try:
                # *** Prepare arguments dictionary for format_map ***
                format_data_chapters = {
                    "video_duration": f"{self.manifest.source_video.duration:.3f}",
                    "segment_duration": segment.segment_duration,
                    "start_time": f"{segment.start_time:.3f}",
                    "end_time": f"{segment.end_time:.3f}",
                    "number_of_frames": segment.number_of_frames
                    # Note: Lens is added later to user prompt for chapters
                }
                # *** Use format_map instead of format ***
                system_prompt = system_prompt_template.format_map(format_data_chapters)

            except KeyError as e: # Should be less likely now, but keep for safety
                # Print the specific missing key
                print(f"ERROR: Missing key {repr(e.args[0])} during chapter system_prompt formatting. Using template as is.")
                system_prompt = system_prompt_template # Fallback to unformatted template
            except Exception as e:
                print(f"ERROR: Unexpected error during chapter system_prompt formatting: {e}. Using template as is.")
                system_prompt = system_prompt_template

            messages.append({"role": "system", "content": system_prompt})

        elif generate_tags:
            # --- EDIT: Simplify and correct the check for using the custom people prompt ---
            # Determine if a valid peoples list was loaded during initialization
            use_custom_people_prompt = bool(self.peoples_list)
            # --- END EDIT ---

            if use_custom_people_prompt:
                 print("DEBUG: Using system_prompt_tags_custom_people") # Add debug print
                 if not hasattr(analysis_config, 'system_prompt_tags_custom_people'):
                      print("Warning: AnalysisConfig missing 'system_prompt_tags_custom_people'. Falling back.")
                      # Fallback to standard tags prompt if custom one is missing in config
                      system_prompt_template = getattr(analysis_config, 'system_prompt_tags', "Default system prompt for tags (custom people fallback).")
                 else:
                      system_prompt_template = getattr(analysis_config, 'system_prompt_tags_custom_people', "Default system prompt for custom people tags.")
            else:
                 print("DEBUG: Using standard system_prompt_tags") # Add debug print
                 if not hasattr(analysis_config, 'system_prompt_tags'):
                      print("Warning: AnalysisConfig missing 'system_prompt_tags'. Falling back.")
                      system_prompt_template = getattr(analysis_config, 'system_prompt', "Default system prompt for tags.")
                 else:
                      system_prompt_template = getattr(analysis_config, 'system_prompt_tags', "Default system prompt for tags.")

            if not hasattr(analysis_config, 'system_prompt_lens_tags'):
                 print("Warning: AnalysisConfig missing 'system_prompt_lens_tags'. Falling back.")
                 lens_prompt = getattr(analysis_config, 'lens_prompt', "Default lens prompt for tags.")
            else:
                 lens_prompt = getattr(analysis_config, 'system_prompt_lens_tags', "Default lens prompt for tags.")

            # Get results template string for the final reminder, but don't format it into system prompt
            if not hasattr(analysis_config, 'results_template_tags'):
                 print("Warning: AnalysisConfig missing 'results_template_tags'. Falling back.")
                 results_template_obj = getattr(analysis_config, 'results_template', {"global_tags": {}})
                 results_template_str_for_reminder = json.dumps(results_template_obj, indent=2)
            else:
                 results_template_obj = getattr(analysis_config, 'results_template_tags', {"global_tags": {}})
                 results_template_str_for_reminder = json.dumps(results_template_obj, indent=2)

            # --- Correctly format the custom people definitions ---
            formatted_people_definitions = "No specific person definitions provided."
            if isinstance(self.peoples_list, dict) and "peoples" in self.peoples_list and isinstance(self.peoples_list["peoples"], list):
                definitions_list = []
                for person_def in self.peoples_list["peoples"]:
                    if isinstance(person_def, dict) and "label" in person_def and "description" in person_def:
                        label = person_def.get("label", "N/A")
                        desc = person_def.get("description", "N/A")
                        definitions_list.append(f"- label: {label}\n  description: {desc}")
                if definitions_list:
                    formatted_people_definitions = "\n".join(definitions_list)
            # --- END Formatting ---

            # --- Get Object/Action defs (using existing helper is fine here) ---
            object_defs = self._get_list_definitions(self.objects_list, "objects")
            action_defs = self._get_list_definitions(self.actions_list, "actions")
            # --- END Object/Action defs ---

            explicit_object_reminder = "" if object_defs != "No specific object definitions provided." else \
                 "Reminder: You MUST tag standard objects like 'Safety Helmet' and 'High-Visibility Jacket'."
            explicit_action_reminder = "" if action_defs != "No specific action definitions provided." else \
                 "Reminder: Tag significant standard actions (e.g., Walking, Using tool, Operating machinery)."

            try:
                # *** Prepare arguments dictionary for format_map ***
                format_data_tags = {
                    "start_time": f"{chunk_start_time:.3f}" if chunk_start_time is not None else f"{segment.start_time:.3f}",
                    "end_time": f"{chunk_end_time:.3f}" if chunk_end_time is not None else f"{segment.end_time:.3f}",
                    "number_of_frames": len(frames_subset) if frames_subset else segment.number_of_frames,
                    # --- Use the formatted list string ---
                    "people_definitions": formatted_people_definitions,
                    # --- END Use formatted list ---
                    "object_definitions": object_defs,
                    "action_definitions": action_defs,
                    "explicit_object_reminder": explicit_object_reminder,
                    "explicit_action_reminder": explicit_action_reminder,
                    "video_duration": f"{self.manifest.source_video.duration:.3f}",
                    "segment_duration": segment.segment_duration,
                    # --- Known tags are disabled, keep keys commented/removed if needed ---
                    # "known_persons": "...",
                    # "known_actions": "...",
                    # "known_objects": "...",
                }
                # *** Use format_map instead of format ***
                system_prompt = system_prompt_template.format_map(format_data_tags)

            except KeyError as e: # Should be less likely now, but keep for safety
                # Print the specific missing key
                print(f"ERROR: Missing key {repr(e.args[0])} during tag system_prompt formatting. Using template as is.")
                system_prompt = system_prompt_template # Fallback to unformatted template
            except Exception as e:
                print(f"ERROR: Unexpected error during tag system_prompt formatting: {e}. Using template as is.")
                system_prompt = system_prompt_template

            messages.append({"role": "system", "content": system_prompt})

        else:
             print("Warning: _generate_segment_prompt called without specifying chapters or tags.")
             return None

        # --- Prepare user content ---
        user_content = []
        formatted_lens_prompt = ""

        if generate_chapters and lens_prompt:
             try:
                formatted_lens_prompt = lens_prompt.format(
                    start_time=f"{segment.start_time:.3f}",
                    end_time=f"{segment.end_time:.3f}",
                    number_of_frames=segment.number_of_frames,
                    transcription_context=transcription_context
                )
             except KeyError as e:
                # Print the specific missing key
                print(f"ERROR: Missing key {repr(e.args[0])} during chapter lens_prompt formatting. Using template as is.")
                formatted_lens_prompt = lens_prompt # Fallback to unformatted template
             except Exception as e:
                 print(f"ERROR: Unexpected error during chapter lens_prompt formatting: {e}. Using template as is.")
                 formatted_lens_prompt = lens_prompt
             user_content.append({"type": "text", "text": formatted_lens_prompt})
             user_content.append({"type": "text", "text": f"\n--- Analyzing Segment Frames ({segment.start_time:.3f}s - {segment.end_time:.3f}s) ---"})
        elif generate_tags:
             # Add context and header for tag analysis
             user_content.append({"type": "text", "text": f"Overall Segment Transcription Context: {transcription_context}"})
             start_t = chunk_start_time if chunk_start_time is not None else segment.start_time
             end_t = chunk_end_time if chunk_end_time is not None else segment.end_time
             user_content.append({"type": "text", "text": f"\n--- Analyzing Chunk Frames ({start_t:.3f}s - {end_t:.3f}s) ---"})


        # --- Add Frames and Final Reminder (Common logic) ---
        user_content.append({
            "type": "text",
            "text": """IMPORTANT: Analyze the following frames carefully. Use the provided timestamps to determine accurate start/end times for chapters or tag timecodes."""
        })

        for i, frame_path in enumerate(frames_to_process):
             if i >= len(times_to_process):
                  print(f"Warning: Missing timestamp for frame {i} ({frame_path}). Skipping frame.")
                  continue
             timestamp = times_to_process[i]

             try:
                 image_content = encode_image_base64(frame_path)
                 if not image_content:
                      print(f"Warning: Failed to encode frame {i} ({frame_path}). Skipping.")
                      continue
             except Exception as e:
                 print(f"Warning: Error encoding frame {i} ({frame_path}): {e}. Skipping.")
                 continue

             user_content.append({"type": "text", "text": f"\nFrame at {timestamp:.3f}s:"})
             user_content.append({
                 "type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_content}", "detail": "high"}
             })

        # Add FINAL reminder about output format
        final_reminder_text = "CRITICAL: Your output MUST be VALID JSON."
        if generate_chapters:
             final_reminder_text += " containing ONLY the 'chapters' key, with exactly one chapter object inside the array."
             final_reminder_text += f"\n\nExpected Structure:\n{results_template_str_for_reminder}"
        elif generate_tags:
             start_t = chunk_start_time if chunk_start_time is not None else segment.start_time
             end_t = chunk_end_time if chunk_end_time is not None else segment.end_time
             final_reminder_text += f" containing ONLY the 'globalTags' key, with 'persons', 'actions', and 'objects' arrays. Tags MUST be strictly within {start_t:.3f}s to {end_t:.3f}s."
             final_reminder_text += f"\n\nExpected Structure:\n{results_template_str_for_reminder}"

        user_content.append({"type": "text", "text": final_reminder_text})

        # --- Construct and Save Final Prompt ---
        messages.append({"role": "user", "content": user_content})

        try:
            prompt_type = "chapters" if generate_chapters else "tags"
            chunk_suffix = f"_chunk_{chunk_start_time:.3f}s" if chunk_start_time is not None else ""
            prompt_output_path = os.path.join(
                segment.segment_folder_path, f"{segment.segment_name}_prompt_{prompt_type}{chunk_suffix}.json"
            )
            with open(prompt_output_path, "w", encoding="utf-8") as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save generated prompt to file: {e}")

        return messages


    def _get_list_definitions(self, list_data: Optional[Dict], list_type: str) -> str:
         """Helper to extract definitions or instructions from loaded lists."""
         default_text = f"No specific {list_type} definitions provided."
         if not isinstance(list_data, dict):
              return default_text

         # Prioritize 'definitions', then 'instructions', then default
         definitions = list_data.get("definitions")
         if definitions and isinstance(definitions, str) and definitions.strip():
              return definitions.strip()

         instructions = list_data.get("instructions")
         if instructions and isinstance(instructions, str) and instructions.strip():
              return instructions.strip() # Return instructions if definitions missing

         return default_text

    def _merge_tagging_chunks(self, tag_chunk_results: List[Dict]) -> Dict:
        """Merges global_tags from multiple chunk analyses."""
        # Stores tag_name -> list of float timestamps
        merged_timestamps = {
            "persons": defaultdict(list),
            "actions": defaultdict(list),
            "objects": defaultdict(list)
        }

        for chunk_result in tag_chunk_results:
            if not isinstance(chunk_result, dict): continue
            global_tags = chunk_result.get("globalTags", {})
            if not isinstance(global_tags, dict): continue

            for tag_type in ["persons", "actions", "objects"]:
                if tag_type not in global_tags or not isinstance(global_tags[tag_type], list): continue

                for item in global_tags[tag_type]:
                     if not isinstance(item, dict): continue
                     name = item.get("name"); timecodes = item.get("timecodes", [])
                     if not name or not isinstance(name, str) or not name.strip() or not isinstance(timecodes, list): continue

                     cleaned_name = name.strip()
                     # --- Updated Timecode Parsing Logic --- 
                     # Expect timecodes to be a list of strings like ["X.XXXs"]
                     for time_str in timecodes:
                         if isinstance(time_str, str) and time_str.endswith('s'):
                              try:
                                   # Extract float timestamp
                                   timestamp_float = float(time_str[:-1])
                                   # Append the float timestamp to the list for this tag name
                                   merged_timestamps[tag_type][cleaned_name].append(timestamp_float)
                              except (ValueError, TypeError):
                                   print(f"Warning: Skipping invalid timecode string format during chunk merge: {time_str}")
                                   continue
                         elif isinstance(time_str, (int, float)):
                              # Handle cases where it might already be numeric
                              merged_timestamps[tag_type][cleaned_name].append(float(time_str))
                         else:
                             # Handle unexpected format, e.g., the old dict format if it reappears
                             if isinstance(time_str, dict) and 'start' in time_str:
                                 try:
                                     start_f = float(str(time_str["start"]).rstrip('s'))
                                     # If it's a dict, append the start time as the timestamp
                                     merged_timestamps[tag_type][cleaned_name].append(start_f)
                                 except (ValueError, TypeError):
                                      print(f"Warning: Skipping invalid timecode dict format during chunk merge: {time_str}")
                                      continue
                             else:
                                 print(f"Warning: Skipping unexpected timecode format during chunk merge: {time_str} ({type(time_str)})")
                     # --- End Updated Logic --- 

        # Convert defaultdict back to the final list structure required by ActionSummary aggregation
        # Keys 'persons', 'actions', 'objects', 'name', 'timecodes', 'start', 'end' are expected later
        # The _aggregate_tag_timestamps function handles converting the list of floats to intervals.
        # So, we just return the dictionary mapping name to list of floats.
        # Note: This returns a dict like {'persons': {'name1': [t1, t2], 'name2': [t3]}, ...}
        # This structure is *correct* for the input to _aggregate_tag_timestamps in ActionSummary

        final_merged_timestamps_dict = {
            "persons": dict(merged_timestamps["persons"]),
            "actions": dict(merged_timestamps["actions"]),
            "objects": dict(merged_timestamps["objects"])
        }

        # The calling function (_analyze_segment) expects a dict like:
        # {"persons": [...], "actions": [...], "objects": [...]}
        # where each list item is {"name": ..., "timecodes": [...]}
        # The _aggregate_tag_timestamps in ActionSummary does the final conversion to that format.
        # So, _merge_tagging_chunks should return the raw timestamp dictionary.
        # Correction: The _analyze_segment does expect the list-of-dicts format.
        # Let's reformat final_merged_timestamps_dict back to that format here.

        final_list_structure = { "persons": [], "actions": [], "objects": [] }
        for tag_type, name_dict in final_merged_timestamps_dict.items():
            for name, timestamp_list in name_dict.items():
                if timestamp_list:
                     # Store the list of float timestamps directly
                     final_list_structure[tag_type].append({
                          "name": name,
                          "timecodes": sorted(list(set(timestamp_list))) # Deduplicate and sort floats
                     })

        # Return the structure expected by _analyze_segment (list of dicts with float timecodes)
        # This structure is *NOT* what _aggregate_tag_timestamps expects directly,
        # but process_segment_results in ActionSummary handles this conversion.
        return final_list_structure

    def _parse_llm_json_response(self, response, expecting_chapters=True, expecting_tags=True):
         # ... (initial raw_content extraction) ...
         raw_content = response.choices[0].message.content
         content = raw_content # Placeholder for stripped content

         # 1. Strip ```json ... ``` markdown
         if "```json" in content:
             content = content.split("```json")[1].split("```")[0].strip()
         elif content.startswith("```") and content.endswith("```"):
             content = content[3:-3].strip()

         # 2. Attempt to remove single-line // comments
         try:
             lines = content.splitlines()
             cleaned_lines = []
             for line in lines:
                 stripped_line = line.strip()
                 if not stripped_line.startswith("//"):
                     cleaned_lines.append(line)
             content = "\n".join(cleaned_lines)
         except Exception as e:
             print(f"Warning: Pre-parsing comment removal failed: {e}. Attempting parse anyway.")
             # If comment removal fails, proceed with original content

         # 3. Parse the cleaned JSON string
         try:
             parsed_data = json.loads(content)
             if not isinstance(parsed_data, dict):
                 raise json.JSONDecodeError("Response is not a JSON object", content, 0)

             # ... (rest of the existing parsing logic: processing chapters/tags) ...

             # --- EDIT: Return dictionary with camelCase keys ---
             final_parsed_content = {}

             # Process Chapters if expected
             if expecting_chapters:
                 chapters_raw = parsed_data.get("chapters", []) # 'chapters' fine
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
                 final_parsed_content["chapters"] = processed_chapters # 'chapters' fine

             # Process Global Tags if expected
             if expecting_tags:
                 # --- EDIT: Check camelCase 'globalTags' ---
                 global_tags_raw = parsed_data.get("globalTags", {}) # Check camelCase
                 processed_tags = {"persons": [], "actions": [], "objects": []} # Internal keys fine
                 if isinstance(global_tags_raw, dict):
                      for tag_type in ["persons", "actions", "objects"]:
                           tags_list = global_tags_raw.get(tag_type, [])
                           if not isinstance(tags_list, list): tags_list = []
                           valid_tags_for_type = []
                           for item in tags_list:
                                if isinstance(item, dict) and "name" in item and "timecodes" in item: # Internal keys fine
                                     # Add basic validation for timecodes format maybe?
                                     valid_tags_for_type.append(item)
                           processed_tags[tag_type] = valid_tags_for_type
                 # --- EDIT: Assign to camelCase 'globalTags' ---
                 final_parsed_content["globalTags"] = processed_tags # Assign camelCase

             # Ensure default keys exist if not expected but parsed (or if parse failed partially)
             if "chapters" not in final_parsed_content: final_parsed_content["chapters"] = []
             if "globalTags" not in final_parsed_content: final_parsed_content["globalTags"] = {"persons": [], "actions": [], "objects": []}

             return final_parsed_content

         except json.JSONDecodeError as e:
            # ... (existing error handling) ...
            print(f"Error parsing JSON response: {e}")
            print(f"Response content (after potential comment removal):\n{content}")
            fallback = {}
            if expecting_chapters: fallback["chapters"] = []
            if expecting_tags: fallback["globalTags"] = {"persons": [], "actions": [], "objects": []}
            return fallback
         except Exception as e:
             # ... (existing error handling) ...
             print(f"Error processing LLM response: {e}")
             print(f"Response content (after potential comment removal):\n{content}")
             fallback = {}
             if expecting_chapters: fallback["chapters"] = []
             if expecting_tags: fallback["globalTags"] = {"persons": [], "actions": [], "objects": []}
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
            max_tokens=3000,
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
            max_tokens=3000,
        )

        return response