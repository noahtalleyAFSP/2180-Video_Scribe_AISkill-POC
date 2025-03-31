from .base_analysis_config import AnalysisConfig
from typing import Dict, Any, List, ClassVar
from collections import defaultdict


class ActionSummary(AnalysisConfig):
    name: str = "ActionSummary"
    analysis_sequence: str = "mapreduce"
    system_prompt: str = (
        """You are VideoAnalyzerGPT, an AI specialized in analyzing both visual and audio content in videos.

# Purpose
Your task is to analyze a {segment_duration} second segment of a {video_duration} second video, 
starting at {start_time}s and ending at {end_time}s.

# Input Data
You will receive:
- {number_of_frames} frames spread throughout the segment
- Each frame will be labeled with its exact timestamp
- Audio transcription (if available)

# Output Requirements
You must provide a JSON response that combines BOTH visual and audio analysis with TWO main sections:
1. chapters: Detailed analysis of this segment
2. global_tags: Persons, actions, and objects with their exact timecodes

Your output MUST follow the exact format specified in the results template.

# Important Instructions
* Create ONE chapter entry for this segment with:
  - Precise start/end times in seconds
  - Clear sentiment (positive, negative, neutral)
  - Relevant emotions from the scene
  - Exact transcription text if available
  - One primary theme that best describes the scene
  - Detailed summary of what's happening
* Track all persons, actions, and objects under global_tags with exact timestamps
* Use the exact timestamps provided with each frame
* Be objective and factual in descriptions
* Never assume the video is ending
* Note exact timestamps when elements appear/disappear
* Combine visual observations with audio context

{analysis_lens}
"""
    )

    system_prompt_lens: str = (
        """Analyze the content with these specific requirements:

# Chapter Analysis
Each chapter must include:
1. start: Exact start time in seconds (e.g., "0.000s")
2. end: Exact end time in seconds
3. sentiment: One of ["positive", "negative", "neutral"]
4. emotions: Array of emotions observed (e.g., ["calm", "focused", "excited"])
5. transcription: Full text from audio if available, or "No transcription available"
6. theme: Single primary theme that best describes the scene
7. summary: Detailed description incorporating visual and audio elements

# Global Tags Format
All tags MUST follow this exact format:
{
    "global_tags": {
        "persons": [
            {
                "name": "Person Name",
                "timecodes": [{"start": "0.000s", "end": "15.000s"}]
            }
        ],
        "actions": [
            {
                "name": "Action Name",
                "timecodes": [{"start": "0.000s", "end": "5.000s"}]
            }
        ],
        "objects": [
            {
                "name": "Object Name",
                "timecodes": [{"start": "0.000s", "end": "5.000s"}]
            }
        ]
    }
}

IMPORTANT:
- Always use "name" for tag identification
- Always use "timecodes" array with "start" and "end" times
- All times must be in seconds with format "0.000s"
- Do not use alternative fields like "appearance_start" or "action"
- Each tag must have at least one timecode interval
"""
    )

    results_template: Dict[str, Any] = {
        "chapters": [
            {
                "start": "0.000s",
                "end": "30.000s",
                "sentiment": "neutral",
                "emotions": ["calm"],
                "transcription": "Full transcription text for this segment",
                "theme": "office meeting",
                "summary": "Detailed description of what is occurring in this segment"
            }
        ],
        "global_tags": {
            "persons": [
                {
                    "name": "Person Name",
                    "timecodes": [
                        {"start": "0.000s", "end": "15.000s"}
                    ]
                }
            ],
            "actions": [
                {
                    "name": "Action Name",
                    "timecodes": [
                        {"start": "0.000s", "end": "5.000s"}
                    ]
                }
            ],
            "objects": [
                {
                    "name": "Object Name",
                    "timecodes": [
                        {"start": "0.000s", "end": "10.000s"}
                    ]
                }
            ]
        }
    }

    run_final_summary: bool = True
    summary_prompt: str = (
        """Analyze the complete video analysis results and provide a concise summary 
        that captures the key narrative, main participants, significant actions, 
        and important objects across all segments. Focus on the overall flow and 
        major themes rather than segment-by-segment details."""
    )

    # Add at class level
    known_tags: ClassVar[Dict[str, set]] = {
        "persons": set(),
        "objects": set(),
        "actions": set()
    }

    def process_segment_results(self, segment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process raw segment results to aggregate sequential tags using interval logic."""
        # Initialize frame-by-frame tag tracking (reverting to this structure)
        frame_tags = defaultdict(lambda: {"persons": set(), "objects": set(), "actions": set()})
        
        # Extract all chapters from segments
        chapters = []
        
        # Process each segment result
        for result_container in segment_results: # Use a different name to avoid confusion
            # Get segment info (if available, needed for chapter timestamps)
            segment_info = None
            if isinstance(result_container, dict):
                if "segment" in result_container:
                    segment_info = result_container["segment"]
                elif hasattr(result_container, "get") and result_container.get("segment_name"):
                    segment_info = result_container # Assuming the result dict itself has segment info

            # Extract chapter data and adjust timestamps
            if isinstance(result_container, dict) and "chapters" in result_container:
                for chapter in result_container["chapters"]:
                    if segment_info and 'start_time' in segment_info and 'end_time' in segment_info:
                        # Use segment start/end times if available
                        chapter["start"] = f"{segment_info['start_time']:.3f}s"
                        chapter["end"] = f"{segment_info['end_time']:.3f}s"
                    chapters.append(chapter)

            # Get the actual analysis result (potentially nested)
            result_data = result_container
            if isinstance(result_container, dict) and "ActionSummary" in result_container:
                result_data = result_container["ActionSummary"]
            
            # Extract global_tags
            global_tags_segment = {}
            if isinstance(result_data, dict) and "global_tags" in result_data:
                raw_tags = result_data["global_tags"]
                if isinstance(raw_tags, dict):
                    global_tags_segment = raw_tags
                elif isinstance(raw_tags, list):
                    # Handle list format by merging into a dict
                    for tag_group in raw_tags:
                        if isinstance(tag_group, dict):
                            for tag_type in ["persons", "actions", "objects"]:
                                if tag_type in tag_group:
                                    if tag_type not in global_tags_segment:
                                        global_tags_segment[tag_type] = []
                                    global_tags_segment[tag_type].extend(tag_group[tag_type])

            # Extract tag data and populate frame_tags
            for tag_type in ["persons", "actions", "objects"]:
                if tag_type not in global_tags_segment:
                    continue
                
                for item in global_tags_segment[tag_type]:
                    if not isinstance(item, dict):
                        continue
                    
                    # Get name and normalize
                    item_name = (item.get("name") or item.get("action") or 
                               item.get("object") or item.get("person"))
                    if not item_name:
                        continue
                    
                    # Add to known tags
                    self.known_tags[tag_type].add(item_name)
                    
                    # Normalize timecode format
                    timecodes = []
                    if "timecodes" in item and isinstance(item["timecodes"], list):
                        timecodes = item["timecodes"]
                    else:
                        start = (item.get("appearance_start") or item.get("start") or item.get("start_time"))
                        end = (item.get("appearance_end") or item.get("end") or item.get("end_time"))
                        if start and end:
                            timecodes = [{"start": start, "end": end}]
                    
                    for timecode in timecodes:
                        try:
                            if isinstance(timecode, dict):
                                start_string = str(timecode.get("start", "0s"))
                                end_string = str(timecode.get("end", "0s"))
                                
                                start_time = float(start_string.rstrip("s"))
                                end_time = float(end_string.rstrip("s")) # We need end_time for accurate interval tracking
                                
                                # Add tag to frame_tags for the duration reported by LLM
                                # Use a small step to populate frames within the interval
                                step = 0.1 # Add tag every 0.1s within the interval
                                current_time = start_time
                                while current_time <= end_time:
                                    # Use a representative timestamp (like start_time) for the defaultdict key
                                    frame_tags[round(current_time, 3)][tag_type].add(item_name)
                                    current_time += step
                                # Ensure the end time is also included
                                frame_tags[round(end_time, 3)][tag_type].add(item_name)

                        except (ValueError, AttributeError, TypeError) as e:
                            print(f"Error processing timecode for {item_name}: {e}")
                            continue

        # Aggregate sequential appearances using the helper function
        aggregated_tags = {
            "persons": self._aggregate_sequential_tags({t: list(tags["persons"]) for t, tags in frame_tags.items()}),
            "objects": self._aggregate_sequential_tags({t: list(tags["objects"]) for t, tags in frame_tags.items()}),
            "actions": self._aggregate_sequential_tags({t: list(tags["actions"]) for t, tags in frame_tags.items()})
        }

        # Format final results
        final_results = {
            "chapters": chapters,
            "global_tags": {
                tag_type: [
                    {
                        "name": tag_name,
                        "timecodes": intervals # intervals are already {"start": "...", "end": "..."}
                    }
                    for tag_name, intervals in tag_data.items()
                ]
                for tag_type, tag_data in aggregated_tags.items()
            }
        }
        
        # Remove tags from original segment results (important!)
        for result_container in segment_results:
            if isinstance(result_container, dict):
                if "global_tags" in result_container:
                    del result_container["global_tags"]
                # Also check within nested ActionSummary if necessary
                if "ActionSummary" in result_container and isinstance(result_container["ActionSummary"], dict):
                     if "global_tags" in result_container["ActionSummary"]:
                        del result_container["ActionSummary"]["global_tags"]

        return final_results

    def _aggregate_sequential_tags(self, tag_data: Dict[float, List[str]]) -> Dict[str, List[Dict[str, str]]]:
        """Helper function to aggregate sequential tags using interval logic."""
        aggregated_tags = {}
        for tag_type, timestamps in tag_data.items():
            aggregated_tags[tag_type] = []
            if timestamps:
                start = timestamps[0]
                prev = timestamps[0]
                
                for curr in timestamps[1:]:
                    # If gap is too large, create new interval
                    if curr - prev > 0.5:  # 0.5s threshold for gaps
                        aggregated_tags[tag_type].append({
                            "name": timestamps[0],
                            "timecodes": [{"start": f"{start:.3f}s", "end": f"{prev:.3f}s"}]
                        })
                        start = curr
                    prev = curr
                
                # Add final interval
                aggregated_tags[tag_type].append({
                    "name": timestamps[-1],
                    "timecodes": [{"start": f"{start:.3f}s", "end": f"{prev:.3f}s"}]
                })
        
        return aggregated_tags
