from .base_analysis_config import AnalysisConfig


class BasicSummary(AnalysisConfig):
    name: str = "BasicSummary"
    analysis_sequence: str = "mapreduce"
    system_prompt: str = (
        """Your job is to analyze a video clip following the instructions below.
## Video and Clip Details:
* overall video is {video_duration} seconds long. 
* the current subclip is {segment_duration} seconds long, starting at {start_time} seconds and ending at {end_time} seconds. 
* You may or may not receive both corresponding audio transcriptions and frame images. 
* {number_of_frames} frame images spread evenly throughout the subclip are provided seconds. 
Use this information in your analysis."""
    )
    lens_prompt: str = (
        """Use the images and any provided transcription to summarize the what is happening in the video clip. For each clip, you will be asked to provide an overall summary in a few sentences, and also to identify key moments in the clip with their approximate time stamp (in seconds)"""
    )
    outputs_prompt: str = (
        """Your response should always be in the form of a json object represented in markdown with a ```json``` code block. It should be properly formed and use double quotes for keys and string values.

Follow this template as a guide of what information to collect and include in your response:

## Template for results
{results_template}"""
    )
    results_template: dict = {
        "segment_summary": "Text summary of what is happening in the clip",
        "segment_start_time": "10s",
        "segment_end_time": "20s",
        "moments": [
            {
                "event_start_time": "11s",
                "summary": "The main character picks up an apple",
            },
            {
                "event_start_time": "13s",
                "summary": "The main character throws the apple at a clock",
            },
        ],
    }
    run_final_summary: bool = False
    summary_prompt: str = (
        """You are being provided the output of of a video analysis system. Your job is to summarize the video based on the provided inputs. Return the summary as a single paragraph."""
    )
