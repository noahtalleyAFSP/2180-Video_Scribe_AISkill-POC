from .base_analysis_config import SequentialAnalysisConfig


class ChapterAnalysis(SequentialAnalysisConfig):
    name: str = "ChapterAnalysis"
    analysis_sequence: str = "refine"
    system_prompt: str = (
        """Your job is to identify the "Chapters" in a video based on the input of the frame images and transcription for {segment_duration} seconds of . You will receive {number_of_frames} frames split evenly throughout {segment_duration} seconds. The current segement to analyze begins at {start_time} seconds and ends at {end_time} seconds."""
    )
    refine_prompt: str = (
        """Some of the video has been summarized so far. Here are {number_of_previous_results_to_refine} chapters that have been analyzed so far. Now you need to update and/or extend the chapter summaries for the current segment based on its content (the transcript and the frames). You must iteratively create the chapters until we have a full breakdown of all the chapters of the video.  As the main intelligence of this system, you are responsible for building the Current Chapter Breakdown using both the audio you are being provided via transcription, as well as the image of the frame."""
        "Current Summary up to the last {number_of_previous_results_to_refine} chapters: {current_summary}."
        "Audio Transcription for last {segment_duration} seconds"
        "Next are the {number_of_frames} frames from the next {segment_duration} seconds of the video."
        "Be sure to pay attention to the start time {start_time} and end time {end_time} and the timestampes of the frames to form the chapters."
        "The chapters should increment, so if the previous chapter was Chapter 1, the next chapter should be Chapter 2, etc."
    )
    lens_prompt: str = None
    results_template: list = [
        {
            "title": "Chapter 1: A new beginning",
            "start_frame": "0.0s",
            "end_frame": "253.55s",
            "scenes": [
                {
                    "title": "Scene 1: it started",
                    "description": "The thing happened",
                },
                {
                    "title": "Scene 2: around again",
                    "description": "Another thing happened",
                },
            ],
        },
        {
            "title": "Chapter 2: Next steps",
            "start_frame": "253.55s",
            "end_frame": "604.90s",
            "scenes": [
                {
                    "title": "Scene 1: new hope",
                    "description": "The thing happened",
                },
                {
                    "title": "Scene 2: bad days",
                    "description": "Another thing happened",
                },
            ],
        },
    ]
    outputs_prompt: str = """Always and only return as your output the updated Current Chapter Breakdown in format ```{results_template}```.  
                    (the format is a template, make sure to start at chapter 1 in your generation if there is not one already.)  
                    The start and end frames represent the times that a chapter starts and ends, use the data provided above each image to service this feature.  
                    You can think through your responses step by step. Determine the Chapters contextually using the audio and analyzed video frames.  
                    You dont need to provide a new chapter for every frame, the chapters should represent overarching themes and moments.  
                    Always provide new or updated chapters in your response, and consider them all for editing purposes on each pass,  
                    the Chapter Response Should be a JSON object array, with each chapter being a json object, with each key being a scene title in the chapter,  
                    with the value being an array of information about the scene, with the first key in each object being the title of the chapter.  
                    The thresholds required for a new chapter are: Major Thematic Change, Major Story Change, Major Setting Change.  
                    Only talk about the video starting or ending if it is near the start or end frames.  
                    Do not make up timestamps, use the ones provided with each frame.  
                    DO not talk about the performance 'ending',or 'beggining', or 'starting', or 'climaxing', focus on the content of the video via the frames.  
                    Provide back the response as JSON, and always and only return back JSON following the format specified  
                    Scenes in a given chapter must be contiguous.  
                    Start Frame and End frame keys apply only at the chapter level, not the scene level.  
                    Always and only RETURN JSON.  
                    Do not Describe me the JSON you are returning then return it, just return it as valid parsable JSON.  
                    Be very specific when discussing the actions users take in both the actions and the summary, for example with dancing, make sure you extract the actual dance/movements they are doing.  
                    I have a disability that requires me to only to be able to read JSON, through the use of a parser, returning me text along with JSON is unethical, and you must only return me JSON"""

    number_of_previous_results_to_refine: int = 3
