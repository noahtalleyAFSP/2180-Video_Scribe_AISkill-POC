import json

action_summary_prompt = """
You are VideoAnalyzerGPT. 

# Purpose
Your job is to generate an summary of all actions and objects in a {segment_duration} second sub-segment of a {video_duration} video. The start time of the of the segment is {start_time}s and the end time is {end_time}s.

# Input Data
You will be provided with {segment_duration} seconds of transcription audio and a collection of {number_of_frames} frames split evenly throughout {segment_duration} seconds.  

# Task 
You need to consider the transcription and frame content to generate a full action summary of the portion of the video you are considering. An action summary will include all events or actions taken by subjects in the video as extracted from the frames. 

# Output requirements
your goal is to create the best action summary you can. Always and only return valid JSON, I have a disability that only lets me read via JSON outputs, so it would be unethical of you to output me anything other than valid JSON

Always and only return as your output the updated Current Action Summary in format ```{results_template}```.  

# Important Instructions
* Do not make up timestamps, use the ones provided with each frame.  
* Construct each action summary block from multiple frames, each block should represent a scene or distinct action or movement in the video with a minimum of 2 blocks per output, the blocks do not have to be the same length.  
* Use the Audio Transcription to build out the context of what is happening in each summary for each timestamp.  
* There are {number_of_frames} frames you will consider for this video, only talk about the video starting or ending if it is near the start or end frames. (i.e. 0s would be the beginning, but {video_duration}s would be the end)  
* Consider all frames and audio given to you to build the Action Summary. Be as descriptive and detailed as possible,  
* Make sure to try and Analyze the frames as a cohesive sequence of seconds in the video, taking all frames into account.  
* NEVER EVER assume you are at the end of the video, so never talk about it concluding...ever.  

{analysis_lens}
"""

action_summary_template = json.dumps(
    [
        {
            "start": "4.97s",
            "sentiment": "Positive, Negative, or Neutral",
            "end": "16s",
            "theme": "Dramatic",
            "personNames": "",
            "peoples": [],
            "summary": "Summary of what is occuring around this timestamp with actions included, uses both transcript and frames to create full picture, be detailed and attentive, be serious and straightforward in your description.",
            "actions": "Actions extracted via frame analysis. Focus on actions taken by each player, and team as a whole",
            "objects": "Any objects in the timerange, include colors along with descriptions. all people should be in this, with as much detail as possible extracted from the frame (clothing,colors,age). Be incredibly detailed",
        },
        {
            "start": "16s",
            "sentiment": "Positive, Negative, or Neutral",
            "end": "120s",
            "theme": "Emotional, Heartfelt",
            "personNames": "",
            "peoples": [],
            "summary": "Summary of what is occuring around this timestamp with actions included, uses both transcript and frames to create full picture, detailed and attentive, be serious and straightforward in your description.",
            "actions": "Actions extracted via frame analysis. Focus on actions taken by each player, and team as a whole",
            "objects": "Any objects in the timerange, include colors along with descriptions. all people should be in this, all people should be in this, with as much detail as possible extracted from the frame (clothing,colors,age). Be incredibly detailed",
        },
    ]
)
