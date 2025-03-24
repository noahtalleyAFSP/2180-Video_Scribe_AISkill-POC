# Video Scribe - Advanced Video Analysis System

## Skill Description
Video Scribe is an advanced AI-powered video analysis system that extracts detailed metadata from video content. It provides rich, structured information about scenes, objects, people, actions, emotions, and themes, enabling comprehensive cataloging and search capabilities for video content.

## Goal of the Pipeline
The goal is to transform raw video files into structured JSON metadata that describes the content in detail, with precise timestamped segments for better searchability and understanding.

## LLM and Resources Used
- **Primary AI Model**: Azure OpenAI GPT-4o with Vision capabilities
- **Face Recognition**: Azure Face API for person identification
- **Authentication**: Azure Active Directory for secure access
- **Storage**: Local file system for processed results and intermediate files

## Inputs (Variables, Parameters, and Files)

### Command Line Parameters
```
python analyze_video.py <video_path> [options]

Options:
  --face-model        Azure Face API person group ID for custom face recognition
  --peoples-list      Path to a JSON file containing people to identify in the video
  --emotions-list     Path to a JSON file containing emotions to detect in the video
  --objects-list      Path to a JSON file containing objects to detect in the video
  --themes-list       Path to a JSON file containing themes to classify in the video
  --actions-list      Path to a JSON file containing actions to detect in the video
  --lens              Path to a JSON file containing a custom analysis lens
  --fps               Frames per second to extract (default: 0.33)
  --segment-length    Length of video segments in seconds (default: 10)
```

### Input Files Format

#### Video File
- Supports standard video formats (MP4, MOV, AVI)
- No specific resolution requirements, though higher quality improves analysis

#### peoples_list.json (Optional)
```json
{
    "instructions": "You are analyzing an NBA game. Look for players based on their jersey numbers and team uniforms to identify them in the video.",
    "peoples": [
        {
            "name": "LeBron James",
            "description": "Player wearing Lakers purple and gold jersey with number 23"
        },
        {
            "name": "Austin Reaves",
            "description": "Player wearing Lakers purple and gold jersey with number 15"
        }
    ]
}
```

#### emotions_list.json (Optional)
```json
{
    "instructions": "Analyze each video segment for the following emotions and report them in the emotions field.",
    "emotions": [
        {
            "name": "Joy",
            "description": "Expressions of happiness, delight, elation, or pleasure. Look for smiles, laughter, and celebrating."
        },
        {
            "name": "Excitement",
            "description": "Expressions of enthusiasm or eagerness. Look for animated gestures, wide eyes, or energetic movements."
        }
    ]
}
```

#### objects_list.json (Optional)
```json
{
    "instructions": "Identify the following objects in the video frames and include them in the objects field.",
    "objects": [
        {
            "name": "Basketball",
            "description": "Orange spherical ball used in the game, typically with black lines/grooves"
        },
        {
            "name": "Basketball Court",
            "description": "Wooden floor with markings including three-point line, free throw line, and center circle"
        }
    ]
}
```

#### themes_list.json (Optional)
```json
{
    "instructions": "Identify the overall theme of each video segment from the following list.",
    "themes": [
        {
            "name": "Fast-Paced Action",
            "description": "Rapid movement, high energy plays, quick transitions"
        },
        {
            "name": "Strategic Play",
            "description": "Deliberate positioning, planned movements, tactical decision-making"
        }
    ]
}
```

#### actions_list.json (Optional)
```json
{
    "instructions": "Identify specific basketball actions occurring in the video frames.",
    "actions": [
        {
            "name": "Jump Shot",
            "description": "Player jumps and shoots the ball with proper form and follow-through"
        },
        {
            "name": "Layup",
            "description": "Player approaches the basket and gently lays the ball off the backboard or directly into the hoop"
        }
    ]
}
```

#### lens.json (Optional)
```json
{
    "lens": "You are a sportscaster renowned for your entertaining play-call style. Use energetic, colorful language and sports-specific terminology. Emphasize dramatic moments and player achievements with excitement."
}
```

#### Environment File (.env)
Required environment variables:
```
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2023-09-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_FACE_ENDPOINT=https://your-face-endpoint.cognitiveservices.azure.com/
AZURE_FACE_API_KEY=your-face-api-key
AZURE_FACE_API_VERSION=1.0
```

## API Calls and Payload

### OpenAI Vision API
For each video segment, the system makes calls to the Azure OpenAI Vision API:

```python
response = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": "Analyze this video segment."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_content}"}},
            # Additional frames with timestamps
        ]}
    ],
    max_tokens=2000,
)
```

### Azure Face API
For person identification:

```python
# Detection
detect_response = face_client.face.detect_with_stream(
    image=image_data,
    detection_model='detection_03',
    recognition_model='recognition_04',
    return_face_attributes=['age', 'gender', 'headPose', 'smile', 'facialHair', 'glasses']
)

# Identification
identify_response = face_client.face.identify(
    face_ids=[face.face_id for face in detect_response],
    person_group_id=person_group_id
)
```

## Preprocessing Requirements
The system preprocesses videos before analysis:

1. **Video Segmentation**: Divides videos into configurable length segments (default 10 seconds)
2. **Frame Extraction**: Captures frames at specified intervals (default 0.33 FPS)
3. **Audio Extraction**: (Optional) Extracts audio for transcription
4. **Face Processing**: (Optional) Processes frames for face detection and recognition

**Libraries Used**:
- FFmpeg for video and audio processing
- OpenCV for frame extraction and image processing
- Azure Speech SDK for transcription (if enabled)

## Expected Output

The system produces a structured JSON output with detailed information about the video content:

```json
[
    {
        "start": "0.000s",
        "sentiment": "Positive",
        "end": "5.000s",
        "theme": "Sports Action",
        "personNames": "LeBron James, Austin Reaves",
        "peoples": ["LeBron James", "Austin Reaves"],
        "emotions": ["Excitement", "Determination"],
        "objects": ["Basketball", "Basketball Court", "Lakers Jersey"],
        "actions": ["Jump Shot", "Rebound", "Pass"],
        "summary": "LeBron James drives toward the basket with determination while Austin Reaves positions himself near the three-point line. The crowd cheers enthusiastically as the Lakers execute their offensive play.",
        "actions": "LeBron dribbles the ball forcefully, transitions into a driving maneuver toward the basket, while Reaves performs a cutting motion toward the open space.",
        "objects": "Basketball (orange), Lakers jerseys (purple and gold), court (polished wood with team logos), digital scoreboard displaying game time and score."
    },
    {
        "start": "5.000s",
        "sentiment": "Neutral",
        "end": "10.000s",
        "theme": "Team Strategy",
        "personNames": "LeBron James",
        "peoples": ["LeBron James", "Austin Reaves"],
        "emotions": ["Concentration", "Determination"],
        "objects": ["Basketball", "Basketball Court", "Scoreboard"],
        "actions": ["Pass", "Defensive Stance", "Screen"],
        "summary": "The play continues with strategic positioning. LeBron passes the ball while teammates adjust their formation on the court.",
        "actions": "LeBron executes a precise chest pass, players shift positions according to their offensive strategy, defensive players attempt to intercept.",
        "objects": "Basketball, players in various colored jerseys, referee in black and white striped uniform, court markings clearly visible."
    }
]
```

## Chunking and Result Stitching

The system employs a parallelized processing approach to handle videos of any length:

1. **Segmentation**: Videos are split into manageable segments (configurable, default 10 seconds)
2. **Parallel Processing**: Each segment is analyzed independently, with configurable concurrency limits
3. **Result Compilation**: Results from individual segments are combined into a comprehensive manifest
4. **Face Recognition Batching**: Face API calls are batched to optimize processing time and stay within rate limits
5. **Error Handling**: Robust error handling for network issues, timeouts, or quality problems with face recognition

### Performance Considerations:
- System supports async processing for optimal performance
- Configurable FPS allows balancing between processing speed and analysis depth
- Smart chunking ensures complete context preservation at segment boundaries
- Self-healing mechanism for handling transient API failures

## Custom Analysis Lens

The lens feature allows you to completely customize the analysis style by providing a specific perspective or voice through which the video is analyzed. This doesn't change the output structure but influences how the content is described and what aspects are emphasized.

Examples of lens applications:
- Sports commentator style
- Technical analysis focus
- Cinematic/film criticism perspective
- Educational/instructional emphasis

The lens is applied at the system level but doesn't appear in the final JSON output - it only affects how the system interprets and describes the video content.

## Installation and Setup

1. Clone the repository:
```
git clone https://github.com/your-username/video-scribe.git
cd video-scribe
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Set up environment variables (create a .env file with required variables)

4. Run the analysis:
```
python analyze_video.py path/to/your/video.mp4 --face-model your_face_model_id --peoples-list peoplesList.json --emotions-list emotionsList.json --objects-list objectsList.json --themes-list themesList.json --actions-list actionsList.json --lens lens.json
```

## Requirements

- Python 3.8+
- FFmpeg (for video preprocessing)
- Azure subscription with OpenAI and Face API access
- Required Python packages (see requirements.txt)

## OVERVIEW

Video Scribe is an advanced computer vision and AI-powered tool designed to extract meaningful insights from video content. The system leverages Azure's Face API for face recognition, OpenAI's vision models for scene understanding, and custom processing for temporal analysis. It analyzes video files frame-by-frame to detect people, identify objects, interpret actions, and provide temporal summaries of content. The system outputs structured JSON data that catalogues who appears in the video, what is happening in each scene, notable objects, sentiment, and thematic elements, making it ideal for content cataloguing, video indexing, and automated description generation.

## WORKFLOW

1. **Video Input & Preprocessing**:
   - The system accepts a video file and optional configuration parameters
   - The video is divided into segments of configurable duration
   - Frames are extracted at a specified rate (frames per second)

2. **Face Recognition**:
   - When a custom face model is provided, the system uses Azure Face API
   - Each extracted frame is analyzed for faces
   - Identified faces are matched against the provided face group model
   - Recognition confidence and metadata are recorded

3. **Vision Analysis**:
   - Extracted frames are processed through multimodal vision models
   - Each segment receives a detailed analysis including:
     - Scene understanding and object detection
     - Action recognition
     - Contextual interpretation 
     - Theme categorization
     - Emotion detection
     - Custom object tracking

4. **Temporal Mapping**:
   - Analyses are mapped to specific timestamps with millisecond precision (e.g., "10.023s")
   - Face recognition data is aligned with timestamps
   - Identified people are associated with specific time segments

5. **JSON Generation**:
   - Results are compiled into structured JSON formats
   - Field normalization and UTF-8 encoding ensure proper character representation
   - Final output includes video manifest and detailed action summaries

## INPUT VARIABLES

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `video_file` | Path to input video file | (Required) | "C:\path\to\video.mp4" |
| `--face-model` | Custom face group ID for recognition | None | "custom_c72ed963" |
| `--peoples-list` | JSON file with people to identify | None | "peoplesList.json" |
| `--emotions-list` | JSON file with emotions to detect | None | "emotionsList.json" |
| `--objects-list` | JSON file with objects to identify | None | "objectsList.json" |
| `--themes-list` | JSON file with themes to classify | None | "themesList.json" |
| `--actions-list` | JSON file with actions to detect | None | "actionsList.json" |
| `--lens` | Custom analysis perspective | None | "lens.json" |
| `--fps` | Frame extraction rate | 0.33 | 1.0 |
| `--segment-length` | Segment duration in seconds | 10 | 30 |

## DATA MODEL

### Video Manifest Structure

The video manifest is the primary output, containing metadata and analysis results:

```json
{
  "source_video": {
    "file_path": "string",
    "duration": "float",
    "width": "int",
    "height": "int",
    "fps": "float",
    "creation_time": "datetime"
  },
  "processing_info": {
    "segment_duration": "float",
    "analysis_fps": "float",
    "start_time": "datetime",
    "end_time": "datetime",
    "total_frames_processed": "int",
    "face_model_id": "string (optional)"
  },
  "segments": [
    {
      "segment_id": "string",
      "start_time": "float",
      "end_time": "float",
      "segment_duration": "float",
      "number_of_frames": "int",
      "segment_frames_file_path": ["string"],
      "segment_folder_path": "string",
      "analyzed_result": {
        "ActionSummary": [
          {
            "start": "string",
            "end": "string",
            "sentiment": "string",
            "theme": "string",
            "personNames": "string",
            "summary": "string",
            "actions": "string",
            "objects": "string"
          }
        ]
      }
    }
  ]
}
```

### Action Summary Structure

The action summary provides detailed analysis for each segment:

```json
[
  {
    "start": "0.0s",
    "end": "5.93s",
    "sentiment": "Neutral",
    "theme": "Casual",
    "personNames": "John Doe, Jane Smith",
    "summary": "Detailed description of what occurs in this segment",
    "actions": "Specific actions observed in the sequence",
    "objects": "Objects identified with details including colors and attributes"
  }
]
```

## BUSINESS RULES FOR OBJECT PROCESSING

1. **Face Recognition**:
   - Only individuals with a confidence score above 0.5 are included in identification results
   - The `personNames` field only contains names of individuals identified by the Azure Face API
   - If multiple people are identified in a segment, names are comma-separated

2. **Name Handling**:
   - Names from face recognition appear exclusively in the `personNames` field
   - The system avoids using custom-recognized names in other descriptive fields
   - Public figures recognized by the vision model can be mentioned naturally in descriptions

3. **Temporal Alignment**:
   - Time values are represented in seconds with format "X.XXs"
   - People are associated with specific timestamps where they appear
   - Each segment contains time-specific analysis

4. **UTF-8 Handling**:
   - All output uses UTF-8 encoding to properly handle special characters
   - Non-ASCII characters are preserved in their original form, not escaped

5. **Theme Classification**:
   - Each segment is assigned a theme based on visual and contextual cues
   - Themes are concise descriptors (e.g., "Dramatic", "Casual", "Formal")

## JSON OUTPUT STRUCTURE

### Complete Video Manifest:

```json
{
  "source_video": {
    "file_path": "C:\\path\\to\\video.mp4",
    "duration": 120.5,
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "creation_time": "2023-07-15T14:30:22.456123"
  },
  "processing_info": {
    "segment_duration": 10.0,
    "analysis_fps": 0.33,
    "start_time": "2023-07-15T15:45:12.123456",
    "end_time": "2023-07-15T15:48:35.654321",
    "total_frames_processed": 40,
    "face_model_id": "custom_c72ed963"
  },
  "segments": [
    {
      "segment_id": "segment_0",
      "start_time": 0.0,
      "end_time": 10.0,
      "segment_duration": 10.0,
      "number_of_frames": 4,
      "segment_frames_file_path": [
        "path/to/frame_0.jpg",
        "path/to/frame_1.jpg",
        "path/to/frame_2.jpg",
        "path/to/frame_3.jpg"
      ],
      "segment_folder_path": "path/to/segment_0/",
      "analyzed_result": {
        "ActionSummary": [
          {
            "start": "0.0s",
            "end": "5.0s",
            "sentiment": "Positive",
            "theme": "Casual",
            "personNames": "John Doe",
            "summary": "A person enters the room and positions themselves in front of the camera. The environment appears calm and casual.",
            "actions": "The individual walks into the frame and adjusts their position slightly.",
            "objects": "A desk with a lamp, a computer monitor displaying a presentation, and a bookshelf with various colored books are visible in the background."
          },
          {
            "start": "5.0s",
            "end": "10.0s",
            "sentiment": "Neutral",
            "theme": "Informational",
            "personNames": "John Doe, Jane Smith",
            "summary": "Two individuals are now visible in the frame, appearing to discuss something of importance.",
            "actions": "One person is gesturing toward a document while the other is nodding in agreement.",
            "objects": "A document with charts is now visible on the desk. Both individuals are dressed in business casual attire, with one wearing a blue shirt and the other a gray blazer."
          }
        ]
      }
    }
  ]
}
```

### Action Summary Output:

```json
[
  {
    "start": "0.0s",
    "end": "2.97s",
    "sentiment": "Neutral",
    "theme": "Casual",
    "personNames": "Noah Talley",
    "summary": "An individual enters the room and positions himself slightly in front of a camera setup. The environment appears calm and there seems to be no significant conversation or audio cues at this moment.",
    "actions": "The person is seen entering the room and aligning himself with the camera viewpoint.",
    "objects": "A shelf filled with water bottles, a floor lamp emitting warm light, and a bed with white sheets are present in the frame. The camera is directed towards these elements, providing a view of the room's setup."
  },
  {
    "start": "2.97s",
    "end": "5.93s",
    "sentiment": "Positive",
    "theme": "Informal",
    "personNames": "Noah Talley",
    "summary": "An individual presents a favorable gesture to the camera by raising his right hand, forming a thumbs-up sign. This gesture is typically associated with approval or agreement, indicating a positive sentiment in the scene.",
    "actions": "The person raises his hand to show a thumbs-up gesture, expressing a positive sentiment or approval.",
    "objects": "The previous objects remain consistent with the scene: the shelf with water bottles, the floor lamp, and the bed with white sheets. The person is additionally seen displaying the thumbs-up gesture as a new element in this frame."
  }
]
```

## FUTURE ITEMS

1. **Enhanced Audio Analysis**:
   - Implement more sophisticated audio transcription
   - Incorporate audio sentiment analysis
   - Detect environmental sounds and classify them

2. **Advanced Object Tracking**:
   - Track object persistence across frames
   - Identify object interactions and relationships
   - Calculate object movement patterns

3. **Multi-language Support**:
   - Provide analysis output in multiple languages
   - Support non-English video content analysis

4. **Interactive Video Timeline**:
   - Develop a UI for navigating analyzed content
   - Enable searching video by person, object, or action
   - Display analysis overlays on video playback

5. **Relationship Detection**:
   - Identify relationships between people in the video
   - Detect social dynamics and interaction patterns

6. **Custom Theme Training**:
   - Allow users to train custom theme recognition
   - Enable domain-specific theme categorization

7. **Batch Processing**:
   - Support for processing multiple videos concurrently
   - Queue management for large video libraries

## CONCLUSION

The Video Analysis System represents a powerful integration of computer vision, face recognition, and AI-powered content understanding. By transforming unstructured video content into rich, structured data, the system enables a wide range of applications from content indexing to accessibility enhancement.

The modular design allows for configuration flexibility while maintaining consistent output formats. The separation of face recognition from general content description ensures privacy considerations while still providing detailed analysis. As video content continues to grow exponentially across industries, systems like this provide essential tools for managing, searching, and deriving insights from visual media at scale.

With planned enhancements addressing audio analysis, object tracking, and user interaction, the system is positioned to evolve into an even more comprehensive media analysis platform that can adapt to diverse industry needs and use cases.

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.