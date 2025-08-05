from .base_analysis_config import AnalysisConfig
from typing import Dict, Any, List, ClassVar, Optional, TYPE_CHECKING
from collections import defaultdict
import os
from datetime import datetime, timezone
from ..cobra_utils import seconds_to_iso8601_duration
import json
import logging
from ..models.video import Segment, VideoManifest
from ..models.environment import CobraEnvironment

logger = logging.getLogger(__name__)

# Base system prompt template that will be formatted with language-specific instructions.
# The `{{` and `}}` are used to escape single braces that are part of the JSON example.
_SYSTEM_PROMPT_TEMPLATE = """
You are VideoScribeGPT. Your task is to analyze video segments by processing a JSON object containing frame-by-frame analysis, audio transcription, and pre-identified object/person tracks.

Your goal is to synthesize this multimodal data into a rich, structured JSON output.

**Input Data Structure:**
- **Segment Context:** Start and end times of the current video segment.
- **Transcription:** Full audio transcription for the segment.
- **YOLO Tracks:** A list of objects and persons detected by a computer vision model, including their class, a unique track ID, and active timecodes within the segment.
- **Frames:** A sequence of images from the segment for your visual analysis.

**Your Task & Output Schema:**
Based on all the provided information, generate a single chapter summary and a list of global tags for the segment. Your output MUST be a single, valid JSON object enclosed in ```json ... ```, strictly adhering to the following schema:

```json
{{
  "chapters": [
    {{
      "start": "0.000s",
      "end": "30.000s",
      "summary": "A concise, detailed summary of all significant events, actions, and spoken content within this single video segment. Mention key persons and objects involved.",
      "shotDescription": "A brief, present-tense description of the main visual elements and camera work in the shot (e.g., 'Medium shot of a person speaking at a desk')."
    }}
  ],
  "globalTags": {{
    "persons": [
      {{
        "id": 1,
        "classDescription": "Man in blue shirt",
        "yoloClass": "person",
        "timecodes": [{{ "start": "1.234s", "end": "15.678s" }}]
      }}
    ],
    "actions": [
      {{
        "classDescription": "Speaking",
        "timecodes": [{{ "start": "1.234s", "end": "15.678s" }}]
      }}
    ],
    "objects": [
      {{
        "id": 2,
        "classDescription": "Laptop computer",
        "yoloClass": "laptop",
        "timecodes": [{{ "start": "0.500s", "end": "29.500s" }}]
      }}
    ]
  }}
}}
```

**Instructions & Rules:**
{language_specific_instructions}
"""

# Language-specific instruction sets
_LANGUAGE_INSTRUCTIONS = {
    "en": """
1.  **`chapters.summary`**: Write in clear, descriptive English. Synthesize visuals, audio, and text to create a comprehensive narrative of the segment.
2.  **`chapters.shotDescription`**: Describe the visual composition in English.
3.  **`globalTags`**:
    *   **`persons` / `objects`**: Use the `classDescription` from the input YOLO tracks. If a track ID is present but lacks a good description, create a concise one (e.g., "Woman in red dress"). Ensure all significant, tracked persons/objects are listed.
    *   **`actions`**: Identify and list key actions (e.g., "talking", "walking", "presenting").
    *   **`timecodes`**: Accurately copy the start/end timecodes for each tag from the input data corresponding to this segment.
4.  Ensure all output text is in English.
""",
    "fr": """
1.  **`chapters.summary`**: Rédigez en français clair et descriptif. Synthétisez les visuels, l'audio et le texte pour créer un récit complet du segment.
2.  **`chapters.shotDescription`**: Décrivez la composition visuelle en français.
3.  **`globalTags`**:
    *   **`persons` / `objects`**: Utilisez la `classDescription` des pistes YOLO fournies. Si un ID de piste est présent mais sans bonne description, créez-en une concise (par ex., "Femme en robe rouge"). Assurez-vous que toutes les personnes/objets importants et suivis sont listés.
    *   **`actions`**: Identifiez et listez les actions clés (par ex., "parler", "marcher", "présenter").
    *   **`timecodes`**: Copiez précisément les timecodes de début/fin pour chaque tag à partir des données d'entrée correspondant à ce segment.
4.  Assurez-vous que tout le texte en sortie est en français.
""",
    "ar": """
1.  **`chapters.summary`**: اكتب ملخصًا واضحًا ومفصلاً باللغة العربية الفصحى. قم بتوليف العناصر المرئية والصوتية والنصية لإنشاء سرد شامل للقطعة.
2.  **`chapters.shotDescription`**: صف التكوين البصري باللغة العربية الفصحى.
3.  **`globalTags`**:
    *   **`persons` / `objects`**: استخدم `classDescription` من مسارات YOLO المدخلة. إذا كان معرف المسار موجودًا ولكنه يفتقر إلى وصف جيد، فأنشئ وصفًا موجزًا (على سبيل المثال، "امرأة ترتدي فستانًا أحمر"). تأكد من إدراج جميع الأشخاص/الكائنات المهمة التي تم تتبعها.
    *   **`actions`**: حدد الأفعال الرئيسية وأدرجها (على سبيل المثال، "يتحدث"، "يمشي"، "يقدم عرضًا").
    *   **`timecodes`**: انسخ بدقة الطوابع الزمنية للبداية/النهاية لكل علامة من بيانات الإدخال المقابلة لهذه القطعة.
4.  تأكد من أن كل النصوص الناتجة باللغة العربية الفصحى.
""",
    "ary": """
1.  **`chapters.summary`**: كتب ملخص واضح ومفصل بالدارجة المغربية. جمع ما بين الصور، الصوت، والنص باش تصاوب قصة كاملة على هاد الجزء.
2.  **`chapters.shotDescription`**: وصف لي شنو كيبان ف الصورة بالدارجة المغربية.
3.  **`globalTags`**:
    *   **`persons` / `objects`**: ستعمل `classDescription` لي جاية من YOLO. يلا كان شي ID ديال شي حاجة ولكن ماعندوش وصف مقاد، صاوب واحد قصير (بحال "مرأة لابسة كسوة حمرا"). تأكد من أن كاع الناس والحوايج المهمين لي تتبعو مكتوبين.
    *   **`actions`**: شوف شنو هي الأفعال المهمة وكتبها (بحال "كايهضر"، "كيتمشى"، "كيشرح شي حاجة").
    *   **`timecodes`**: نقل الوقت ديال البداية والنهاية ديال كل علامة من داكشي لي تعطا ليك فهاد الجزء.
4.  تأكد من أن كاع الكتابة لي غتخرج تكون بالدارجة المغربية.
"""
}

class ActionSummary(AnalysisConfig):
    """
    This AnalysisConfig enforces that the LLM must produce a JSON response with
    top-level 'chapters' and 'globalTags' keys. The 'chapters' must have exactly
    one object describing the segment, and 'globalTags' must always include
    'persons', 'actions', and 'objects' arrays, even if they are empty.
    
    The code below also includes a process_segment_results() method that merges
    any segment-level tags into continuous intervals at second-level accuracy.
    """

    name: str = "ActionSummary"
    analysis_sequence: str = "mapreduce"

    # --- ADDED: Configurable list of shot types ---
    SHOT_TYPES: ClassVar[List[str]] = [
        "Establishing Shot", "Extreme Wide Shot (EWS)", "Wide Shot (WS)", "Full Shot (FS)",
        "Medium Wide Shot (MWS)", "Medium Long Shot (MLS)", "Medium Shot (MS)", "Cowboy Shot",
        "Medium Close-Up (MCU)", "Close-Up (CU)", "Extreme Close-Up (ECU)", "Two-Shot",
        "Three-Shot", "Reverse Angle", "Over-the-Shoulder", "Point-of-View (POV) Shot",
        "Reaction Shot", "Insert Shot", "Cutaway", "Dutch Angle", "Tracking/Dolly Shot",
        "Crane/Jib Shot", "Handheld/Steadicam Shot", "Whip-Pan (Swish-Pan)", "Special / Other"
    ]

    # --- ADDED: Configurable list of Asset Categories ---
    ASSET_CATEGORIES: ClassVar[List[str]] = [
        "Sports", "Drama", "Comedy", "News", "Documentary", "Social Media",
        "Commercial/Advertisement", "Educational", "Music Video", "Gaming",
        "Lifestyle/Vlog", "Technology", "Travel", "Other"
    ]

    # This method will be used by VideoAnalyzer to get the correct, language-formatted prompt.
    def get_system_prompt(self, language_code: Optional[str] = 'en') -> str:
        """
        Returns the system prompt template formatted with the appropriate language instructions.
        """
        # Default to English if the language code is not supported or not provided.
        safe_lang_code = language_code if language_code in _LANGUAGE_INSTRUCTIONS else 'en'
        instructions = _LANGUAGE_INSTRUCTIONS[safe_lang_code]
        return _SYSTEM_PROMPT_TEMPLATE.format(language_specific_instructions=instructions)

    # The `system_prompt_template` is now dynamically generated, but we can keep a default.
    @property
    def system_prompt_template(self) -> str:
        return self.get_system_prompt('en')
    
    results_template: Dict[str, Any] = {
        "chapters": [
            {
                "start": "0.000s",
                "end": "30.000s",
                "summary": "",
                "shotDescription": ""
            }
        ],
        "globalTags": {
            "persons": [],
            "actions": [],
            "objects": []
        }
    }
    
    # Whether we run a final summary across all segments after everything finishes
    run_final_summary: bool = True

    # The prompt used for the final summary across the entire video
    summary_prompt:  ClassVar[str] = (
        """Analyze the complete video analysis results (chapters and tags) provided below to understand the video's content.
        Your task is to generate TWO summaries and return them ONLY as a single JSON object.

        Additionally, classify the overall video content:
        *   `category`: Choose the **single most appropriate** category for the entire video from this list: {asset_categories_list}
        *   `subCategory`: (Optional) If applicable and obvious, provide a specific sub-category (e.g., for 'Sports', maybe 'Basketball Highlights'; for 'News', maybe 'Political Report'). **DO NOT invent a sub-category if one isn't clear.** Leave it as `null` or omit the key if unsure.

        1.  **description**: A very concise, 1-2 sentence description summarizing the absolute core gist or main topic of the video content. Focus on what the video is primarily *about*. Avoid listing specific details unless essential for the core topic.
        2.  **summary**: A detailed summary that captures the key narrative, main participants, significant actions, and important objects across all segments. Focus on the overall flow and major themes rather than segment-by-segment details. **The summary should be {summary_length_instruction}**

        CRITICAL: You MUST return ONLY a valid JSON object containing `description`, `summary`, `category`, and optionally `subCategory`. Example format:
        ```json
        {{
          "category": "Sports",
          "subCategory": "Basketball Highlights",
          "description": "A 1-2 sentence description...",
          "summary": "The detailed summary..."
        }}
        ```
        Do not include any text outside of this JSON structure."""
    )
