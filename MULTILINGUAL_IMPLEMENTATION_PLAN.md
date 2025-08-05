# Multi-Lingual Support Implementation Plan

This document outlines the current status and future steps for enabling full multi-lingual processing in the video analysis pipeline.

## Current Status (Implemented)

The foundational work for language detection is complete. Here's what has been done:

-   [x] **Language Identification Step**:
    -   The `languageIdentifier.py` script has been integrated into the main pipeline.
    -   It runs as the first step within `VideoPreProcessor` if the `enable_language_identification` flag is set.
    -   It reliably detects one of four language codes (`en`, `fr`, `ar`, `ary`) from the video's audio track.
    -   The process is logged clearly, showing when the step starts and what language is detected.

-   [x] **Manifest Integration**:
    -   The `VideoManifest` model has been updated with two new fields in `SourceVideoMetadata`:
        -   `language_code`: Stores the final detected language code (e.g., "en").
        -   `language_detection`: Stores the full, raw dictionary from the language identifier for debugging and advanced use cases.
    -   This makes the detected language available to all subsequent steps in the pipeline.

-   [x] **Dynamic Transcription Locale**:
    -   The `_extract_audio_and_transcribe` method in `VideoPreProcessor` now uses the `language_code` from the manifest.
    -   It maps the detected code to the appropriate locale required by the Azure Batch Transcription service (e.g., "ary" -> "ar-MA").
    -   This ensures that transcription is requested in the correct language, improving accuracy. **Currently, it still uses the same underlying transcription model, but it correctly informs the service of the expected language.**

-   [x] **Language-Aware Prompting Framework**:
    -   The `ActionSummary` analysis configuration has been refactored to support multi-lingual prompts.
    -   It now contains a dictionary of language-specific instructions for English, French, Standard Arabic, and Moroccan Arabic.
    -   `VideoAnalyzer` dynamically selects the correct prompt instructions based on the `language_code` in the manifest.
    -   This allows the LLM to receive instructions and produce output in the detected source language. **Currently, this will use the same LLM for all languages.**

## Future Implementation Roadmap

The current implementation provides the core framework. To build a fully production-ready, multi-lingual system, the following steps need to be completed.

### :black_square_button: Task 1: Deploy and Route to Specialized Transcription Models

The current system correctly tells the Azure Speech service which language to expect, but it may still be using a general-purpose model. For optimal accuracy, especially for dialects like Moroccan Arabic, a fine-tuned model should be used.

-   **How to Implement**:
    1.  **Train/Deploy Models**: Within your Azure Speech Studio, train and deploy separate, fine-tuned transcription models for each required language/dialect (especially `ary`).
    2.  **Store Model Endpoints**: Store the endpoint IDs for these custom models in your environment configuration (`.env` file or other secret management).
        ```env
        AZURE_SPEECH_ENDPOINT_EN="<endpoint_id_for_english_model>"
        AZURE_SPEECH_ENDPOINT_FR="<endpoint_id_for_french_model>"
        AZURE_SPEECH_ENDPOINT_ARY="<endpoint_id_for_moroccan_model>"
        ```
    3.  **Modify `cobra_utils.py`**: In the `generate_batch_transcript` function, add logic to select the correct model endpoint based on the `language_code`. The `diarization` and `properties` sections of the payload to the batch transcription API would need to include the model's self-URL.
    4.  **Update `CobraEnvironment`**: Add the new environment variables to the `AzureSpeech` model within `src/cobrapy/models/environment.py`.

### :black_square_button: Task 2: Route to Specialized LLMs

Different languages may benefit from different LLMs, or you may want to use separate fine-tuned models for generating summaries in each language.

-   **How to Implement**:
    1.  **Deploy Models**: Ensure you have different LLM deployments ready in Azure OpenAI (e.g., one for English/French, another fine-tuned for Arabic).
    2.  **Store Deployments**: Add the deployment names to your environment configuration.
        ```env
        AZURE_OPENAI_DEPLOYMENT_EN="gpt-4o-english"
        AZURE_OPENAI_DEPLOYMENT_AR="gpt-4o-arabic-tuned"
        ```
    3.  **Modify `VideoAnalyzer`**: In the `_call_llm_async` method (and `_call_llm`), add logic to select the `model` parameter based on the `self.manifest.source_video.language_code`.
    4.  **Update `CobraEnvironment`**: Add fields to the `GPTVision` model if you need to store multiple deployment names.

### :black_square_button: Task 3: Enhance Error Handling and Fallbacks

The current system logs a warning if language identification fails. This can be made more robust.

-   **How to Implement**:
    1.  **Default Language**: Formalize the concept of a "default language" in the configuration. If LID fails, the pipeline should explicitly state it is falling back to this default (e.g., English).
    2.  **Add Telemetry**: Implement logging to an external monitoring service (like Azure Application Insights) to track LID success/failure rates and the distribution of detected languages. This will help identify issues with specific types of content.

### :black_square_button: Task 4: Expand Language Prompt Library

The prompt library in `action_summary.py` is now structured for expansion.

-   **How to Implement**:
    1.  **Add New Languages**: To support a new language (e.g., Spanish), simply add a new key-value pair to the `_LANGUAGE_INSTRUCTIONS` dictionary with the translated and culturally adapted instructions.
    2.  **Refine Prompts**: Continuously refine the prompts for each language based on the quality of the LLM output. Pay close attention to nuances and idiomatic expressions.
