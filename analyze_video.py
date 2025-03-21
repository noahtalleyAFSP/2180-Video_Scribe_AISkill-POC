from cobrapy import VideoClient
from cobrapy.analysis import ActionSummary
import logging
import os
import sys
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)

def analyze_video(video_path, person_group_id=None, peoples_list_path=None, emotions_list_path=None, fps=0.33, segment_length=10):
    """Analyze a video using Azure Speech and OpenAI Vision."""
    try:
        # Normalize paths
        video_path = os.path.abspath(video_path)
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        
        print(f"Analyzing video: {video_path}")
        print(f"Using env file: {env_path}")
        if person_group_id:
            print(f"Using face recognition with person group ID: {person_group_id}")
        if peoples_list_path:
            print(f"Using peoples list for additional identification: {peoples_list_path}")
        if emotions_list_path:
            print(f"Using emotions list for emotion detection: {emotions_list_path}")
        print(f"Using {fps} frames per second")
        
        # Initialize client
        client = VideoClient(
            video_path=video_path,
            env_file_path=env_path
        )
        
        # Preprocess
        print("Preprocessing video...")
        client.preprocess_video(
            segment_length=segment_length,  # segment length in seconds
            fps=fps,                       # frames per second
            generate_transcripts_flag=True,
            overwrite_output=True
        )
        
        # Analyze
        print("Analyzing video content...")
        results = client.analyze_video(
            analysis_config=ActionSummary(),  # Use ActionSummary for detailed analysis
            run_async=True,                  # Run analysis asynchronously for better performance
            max_concurrent_tasks=4,          # Adjust based on your CPU cores
            person_group_id=person_group_id, # Pass the custom face model ID
            peoples_list_path=peoples_list_path,  # Pass the peoples list for identification
            emotions_list_path=emotions_list_path  # Pass the emotions list for emotion detection
        )
        
        print(f"Analysis complete. Results written to: {client.manifest.processing_params.output_directory}")
        return results
        
    except Exception as e:
        error_message = str(e)
        print(f"Error: {error_message}")
        
        # Provide helpful advice for common errors
        if "QUALITY_FOR_RECOGNITION" in error_message:
            print("\n===== FACE QUALITY ERROR SOLUTION =====")
            print("The error indicates that faces in the video frames are not meeting Azure's quality standards for recognition.")
            print("\nPossible solutions:")
            print("1. We've modified the code to bypass the quality check - please try again.")
            print("2. Try using a different video with clearer, well-lit faces that face forward.")
            print("3. If the issue persists, try increasing the fps value to get clearer frames:")
            print("   python analyze_video.py \"" + video_path + "\" --face-model " + (person_group_id or "your_model_id") + " --fps 1.0")
            print("\nNote: Even with the quality error, basic face detection should still work, but identification may be limited.")
        
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a video with custom face recognition")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--face-model", help="Azure Face API person group ID for custom face recognition")
    parser.add_argument("--peoples-list", help="Path to a JSON file containing people to identify in the video")
    parser.add_argument("--emotions-list", help="Path to a JSON file containing emotions to detect in the video")
    parser.add_argument("--fps", type=float, default=0.33, 
                       help="Frames per second to extract (default: 0.33). Increase to 1.0 or higher for better face detection.")
    parser.add_argument("--segment-length", type=int, default=10, help="Length of video segments in seconds (default: 10)")
    parser.add_argument("--ignore-face-quality", action="store_true", help="Attempt face identification regardless of quality issues")
    args = parser.parse_args()
    
    analyze_video(
        args.video_path, 
        person_group_id=args.face_model, 
        peoples_list_path=args.peoples_list,
        emotions_list_path=args.emotions_list,
        fps=args.fps, 
        segment_length=args.segment_length
    ) 