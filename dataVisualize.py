import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
# Use the absolute path provided by the user for the *directory* containing the JSONs
BASE_OUTPUT_DIR = r"C:\Users\Admin\Desktop\support partners\Q2\Video Scribe V1\AIR-AI-Models-Cobra-0.1.0\dataAnalysisOutput"

# Define the JSON files and their corresponding chunk sizes
# Ensure these filenames match exactly what you have in BASE_OUTPUT_DIR
ANALYSIS_RESULTS = {
    1: os.path.join(BASE_OUTPUT_DIR, "1SecondActionSummary.json"),
    5: os.path.join(BASE_OUTPUT_DIR, "5SecondActionSummary.json"),
    10: os.path.join(BASE_OUTPUT_DIR, "10SecondActionSummary.json"),
    15: os.path.join(BASE_OUTPUT_DIR, "15SecondActionSummary.json")
}

# Define path for saving comparison outputs
COMPARISON_OUTPUT_DIR = BASE_OUTPUT_DIR # Save comparison in the same base directory

# --- End Configuration ---

# --- Helper Functions (Keep these as they are) ---

def extract_tag_counts(json_path: str) -> dict:
    """Loads ActionSummary JSON and counts person, action, object tags."""
    counts = {"persons": 0, "actions": 0, "objects": 0}
    print(f"Attempting to load: {json_path}")
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return None # Return None if file not found

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        action_summary = data.get("actionSummary", {})
        # Ensure keys exist before trying to get length
        counts["persons"] = len(action_summary.get("person", [])) if action_summary.get("person") is not None else 0
        counts["actions"] = len(action_summary.get("action", [])) if action_summary.get("action") is not None else 0
        counts["objects"] = len(action_summary.get("object", [])) if action_summary.get("object") is not None else 0
        print(f"Extracted counts from {os.path.basename(json_path)}: {counts}")
    except FileNotFoundError: # Should be caught by os.path.exists, but keep for safety
        print(f"Error: JSON file not found at {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return None
    except Exception as e:
        print(f"Error extracting counts from {json_path}: {e}")
        return None
    return counts

def create_comparison_table(results: dict) -> pd.DataFrame:
    """Creates a pandas DataFrame from the collected results."""
    data = []
    # Ensure results is not empty and contains valid data
    valid_results = {k: v for k, v in results.items() if v is not None and isinstance(v, dict)}
    if not valid_results:
        print("No valid results to create comparison table.")
        return pd.DataFrame() # Return empty DataFrame

    for chunk_size, counts in valid_results.items():
        data.append({
            "Frames per Tag Chunk": chunk_size,
            "Persons Found": counts.get("persons", 0), # Use .get for safety
            "Actions Found": counts.get("actions", 0),
            "Objects Found": counts.get("objects", 0),
        })
    if not data:
        print("No data rows to add to the DataFrame.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    # Only set index if 'Frames per Tag Chunk' column exists and is suitable
    if "Frames per Tag Chunk" in df.columns:
        try:
            df = df.set_index("Frames per Tag Chunk")
        except KeyError:
             print("Warning: Could not set 'Frames per Tag Chunk' as index.")
    return df

def plot_comparison_graph(df: pd.DataFrame, output_dir: str):
    """Creates and saves a bar chart comparing tag counts."""
    if df.empty:
        print("DataFrame is empty, cannot generate plot.")
        return

    try:
        ax = df.plot(kind='bar', figsize=(12, 7), rot=0)

        plt.title('Comparison of Tags Found vs. Frames per Tag Chunk')
        plt.xlabel('Max Frames Processed per Tagging Chunk')
        plt.ylabel('Number of Tags Found')
        plt.legend(title='Tag Type')
        plt.tight_layout()

        # Add counts above bars
        for container in ax.containers:
            ax.bar_label(container)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "tag_comparison_by_frames_per_chunk.png")
        plt.savefig(plot_path)
        print(f"Comparison graph saved to: {plot_path}")
        # plt.show() # Uncomment to display the plot directly
    except Exception as e:
        print(f"Error generating or saving plot: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    overall_start_time = time.time()
    script_start_time = time.time() # Use a separate timer if get_elapsed_time is removed

    print(f"--- Starting Data Comparison ---")
    print(f"Loading results from: {BASE_OUTPUT_DIR}")

    results_by_chunk_size = {}
    failed_loads = []

    # Load results from predefined JSON files
    for chunk_size, json_path in ANALYSIS_RESULTS.items():
        print(f"\nProcessing chunk size: {chunk_size} from {os.path.basename(json_path)}")
        counts = extract_tag_counts(json_path)
        if counts:
            results_by_chunk_size[chunk_size] = counts
        else:
            failed_loads.append(chunk_size)
            # Record failure if loading/extraction function returned None
            results_by_chunk_size[chunk_size] = None # Mark as failed/None

    # Analyze and Visualize Results
    # Filter out failed/None results before creating table/plot
    valid_results = {k: v for k, v in results_by_chunk_size.items() if v is not None}

    if valid_results:
        print("\n--- Comparison Results (Successful Loads) ---")
        comparison_df = create_comparison_table(valid_results)

        # Check if DataFrame is empty before proceeding
        if not comparison_df.empty:
             # Check if DataFrame has data before printing/plotting
             if comparison_df.shape[0] > 0:
                  print(comparison_df.to_markdown()) # Print table in markdown format
                  plot_comparison_graph(comparison_df, COMPARISON_OUTPUT_DIR)
             else:
                  print("Comparison DataFrame was created but is empty. Skipping table output and plot generation.")
        else:
             print("Comparison DataFrame could not be created or is empty. Skipping table output and plot generation.")

    else:
        print("\nNo analysis results loaded successfully. Cannot generate comparison.")

    # Print summary of failed loads
    if failed_loads:
        print(f"\nWarning: Failed to load or process results for the following chunk sizes: {failed_loads}")

    # Calculate elapsed time
    elapsed_time = time.time() - script_start_time
    print(f"\n--- Script finished in {elapsed_time:.2f}s ---")
