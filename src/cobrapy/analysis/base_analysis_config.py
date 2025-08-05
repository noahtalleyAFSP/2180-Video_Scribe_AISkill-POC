import json
from pydantic import BaseModel, Field
from typing import List, Union, Optional


class AnalysisConfig(BaseModel):
    name: str = Field("BaseAnalysis", description="Name of the analysis")
    analysis_sequence: str = Field(enum=["mapreduce", "refine"])
    system_prompt: str = None
    lens_prompt: str = None
    results_template: Union[dict, List[dict], str] = Field(default_factory=list)
    outputs_prompt: str = """"""
    PERSON_LINKING_CONFIDENCE_THRESHOLD: Optional[float] = Field(
        0.7, 
        description="Confidence threshold for linking LLM person names to refined track IDs during segment analysis."
    )

    def generate_system_prompt_template(self, **kwargs):
        results_template_str = json.dumps(self.results_template)
        results_template_str = results_template_str.replace("{", "{{").replace(
            "}", "}}"
        )
        prompt_template = (
            f"""{self.system_prompt}\n\n"""
            f"""## Additional Instruction\n"""
            f"""{self.lens_prompt}\n\n"""
            f"""## Output Instructions\n"""
            f"""{self.outputs_prompt.format(results_template=results_template_str, **kwargs)}"""
        )
        return prompt_template


class ParallelAnalysisConfig(AnalysisConfig):
    name: str = "ParallelAnalysis"
    analysis_sequence: str = "mapreduce"
    run_final_summary: Optional[bool] = Field(
        False,
        description="If True, at the end of the analysis a final summarization is run based on the generated results.",
    )
    final_summary_prompt: Optional[str] = Field(
        None,
        description="Prompt that will be used when summarizing the generated results from an analysis.",
    )


class SequentialAnalysisConfig(AnalysisConfig):
    name: str = "SequentialAnalysis"
    analysis_sequence: str = "refine"
    refine_prompt: Optional[str] = Field(
        (
            "This is the output from the last {number_of_previous_results_to_refine} chapters: {current_n_results}"
        ),
        description="Prompt that will be used as an additional prompt for when refining the generated results.",
    )

    number_of_previous_results_to_refine: int = 3

    def generate_system_prompt_template(self, is_refine_step=False, **kwargs):
        results_template_str = json.dumps(self.results_template)
        results_template_str = results_template_str.replace("{", "{{").replace(
            "}", "}}"
        )

        if is_refine_step:
            system_prompt = self.system_prompt + "/n/n" + self.refine_prompt
        else:
            system_prompt = self.system_prompt
        prompt_template = (
            f"""{system_prompt}\n\n"""
            f"""## Additional Instruction\n"""
            f"""{self.lens_prompt}\n\n"""
            f"""## Output Instructions\n"""
            f"""{self.outputs_prompt.format(results_template=results_template_str, **kwargs)}"""
        )
        return prompt_template
