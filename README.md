# Developing Medical Assistant Bot 
# Opik: Fine-Tuned LLM Evaluation

## Model Evaluation
Opik utilizes various metrics to evaluate fine-tuned LLMs:

- **Levenshtein**: Calculates the difference between predictions and ground truth (GT) using the Levenshtein distance, which determines the minimum number of character changes required between two sentences.
  
- **Hallucination**: An LLM-as-a-judge checks if the answer aligns with the user's question, scoring from 0 to 1 (0 indicates no hallucinations).
  
- **Moderation**: Similar to the hallucination metric, this checks the safety of the response, with a score range of 0 to 1 (0 indicates a completely safe response).
  
- **Style**: A custom metric that assesses if the writing style is appropriate for a blog post. Scores range from 0.0 (inappropriate) to 1.0 (appropriate), with 0.5 indicating a middle ground.

Opik ensures the validity of input and output data, confirming that outputs are in JSON format and scores are between 0 and 1.

## Example Interaction
**Question**: "What are the treatments for Paget's Disease of Bone?"  
**Answer**: "The Food and Drug Administration has approved various medications that can help halt or slow the progression of Paget's Disease of Bone, as well as alleviate pain and other symptoms. These treatments primarily include two categories: bisphosphonates and calcitonin. Among these, doctors most commonly prescribe one of the four most effective bisphosphonates: risedronate, alendronate, pamidronate, and zoledronic acid."

**Question**: "What causes Urinary Tract Infections?"  
**Answer**: "Urinary tract infections (UTIs) are primarily caused by bacteria, especially E. coli, entering the urethra and bladder, with higher risks in individuals with spinal cord injuries, urinary blockages, diabetes, weakened immune systems, and women due to anatomical factors. Sexual activity, catheter use, and certain birth control methods can further increase susceptibility."

## Process
The development involved a single SFT step using LoRA for simplicity and cost-effectiveness, alongside Unsloth and TRL to define the fine-tuning script.

- **Unsloth**: A new tool for fine-tuning LLMs that makes training 2x faster and 60% more memory-efficient compared to direct HuggingFace usage. This leads to faster experiments, more iterations, and lower costs.

- **Comet ML**: Used as an experiment tracker to log training metrics across multiple experiments, allowing for comparison and selection of the best model for production.

## Features

### Feature 1: Model Loading
The `load_model()` function loads the pre-trained model and its tokenizer using Unsloth’s FastLanguageModel. The `load_in_4bit` parameter enables 4-bit quantization, significantly reducing the model’s memory footprint while maintaining performance.

### Feature 2: Cost-Effective Training
Training an 8B LLM was achieved on a single “ml.g5.2xlarge” instance with an NVIDIA A10G GPU (24 VRAM), costing approximately $2/hour. Without Unsloth, the training would require a “ml.g5.12xlarge” instance with 4 A10G GPUs, costing about $9/hour. 

This represents a 77.77% reduction in costs, not accounting for faster Unsloth experiments due to reduced IO overhead.

## Assumptions
- The model development assumes that the input data is clean and representative of the target distribution.
- It is assumed that the evaluation metrics accurately reflect model performance.

## Performance
### Strengths
- Cost-effective training and reduced memory usage.
- Fast experimentation cycles leading to quicker iterations and feedback.

### Weaknesses
- Limited by the initial assumptions regarding data quality.
- Potential for hallucinations and moderation issues, which require ongoing monitoring.

## Potential Improvements
- Refining the model architecture for better performance.
- Enhancing the evaluation metrics for more nuanced assessments.
- Exploring additional fine-tuning techniques or datasets to improve model robustness.
