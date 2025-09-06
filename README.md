# Mind the data gap: Missingness Still Shapes Large Language Model Prognoses
This repository allows reproducing the results presented in [Mind the data gap: Missingness Still Shapes Large Language Model Prognoses](). In this work, we investigate the impact of missingness serialization on the zero-shot performance of LLMs.

## Experimental setup
The proposed experiments consist of providing clinical data as inputs and prompting two LLMs (Qwen 3 and OSS-GPT) to predict an outcome of interest. To measure the impact of missingness, we employ two strategies to serialize the data: with and without missingness indicators in the serialized input.

To reproduce the paper's results:

1. **Generate MIMIC-IV MEDS build and task cohorts**  
   Use [MEDS-DEV](https://github.com/Medical-Event-Data-Standard/MEDS-DEV/tree/main) to construct the MIMIC-IV MEDS build and downstream task cohorts.  
   Follow the instructions in that repository to create the required inputs.

2. **Create the final evaluation cohort**

   This step extracts the clinical measurements and formats them into the final evaluation cohort used for inference.
   From this repository, run:
   ```bash
   python main.py --experiment mimic --mode generate_cohort

3. **Run inference**

   Generate LLM predictions by running:
   ```bash
   python main.py --experiment mimic --mode test

## Requirements

- **Python 3.11**
- **vLLM** for efficient inference.

To install with conda:

```bash
conda env create -f environment.yml
conda activate vllm_env

