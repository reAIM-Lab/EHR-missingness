# Mind the data gap: Missingness Still Shapes Large Language Model Prognoses
This repository allows reproducing the results presented in [Mind the data gap: Missingness Still Shapes Large Language Model Prognoses](). In this work, we investigate the impact of missingness serialization on the zero-shot performance of LLMs.

## Experimental setup
The proposed experiments consist of providing clinical data as inputs and prompting two LLMs (Qwen 3 and OSS-GPT) to predict an outcome of interest. To measure the impact of missingness, we employ two strategies to serialize the data: with and without missingness indicators in the serialized input.

To reproduce the paper's results:


Note that you will need to download the `MIMIC-IV` dataset from [](). 


# Requirements
The model relies on `` libraries.
