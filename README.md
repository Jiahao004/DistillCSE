# DistillCSE: Distilled Contrastive Learning for Sentence Embeddings

This is the source code of DistilLCSE, which study the factors that affect the distillation learning for contrastive learning sentence embeddings. 

We identify the high variance in the teacher's output logit distribution that significantly affect the students performance.

We propose Group-P shuffling and multiple teacher components to regulate the high variance issue from teachers. Specifically,

1) Group-P Shuffling: shuffles the top-k logits from teachers. Here, k controls the reliance on teacher's information, larger k reduce while smaller k increase the reliance on teacher logits.
2) Teacher Components: mean pooling the logits from multiple teacher components, therefore, the variance is reduced.

Distilling from homogeneous SimCSE:

| **Method** |**STS12** | **STS13** | **STS14** | **STS15** | **STS16** | **STS-B** | **SICK-R** | **Avg.** | 
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GloVe embeddings (avg.) | 55.14 | 70.66 | 59.73 | 68.25 | 63.66 | 58.02 | 53.76 | 61.32 |
| BERTbase(first-last avg.) | 39.70 | 59.38 | 49.67 | 66.03 | 66.19 | 53.87 | 62.06 | 56.70 |
| BERTbase-flow | 58.40 | 67.1 | 60.85 | 75.16 | 71.22 | 68.66 | 64.47 | 66.55 |
| BERTbase-whitening | 57.83 | 66.9 | 60.9 | 75.08 | 71.31 | 68.24 | 63.73 | 66.28 |
| IS-BERTbase | 56.77 | 69.24 | 61.21 | 75.23 | 70.16 | 69.21 | 64.25 | 66.58 |
| CT-BERTbase | 61.63 | 76.8 | 68.47 | 77.50 | 76.48 | 74.31 | 69.19 | 72.05 |
| ConSERT-BERTbase | 64.64 | 78.49 | 69.07 | 79.72 | 75.95 | 73.97 | 67.31 | 72.74 |
| DiffCSE-BERTbase | 72.28 | 84.43 | 76.47 | 83.90 | 80.54 | 80.59 | 71.23 | 78.49 |
| SimCSE-BERTbase | 68.40 | 82.41 | 74.38 | 80.91 | 78.56 | 76.85 | 72.23 | 76.25 |
| DCLR-BERTbase | 70.81 | 83.73 | 75.11 | 82.56 | 78.44 | 78.31 | 71.59 | 77.22 |
| ArcCSE-BERTbase | 72.08 | 84.27 | 76.25 | 82.32 | 79.54 | 79.92 | 72.39 | 78.11 |
| Vanilla-Distill-BERTbase | 70.85 | 83.49 | 74.84 | 81.52 | 78.19 | 78.60 | 71.69 | 77.03 |
| * DistillCSE-BERTbase | **72.32** | **84.88** | **77.12** | **83.92** | **81.05** | **80.75** | **73.21** | **79.04** |
|     --*group-p shuffling (p=0.1) | 72.39 | 83.51 | 75.71 | 82.97 | 78.87 | 79.48 | 73.24 | 78.02 |
|     --*teacher components | 73.14 | 84.36 | 77.05 | 83.64 | 79.94 | 80.21 | 72.15 | 78.64 |
| SimCSE-BERTlarge | 70.88 | 84.16 | 76.43 | 84.5 | 79.76 | 79.26 | 73.88 | 78.41 |
| DCLR-BERTlarge | 71.87 | 84.83 | 77.37 | 84.7 | 79.81 | 79.55 | 74.19 | 78.90 |
| ArcCSE-BERTlarge | 73.17 | 86.19 | 77.9 | 84.97 | 79.43 | 80.45 | 73.50 | 79.37 |
| Vanilla-Distill-BERTlarge | 72.27 | 85.56 | 77.65 | 84.82 | 80.36 | 80.53 | 75.05 | 79.46 |
| * DisitllCSE-BERT-large | **75.18** | **86.32** | **78.92** | **85.89** | **81.18** | **81.97** | **75.33** | **80.68** |
| SimCSE-RoBERTabase | 70.16 | 81.77 | 73.24 | 81.36 | 80.65 | 80.22 | 68.56 | 76.57 |
| DCLR-RoBERTabase | 70.01 | 83.08 | 75.09 | 83.66 | 81.06 | 81.86 | 70.33 | 77.87 |
| Vanilla-Distill-RoBERTabase | 71.14 | 82.49 | 73.67 | 81.18 | 81.58 | 81.24 | 68.74 | 77.15 |
| * DistillCSE-RoBERTabase | **71.45** | **83.33** | **75.53** | **83.19** | **82.47** | **82.38** | **69.44** | **78.26** |
| SimCSE-RoBERTalarge | 72.86 | 83.99 | 75.62 | 84.77 | 81.8 | 81.98 | 71.26 | 78.90 |
| DCLR-RoBERTalarge | 73.09 | 84.57 | 76.13 | 85.15 | 81.99 | 82.35 | 71.8 | 79.30 |
| Vanilla-Distill-RoBERTalarge | **73.35** | 84.59 | 76.80 | 85.20 | 81.84 | 82.48 | 71.34 | 79.37 |
| * DistillCSE-RoBERTa-large | **74.86** | **85.72** | **78.15** | **86.42** | **83.35** | **84.96** | **73.20** | **80.95** |


## Environment

transformers==4.2.1

datasets==1.18.3

torch==1.10.2

## Run
The running examples are in the .sh bash file.

--n_gpu_for_training: number of gpus for training the model, while the rest of the gpus are used for teacher logits inference

--distill_weight: the weightage for distillation loss

--distill_teacher: the teachers for distillation logits, using whitespace to seperate the path of the teachers

--distil_temp1: the temperature for student in distillation

--disitll_temp2: the temperature for teacher in distillation

--shuffle_start, --shuffle_end: shuffle the descending sorted logits between indices shuffle_start and shuffle_end while distillation, set --shuffle_start 0 and --shuffle_end 6 for default top-k shuffling regulation

--distill_alpha: the weight for base model logits when using both base and large models as teachers for distillation, the large model weight is (1-distill_alpha)
