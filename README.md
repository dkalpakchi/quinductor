# Quinductor
A multilingual data-driven method for generating reading comprehension questions. The official repository for the Quinductor article: https://arxiv.org/abs/2103.10121

## Data
We use [TyDi QA dataset](https://github.com/google-research-datasets/tydiqa), which you can easily get by running `get_tydiqa_data.sh`

## How to induce templates?
1. Generate auxiliary models:
  - IDFs by running `calculate_idf.sh`
  - ranking models by running `get_qword_stat.sh`
2. Induce templates and guards by running `induce_templates.sh`
If you want to induce templates only for a specific language, please choose the correpsonding lines from the shell scripts.

## How to evaluate?
We use [nlg-eval package](https://github.com/Maluuba/nlg-eval) to calculate automatic evaluation metrics. 
This package requires to have hypothesis and ground truth files, where each line correspond to a question generated based on the same sentence.
To generate these files, please run `evaluate.sh` (if you want to induce templates only for a specific language, please choose the correpsonding lines from the shell scripts.).

Then automatic evaluation metrics can be calculated by running a command similar to the following (example is given for Arabic):

```nlg-eval --hypothesis templates/ar/1614104416496133/eval/hypothesis_ar.txt --references templates/ar/1614104416496133/eval/ground_truth_ar_0.txt --references templates/ar/1614104416496133/eval/ground_truth_ar_1.txt --references templates/ar/1614104416496133/eval/ground_truth_ar_2.txt --no-glove --no-skipthoughts```

## Cite us
```
@misc{kalpakchi2021quinductor,
      title={Quinductor: a multilingual data-driven method for generating reading-comprehension questions using Universal Dependencies}, 
      author={Dmytro Kalpakchi and Johan Boye},
      year={2021},
      eprint={2103.10121},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
