# Quinductor
A multilingual data-driven method for generating reading comprehension questions. The official repository for the Quinductor article: https://arxiv.org/abs/2103.10121

## Data
We use [TyDi QA dataset](https://github.com/google-research-datasets/tydiqa), which you can easily get by running `get_tydiqa_data.sh`

## How to work with the induced templates?
Quinductor is now available as a Python package that can be installed via `pip install quinductor`. After that you can download the induce templates for your language by running the following in the Python shell (the example is for English).
```python
>>> import quinductor as qi
>>> qi.download('en')
```
The avaible languages with a wide set of templates are:
- Arabic (`ar`)
- English (`en`)
- Finnish (`fi`)
- Indonesian (`id`)
- Japanese (`ja`)
- Russian (`ru`)

Templates are also available for the other languages listed below, but Quinductor did not manage to induce many templates on the TyDiQA.
- Korean (`ko`)
- Telugu (`te`)

After having downloaded the templates for your language, you can get access to them by running
```python
>>> tools = qi.use('en')
```

Starting from v0.2.0, you can also use the `tools` dictionary to quickly induce QA-pairs using the following piece of code.
```python
import quinductor as qi
import udon2

tools = qi.use("en")
trees = udon2.ConllReader.read_file("example.conll")
res = qi.generate_questions(trees, tools)
print("\n".join([str(x) for x in res]))
```
Each element in the `res` list above will be an instance of `GeneratedQAPair` class, which has the following properties:
* `q` -- generated question as a string
* `a` -- generated answer as a string
* `score` -- the Quinductor score for this QA-pair (the list is sorted in the descending order of the scores)
* `template` -- a list of templates that resulted in the induced QA-pair

## How to induce templates yourself?
1. Generate auxiliary models:
  - IDFs by running `calculate_idf.sh`
  - ranking models by running `get_qword_stat.sh`
2. Induce templates and guards by running `induce_templates.sh`
If you want to induce templates only for a specific language, please choose the correpsonding lines from the shell scripts.

## Using your own templates
Quinductor templates constitute a plain text file with a number of induced templates. However, in order for them to be used, Quinductor requires a number of extra files in addition to the templates file:
* guards file -- a plain text file with guards for all templates, i.e. conditions on the dependency trees that must be satisfied for applying each template
* examples file -- a file containing the sentences from the training corpus that gave rise to each template
* question word model -- a dill binary file containing the question word model (see the associated article for explanations), can be induced by using `qword_stat.py` script
* answer statistics file -- a dill binary file containng the statistics about pos-morph expressions for the root tokens of the answers in the training set, used for filtering (can be induced using `qword_stat.py` script also)
* pos-morph n-gram model folder -- a folder containing a number of plain text files with n-gram models of pos-morph expressions (see the associated article for more details and [ewt_dev_freq.txt](https://github.com/dkalpakchi/quinductor/blob/master/templates/en/pos_ngrams/ewt_dev_freq.txt) for the example of the file format)

Quinductor templates along with all aforementioned extra files constitute a Quinductor model. Each such model must be organized as a folder with the following structure:
```
|- language code
  |- pos_ngrams -- a folder with pos-morph n-gram model
  |- dataset name -- a name of the dataset used for inducing templates
    |- a unique name for templates -- a timestamp if templates induced by the script from this repo
      |- guards.txt -- guards file
      |- templates.txt -- templates file
      |- sentences.txt -- examples file
    |- atmpl.dill -- answer statistics file
    |- qwstats.dill -- question word model file
```

If you want to use a custom Quinductor model, you should organize your folder according to the structure above and give the path to the folder with `templates.txt` file as an extra argument called `templates_folder` to the `qi.use` method, as shown below.
```python
import quinductor as qi
tools = qi.use('sv', templates_folder='my_templates/sv/1613213402519069')
```
If you want only parts of a Quinductor model to differ from one of the default models, you can specify more fine-grained self-explanatory arguments to the `qi.use` method: `guards_files`, `templates_files`, `pos_ng_folder`, `example_files`, `qw_stat_file`, `a_stat_file`.

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
