# LeKUBE: A Legal Knowledge Update Benchmark

Dataset and evaluation code of LeKUBE.

## Data Release

The `./data` folder contains the LeKUBE dataset, with the following files:

- **`LawAdd.json`**: Contains the updated laws in LeKUBE. The data format is a dictionary where the key is the law name and the value is a list with the following elements:
  ```
  [
      Content of the law before the update,
      Content of the law after the update (with changes marked by **...**),
      Method of law modification,
      Reason for modification (empty string for the Release version),
      Crime (empty string if the law belongs to the Civil Code)
  ]
  ```

- **`LawQA.json`**: Contains Multiple-Choice Questions of the Legal Scenario corresponding to each updated law in LeKUBE. In each value, the `"Answer_before"` field is the answer to the question before the law was updated, and the `"Answer_after"` field is the answer after the law was updated.

- **`LawDiscri.json`**: Contains True-or-False Questions of Change in Statute for each updated law in LeKUBE. Each value is a list of dictionaries containing the questions and answers.

- **`LawDiscriNOT.json`**: Contains True-or-False Questions of Change in Statute for 60 non-updated laws, to test the locality of knowledge update methods. The format is the same as `LawDiscri.json`.

- **`LawChapter_sorted.json`**: Contains the updated laws sorted by chapter and article number.

- **`old_law_all.json`**: Contains all national laws and judicial interpretations in China. We used part of the data from the paper "STARD: A Chinese Statute Retrieval Dataset with Real Queries Issued by Non-professionals" ([https://arxiv.org/abs/2406.15313](https://arxiv.org/abs/2406.15313)). The GitHub link is [STARD](https://github.com/oneal2000/STARD/tree/main). This repository provides a sample file. To run the evaluation process smoothly, you should download [STARD-corpus.jsonl](https://github.com/oneal2000/STARD/blob/main/data/corpus.jsonl), convert it to the format of `./data/old_law_all.json` in this repository, and replace it.

## Tasks for Accuracy, Generality, and Locality

Using `Baichuan2Chat13B` as an example, we introduce how each knowledge update method generates answers in the paper. If files cannot run smoothly due to relative paths, you can run the code in the corresponding folder or change the relative paths in the code to absolute paths.

### Preparation

#### Lawformer Index Generation

Run
```
python ./task_generation/search/lawformer_index.py
```

#### Model Fine-Tuning

Run `./train/ft_update/train_baichuanchat.sh` or `./train/lora_update/train_baichuanchat.sh` to start training and obtain a checkpoint. It is recommended to merge the results obtained by Lora into the original weights to match the subsequent code in this repository.

#### Self-Edit

Run
```
python ./task_generation/selfedit_gen_qa.py baichuanchat
```
to generate the corresponding Self-Edit enhanced data in the `./task_generation/selfeditqa` folder, then run `./train/selfedit_update/train_baichuanchat.sh` to start training and obtain a checkpoint.

### Answer Generation

#### KN & ROME

We use the `EasyEdit` library. First, clone [EasyEdit](https://github.com/zjunlp/EasyEdit) to your local machine at `/path/to/easyedit` and install the dependencies.

#### Other Methods

In `./task_generation`, `law_*.py` are task scripts for other knowledge update methods besides Model Editing. The code and tasks correspond as follows:

| File Name             | Corresponding Task                                |
|-----------------------|---------------------------------------------------|
| law_recite.py         | Recitation of Statutes                            |
| law_recall.py         | Recall of Statute Items                           |
| law_discri.py         | True-or-False Questions of Change in Statute      |
| law_scenario_mcq.py   | Multiple-Choice Questions of the Legal Scenario   |
| law_not*.py           | Task of testing Locality                          |

Each file can accept 3-5 external parameters with the following meanings:

**Parameter 1**: Model name, choose from "baichuanchat, chatglm, chatlaw, legalaid".

**Parameter 2**: Prompt type, a positive integer from 0-3 where 0 represents no extra prefix (i.e., prompts for Raw, FT, Lora, and Self-Edit), 1 represents ICL-Golden, 2 represents RAG-BM25, and 3 represents RAG-Lawformer.

**Parameter 3**: Environment variable `CUDA_VISIBLE_DEVICES`.

**Parameter 4** (optional): Custom output file name `result_{tag}.json`.

**Parameter 5** (optional): Path to the model checkpoint. Required for FT, Lora, and Self-Edit.

For example, to update `baichuanchat` using Self-Edit and evaluate the Locality task of Recall of Statute Items, a possible command is:
```
python ./task_generation/law_notrecall.py baichuanchat 0 0 selfedit /path/to/selfedit-baichuanchat-ckpt/
```

## Tasks for Scalability & Retainability

We provide the `LawChapter_sorted.json` file with laws in chapter order. When testing the performance of Scalability or Retainability, you can split `LawAdd.json` and `LawQA.json` based on this file. The training and evaluation process is the same as for the Generality task of Multiple-Choice Questions of the Legal Scenario.

## Result Evaluation

Run
```
python ./task_eval/law_*.py
```
to generate evaluation results in xlsx format in the `"../task_generation/result/*` folder.