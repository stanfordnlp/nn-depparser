# nn-depparser

A re-implementation of `nndep` using PyTorch.

Currently used for training CoreNLP dependency parsing models.

Requires Stanza for some features (auto-tagging with CoreNLP via server).

Originally by Danqi Chen. Leave a GitHub Issue if you have any questions!

## Example Usage

Train a model: 

```bash
python train.py -l universal -d /path/to/data --train_file it-train.conllu --dev_file it-dev.conllu --embedding_file /path/to/it-embeddings.txt --embedding_size 100 --random_seed 21 --learning_rate .005 --l2_reg .01 --epsilon .001 --optimizer adamw --save_path /path/to/experiment-dir --job_id experiment-name --corenlp_tags --corenlp_tag_lang italian --n_epoches 2000
```

Convert to CoreNLP format:

```bash
python gen_model.py -o /path/to/italian-corenlp-parser.txt /path/to/experiment-dir/experiment-name
```
