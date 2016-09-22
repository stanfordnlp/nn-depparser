
## Dependency parsing models for CoreNLP 3.7.0

## Treebanks

* Directory: `/u/nlp/data/dependency_treebanks/corenlp-3.7.0/`.
* See `gen_data.sh` for details.
* `{train|dev|test}.gold.conll`: gold part-of-speech tags.
* `{train|dev|test}.conll`: automatic part-of-speech tags by CoreNLP.

Treebank  | #Train    |  #Dev | #Test
----------| --------- | ---------- | -----
english-wsj  | 39,832  |  1,700 | 2,416
english    |  58,619 | 1,700 | 2,416
chinese   | 126,424 | 2,079 | 2,796
german | 14,118 | 799 | 977
french | 14,554 | 1,596 | 298
spanish | 14,187 | 1,552 | 274


#### English (wsj only)
* PTB3 corpus: `/u/nlp/data/PTB3/`
* Splits: train (sections 2-21), dev (section 22), test (section 23)
* Processed files: `/u/nlp/data/dependency_treebanks/PTB/tree_files/*.mrg`
* UD conversion:
```
    java -mx1g edu.stanford.nlp.trees.UniversalEnglishGrammaticalStructure
    −basic −keepPunct −conllx −treeFile <treebank>
```

#### English
* Training data: the training portion of `wsj`, plus `extraTrain` data: `/u/nlp/data/dependency_treebanks/extraTrain/extraTrain.mrg`
* Using the same UD conversion as above.


#### Chinese
* CTB9 corpus: `/scr/nlp/data/ldc/LDC2016T13-ctb9.0`
* Splits: `docs/ctb9.0-file-list.txt`, \*: test files; !: dev files and all the remaining files will be used as training data.
* Processed files: `/u/nlp/data/dependency_treebanks/CTB9/tree_files/*.mrg`
* UD conversion:
```
    java -mx1g edu.stanford.nlp.trees.international.pennchinese.UniversalChineseGrammaticalStructure
    -basic -keepPunct -conllx -treeFile <treebank>
```


#### German
* `UD v1.3`

#### French
* `UD v1.3`

#### Spanish
* `UD v1.3`


## Performance
