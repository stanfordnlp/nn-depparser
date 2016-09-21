DIR=/u/nlp/data/dependency_treebanks/corenlp-3.7.0
UD_TREEBANK_DIR=/u/nlp/data/dependency_treebanks/UD/1.3/

english-wsj:
	for split in train dev test; do
		echo generate $split..
		java -mx1g edu.stanford.nlp.trees.UniversalEnglishGrammaticalStructure −basic −keepPunct −conllx −treeFile  > /u/nlp/data/dependency_treebanks/PTB/tree_files/$split.mrg > $DIR/$@/$split.gold.conll
	done

english:
	for split in train dev test; do
		echo generate $split..
		java -mx1g edu.stanford.nlp.trees.UniversalEnglishGrammaticalStructure −basic −keepPunct −conllx −treeFile  > /u/nlp/data/dependency_treebanks/PTB/tree_files/$split.mrg > $DIR/$@/$split.gold.conll
	done
	java -mx1g edu.stanford.nlp.trees.UniversalEnglishGrammaticalStructure −basic −keepPunct −conllx −treeFile  > /u/nlp/data/dependency_treebanks/extraTrain/extraTrain.mrg >> $DIR/$@/train.gold.conll

chinese:

french:
	for split in train dev test; do
		echo generate $split..
		ln -s $UD_TREEBANK_DIR/UD_French/fr-ud-$split.conllu $DIR/$@/$split.gold.conll
	done

german:
	for split in train dev test; do
		echo generate $split..
		ln -s $UD_TREEBANK_DIR/UD_German/de-ud-$split.conllu $DIR/$@/$split.gold.conll
	done

spanish:
	for split in train dev test; do
		echo generate $split..
		ln -s $UD_TREEBANK_DIR/UD_German/es-ud-$split.conllu $DIR/$@/$split.gold.conll
	done
