## GloVe PyWrapper : python wrapper for Global Vectors for Word Representation

This module is a thin python wrapper module built upon original code from [GloVe](http://nlp.stanford.edu/projects/glove/).

you can easily make this a submodule to your project by altering built path in `__init__`
### Usage
```bash
./get_text8.sh
make
```
```python 
import glove_pywrapper

CORPUS = "text8"
glove = glove_pywrapper.GloveWrapper(CORPUS, "text8")
#prepare vocabulary count
glove.vocab_count()
#prepare co-occurrence matrix
glove.cooccur()
#reshuffle
glove.shuffle()
#glove train
glove.glove()

```


### License
All work contained in this package is licensed under the Apache License, Version 2.0. See the include LICENSE file.
