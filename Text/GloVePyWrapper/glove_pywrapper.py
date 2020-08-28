# Copyright 2017 The Wenchen Li. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os

condition_f = lambda x,y: x + str(y) if y is not None else ""


class GloveWrapper(object):
    """python glove wrapper for the original c code"""

    def __init__(self, corpus,
                 name,
                 train_dir,
                 vocab_file="vocab.txt",
                 cooccurrence_file="cooccurrence.bin",
                 cooccurrence_shuffle_file="cooccurrence.shuf.bin",
                 builddir="build",
                 save_file="vectors",
                 verbose=2,
                 memory=4.0,
                 vocab_min_count=5,
                 vector_size=50,
                 max_iter=25,
                 window_size=15,
                 binary=2,
                 num_thread=8,
                 x_max=10,
                 max_vocab=100000,
                 symmetric=1,
                 max_product=None,
                 overflow_length=None,
                 overflow_file='tempoverflow',
                 array_size=None,
                 temp_file=None,
                 alpha=.75,
                 eta=.05,
                 model=2,
                 gradsq_file=None,
                 save_gradsq=0,
                 checkpoint_every=None):
        """
        input: below description covers only part of the input flow, for more detail,
            please refer to the different methods below __init__.

            corpus: path to the corpus file
            name:name of the current training for easily differentiating saved trained files
            builddir:where the src is built
            verbose: Set verbosity: 0, 1, or 2 (default, input to all commands)
        """
        self.TRAIN_DIR = train_dir
        if not os.path.exists(self.TRAIN_DIR):
            os.makedirs(self.TRAIN_DIR)
        self.NAME =name + "_"
        self.BUILDDIR =  builddir
        # os.getcwd() + "/model/gloVe/" +
        #vocab count
        self.CORPUS = corpus
        self.VERBOSE = verbose
        self.MAX_VOCAB = max_vocab
        self.VOCAB_MIN_COUNT = vocab_min_count
        self.VOCAB_FILE = self.TRAIN_DIR + self.NAME +vocab_file

        #cooccur
        self.SYMMETRIC = symmetric
        self.WINDOW_SIZE = window_size
        self.MEMORY = memory
        self.MAX_PRODUCT=max_product
        self.OVERFLOW_LENGTH = overflow_length
        self.OVERFLOW_FILE = overflow_file
        self.COOCCURRENCE_FILE = self.TRAIN_DIR + self.NAME + cooccurrence_file

        #shuffle
        self.COOCCURRENCE_SHUF_FILE = self.TRAIN_DIR + self.NAME + cooccurrence_shuffle_file
        self.ARRAY_SIZE = array_size
        self.TEMP_FILE = temp_file

        #glove
        self.SAVE_FILE = self.TRAIN_DIR + self.NAME + save_file
        self.VECTOR_SIZE = vector_size
        self.MAX_ITER = max_iter
        self.BINARY = binary
        self.NUM_THREADS = num_thread
        self.X_MAX = x_max
        self.ALPHA = alpha
        self.ETA = eta
        self.MODEL = model
        self.GRADSQ_FILE = gradsq_file
        self.SAVE_GRADSQ = save_gradsq
        self.CHECKPOINT_EVERY = checkpoint_every



    def vocab_count(self):
        """
        Simple tool to extract unigram counts

            verbose: Set verbosity: 0, 1, or 2 (default, input to all commands)
            vocab_file:File containing vocabulary (truncated unigram counts, produced by 'vocab_count');
                default vocab.txt
            max_vocab:Upper bound on vocabulary size, i.e. keep the <int> most frequent words.
                The minimum frequency words are randomly sampled so as to obtain an even
                distribution over the alphabet.(input to 'vocab_count')
            vocab_min_count:Lower limit such that words which occur fewer than vocab_min_count
                times are discarded.(input to 'vocab_count')
        """
        # vocabulary count
        print ("vocab count")
        vocab_count_command = self.BUILDDIR + "/" + "vocab_count " + \
                              "-max-vocab " +str(self.MAX_VOCAB) + \
                              "-min-count " + str(self.VOCAB_MIN_COUNT) + \
                              " -versbose " + str(self.VERBOSE) + \
                              " < " + self.CORPUS + " > " + self.VOCAB_FILE

        print (vocab_count_command)
        os.system(vocab_count_command)

    def cooccur(self):
        """
        Tool to calculate word-word cooccurrence statistics

            symmetric:If symmetric = 0, only use left context;
                if symmetric = 1 (default), use left and right
            window_size:Number of context words to the left (and to the right, if symmetric = 1);
                default 15(input to 'cooccur')
            memory:Soft limit for memory consumption, in GB -- based on simple heuristic,
                so not extremely accurate; default 4.0(input to 'cooccur','shuffle')
            max-product:Limit the size of dense cooccurrence array by specifying the max product
                <int> of the frequency counts of the two cooccurring words.This value
                overrides that which is automatically produced by '-memory'.
                Typically only needs adjustment for use with very large corpora.(input to 'cooccur')
            overflow-file:Filename, excluding extension, for temporary files; default overflow
                (input to 'cooccur')
            overflow-length:Limit to length <int> the sparse overflow array, which buffers cooccurrence
                data that does not fit in the dense array, before writing to disk.
                This value overrides that which is automatically produced by '-memory'.
                Typically only needs adjustment for use with very large corpora.(input to 'cooccur')
            cooccurrence_file:name of co occurrence file
        """
        # coocurrence matrix
        print ("coocurr matrix")
        coocurrence_command = self.BUILDDIR + "/" + "cooccur" + \
                              " -memory " + str(self.MEMORY) + \
                              " -vocab-file " + self.VOCAB_FILE + \
                              " -versbose " + str(self.VERBOSE) + \
                              " -window-size " + str(self.WINDOW_SIZE) + \
                              " -symmetric " + str(self.SYMMETRIC) + \
                              condition_f(" -overflow-length ",self.OVERFLOW_LENGTH) + \
                              condition_f(" -max-product ", self.MAX_PRODUCT) + \
                              " -overflow-file " + self.OVERFLOW_FILE + \
                              " < " + self.CORPUS + " > " + self.COOCCURRENCE_FILE

        print (coocurrence_command)
        os.system(coocurrence_command)

    def shuffle(self):
        """
        Tool to shuffle entries of word-word cooccurrence files

            memory:Soft limit for memory consumption, in GB -- based on simple heuristic,
                so not extremely accurate; default 4.0(input to 'cooccur','shuffle')
            array-size:Limit to length <int> the buffer which stores chunks of data to shuffle before
                writing to disk. This value overrides that which is automatically produced by '-memory'.
                (input to 'shuffle')
            temp-file:Filename, excluding extension, for temporary files; default temp_shuffle
                (input to 'shuffle')
            cooccurrence_shuffle_file:Binary input file of shuffled cooccurrence data
                (produced by 'cooccur' and 'shuffle'); default cooccurrence.shuf.bin
        """
        # coocurrence matrix reshuffle
        print ("coocurr matrix shuffle")
        shuffle_command = self.BUILDDIR + "/" + "shuffle " +\
                          " -memory " + str(self.MEMORY) + \
                          " -versbose " + str(self.VERBOSE) + \
                          condition_f(" -array-size ", self.ARRAY_SIZE) + \
                          condition_f(" -temp-file ", self.TEMP_FILE) + \
                          " < " + self.COOCCURRENCE_FILE + " > " + self.COOCCURRENCE_SHUF_FILE



        print (shuffle_command)
        os.system(shuffle_command)

    def glove(self):
        """
        Global Vectors for Word Representation

            vector_size:Dimension of word vector representations (excluding bias term); default 50
                (input to 'vocab_count')
            max_iter:Number of training iterations; default 25(input to 'glove')
            save_file:Filename, excluding extension, for word vector output; default vectors
            binary:Save output in binary format (0: text, 1: binary, 2: both); default 0
                (input to 'glove')
            num_thread:Number of threads; default 8(input to 'glove')
            x_max:Parameter specifying cutoff in weighting function; default 100.0(input to 'glove')

            gradsq-file:Filename, excluding extension, for squared gradient output; default gradsq
                (input to 'glove')
            save-gradsq:Save accumulated squared gradients; default 0 (off); ignored
                if gradsq-file is specified(input to 'glove')
            checkpoint-every:Checkpoint a  model every <int> iterations; default 0 (off)
                (input to 'glove')
            model:for word vector output (for text output only); default 2
                0: output all data, for both word and context word vectors, including bias terms\n");
                1: output word vectors, excluding bias terms\n");
                2: output word vectors + context word vectors, excluding bias terms
                (input to 'glove')
            alpha:Parameter in exponent of weighting function; default 0.75(input to 'glove')
            eta:Initial learning rate; default 0.05(input to 'glove')
        """
        # glove training
        print ("train glove")
        glove_command = self.BUILDDIR + "/" + "glove " + \
                        " -save-file " + self.SAVE_FILE + \
                        " -threads " + str(self.NUM_THREADS) + \
                        " -input-file " + self.COOCCURRENCE_SHUF_FILE + \
                        " -x-max " + str(self.X_MAX) + \
                        " -iter " + str(self.MAX_ITER) + \
                        " -vector-size " + str(self.VECTOR_SIZE) + \
                        " -binary " + str(self.BINARY) + \
                        " -vocab-file " + self.VOCAB_FILE + \
                        " -verbose " + str(self.VERBOSE) + \
                        " -alpha " + str(self.ALPHA) + \
                        " -eta " + str(self.ETA) + \
                        " -model " + str(self.MODEL) + \
                        condition_f(" -gradsq-file ",self.GRADSQ_FILE) + \
                        " -save-gradsq " + str(self.SAVE_GRADSQ) + \
                        condition_f(" -checkpoint-every ",self.CHECKPOINT_EVERY)

        print (glove_command)
        os.system(glove_command)

    def eval(self):
        """
        if [ "$CORPUS" = 'text8' ]; then
           echo "$ python eval/evaluate.py"
           python eval/evaluate.py
        fi
        """
        pass

if __name__ == '__main__':
    CORPUS = "text8"
    glove = GloveWrapper(CORPUS, "text8")
    ## prepare
    ###prepare vocabulary count
    glove.vocab_count()
    ###prepare co-occurrence matrix
    glove.cooccur()
    ###reshuffle
    glove.shuffle()
    ###glove train
    glove.glove()
