#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Sequence model
"""
from tqdm import tqdm
import os
import tensorflow as tf
from keras.optimizers import Adam
from collections import OrderedDict, defaultdict
from copy import deepcopy
from keras.models import Model
from kipoi_utils.data_utils import numpy_collate_concat
from bpnet.data import nested_numpy_minibatch
from bpnet.utils import flatten, fnmatch_any, _listify
from bpnet.functions import mean
import gin
from gin import config
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# import tensorflow as tf
# from tensorflow import keras
import keras
from keras import layers
from bpnet.multi_head_attention import MultiHeadAttention
from bpnet.layer_normalization import LayerNormalization

from keras.preprocessing import sequence
from keras.datasets import imdb
from matplotlib import pyplot as plt
import pandas as pd

from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Embedding,Dense

class Self_Attention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        #inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(4,input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
        print(WQ.shape,WK.shape, WV.shape)
        K.permute_dimensions(WK, [0, 2, 1])
        K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))
#         print("WQ.shape",WQ.shape)

#         print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)


        QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))

        QK = QK / (self.output_dim**0.5)

        QK = K.softmax(QK)

        print("QK.shape",QK.shape)

        V = K.batch_dot(QK,WV)

        return V

    def compute_output_shape(self, input_shape):

        return (input_shape[0],input_shape[1],self.output_dim)

from keras.layers.core import Flatten
    
class SeqModel:
    """Model interpreting the genome sequence
    """

    def __init__(self,
                 body,
                 # will be used for each task
                 heads,
                 tasks,
                 optimizer=Adam(lr=0.004),
                 seqlen=None,
                 input_shape=None,
                 input_name='seq'
                 ):
        """

        Args:
          seqlen: Explicit input sequence length
        Note: heads or body should work like parametrized functions building
        up a functional Keras model

        1. build the keras model (don't )
        2. compile the Keras model
        """
        import keras.layers as kl
        self.body = body
        self.tasks = tasks
        self.heads = heads
        self.seqlen = seqlen
        embed_dim = 32  # Embedding size for each token
        num_heads = 2  # Number of attention heads
        ff_dim = 32  # Hidden layer size in feed forward network inside transformer
        vocab_size = 20000  # Only consider the top 20k words
        maxlen = 200  # Only consider the first 200 words of each movie review
        if input_shape is None:
            input_shape = (self.seqlen, 4)
        inp = kl.Input(shape=input_shape, name=input_name)
        print('*******************************************')
        print('*******************************************')
        print('*******************************************')
        print('transformer working',input_shape)
        print('*******************************************')
        print('*******************************************')
        print('*******************************************')
#         embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
#         x = embedding_layer(inp)
#         transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
#         inp = transformer_block(inp)
        max_features = 1000
        _inp = Flatten()(inp)
        _inp = Dense(200)(_inp)
        embeddings = Embedding(max_features, 4)(_inp)
        print(embeddings.shape)

        _inp = Self_Attention(4)(embeddings)
#         inp = Flatten()(inp)
#         inp = Dense(4)(inp)
        print(_inp.shape, 'inp')
        bottleneck = body(_inp)
        self.bottleneck_name = bottleneck.name  # remember the bottleneck tensor name

        # create different heads
        outputs = []
        self.all_heads = defaultdict(list)
        self.losses = []
        self.loss_weights = []
        self.target_names = []
        self.postproc_fns = []
        bias_inputs = []
        for task in tasks:
            for head in heads:
                head = head.copy()
                self.all_heads[task].append(head)
                out, bias_input = head(bottleneck, task)
                
                outputs.append(out)
                bias_inputs += bias_input
                self.target_names.append(head.get_target(task))
                self.postproc_fns.append(head.postproc_fn)
                self.losses.append(head.loss)
                self.loss_weights.append(head.loss_weight)

        # create and compile the model
        print(bias_inputs[0].shape, inp.shape)
        self.model = Model([inp] + bias_inputs, outputs)
        self.model.compile(optimizer=optimizer,
                           loss=self.losses, loss_weights=self.loss_weights)

        # start without any contribution function
        self.contrib_fns = {}

    def _get_input_tensor(self):
        return self.model.inputs[0]

    def get_bottleneck_tensor(self, graph=None):
        if graph is None:
            graph = tf.get_default_graph()
        return graph.get_tensor_by_name(self.bottleneck_name)

    def bottleneck_model(self):
        return Model(self._get_input_tensor(),
                     self.get_bottleneck_tensor())

    def preact_model(self):
        # TODO - be able to serialize head.get_preact_tensor into json
        outputs = [head.get_preact_tensor()
                   for task, heads in self.all_heads.items()
                   for head in heads]
        return Model(self._get_input_tensor(), outputs)

    def predict_preact(self, seq, batch_size=256):
        m = self.preact_model()
        preds = m.predict(seq, batch_size=batch_size)
        return {k: v for k, v in zip(self.target_names, preds)}

    def neutral_bias_inputs(self, length, seqlen):
        """Compile a set of neutral bias inputs
        """
        return dict([head.neutral_bias_input(task, length, seqlen)
                     for task, heads in self.all_heads.items()
                     for head in heads if head.use_bias])

    def get_intp_tensors(self, preact_only=True):
        intp_targets = []
        for task, heads in self.all_heads.items():
            for head in heads:
                # TODO - be able to serialize head.intp_tensors into json
                for k, v in head.intp_tensors(preact_only=preact_only).items():
                    intp_targets.append((head.get_target(task) + "/" + k, v))
        return intp_targets

    def _contrib_deeplift_fn(self, x, name, preact_only=True):
        """Deeplift contribution score tensors
        """
        k = f"deeplift/{name}"
        if k in self.contrib_fns:
            return self.contrib_fns[k]

        import deepexplain
        from deepexplain.tensorflow.methods import DeepLIFTRescale
        from deepexplain.tensorflow import DeepExplain
        from bpnet.external.deeplift.dinuc_shuffle import dinuc_shuffle
        from keras.models import load_model, Model
        import keras.backend as K
        import numpy as np
        import tempfile

        self.contrib_fns = {}
        with tempfile.NamedTemporaryFile(suffix='.pkl') as temp:
            self.model.save(temp.name)
            K.clear_session()
            self.model = load_model(temp.name)

        # get the interpretation tensors
        intp_names, intp_tensors = list(zip(*self.get_intp_tensors(preact_only)))
        if name not in intp_names:
            raise ValueError(f"name {name} not in intp_names: {intp_names}")
        # input_tensor = self._get_input_tensor()
        input_tensor = self.model.inputs

        if isinstance(x, list):
            x_subset = [ix[:1] for ix in x]
        elif isinstance(x, dict):
            x_subset = [v[:1] for k, v in x.items()]
        else:
            x_subset = x[:1]

        with deepexplain.tensorflow.DeepExplain(session=K.get_session()) as de:
            fModel = Model(inputs=input_tensor, outputs=intp_tensors)
            target_tensors = fModel(input_tensor)
            for name, target_tensor in zip(intp_names, target_tensors):
                # input_tensor = fModel.inputs[0]
                self.contrib_fns["deeplift/" + name] = de.explain('deeplift',
                                                                  target_tensor,
                                                                  # NOTE: deepexplain will always take
                                                                  # the first element by definition
                                                                  input_tensor,
                                                                  x_subset)

        return self.contrib_fns[k]

    def _contrib_grad_fn(self, x, name, preact_only=True):
        """Gradient contribution score tensors
        """
        k = f"grad/{name}"
        if k in self.contrib_fns:
            return self.contrib_fns[k]
        from keras.models import load_model, Model
        import keras.backend as K
        import numpy as np
        import tempfile

        self.contrib_fns = {}

        # get the interpretation tensors
        intp_names, intp_tensors = list(zip(*self.get_intp_tensors(preact_only)))
        if name not in intp_names:
            raise ValueError(f"name {name} not in intp_names: {intp_names}")

        input_tensor = self.model.inputs

        if isinstance(x, list):
            x_subset = [ix[:1] for ix in x]
        elif isinstance(x, dict):
            x_subset = [v[:1] for k, v in x.items()]
        else:
            x_subset = x[:1]

        fModel = Model(inputs=input_tensor, outputs=intp_tensors)
        target_tensors = fModel(input_tensor)
        for name, target_tensor in zip(intp_names, target_tensors):
            # input_tensor = fModel.inputs[0]
            self.contrib_fns["grad/" + name] = K.function(input_tensor,
                                                          K.gradients(target_tensor, input_tensor[0]))

        return self.contrib_fns[k]

    def contrib_score(self, x, name, method='grad', batch_size=512, preact_only=False):
        """Compute the contribution score

        Args:
          x: one-hot encoded DNA sequence
          name: which interepretation method to compute
          method: which contribution score to use. Available: grad or deeplift
        """
        # Do we need bias?
        if not isinstance(x, dict) and not isinstance(x, list):
            seqlen = x.shape[1]
            x = {'seq': x, **self.neutral_bias_inputs(len(x), seqlen=seqlen)}

        if method == "deeplift":
            fn = self._contrib_deeplift_fn(x, name, preact_only=preact_only)
        elif method == "grad":
            fn = self._contrib_grad_fn(x, name, preact_only=preact_only)
        else:
            raise ValueError("Please provide a valid contribution scoring method: grad, deeplift")

        def input_to_list(input_names, x):
            if isinstance(x, list):
                return x
            elif isinstance(x, dict):
                return [x[k] for k in input_names]
            else:
                return [x]
        input_names = self.model.input_names
        assert input_names[0] == "seq"

        if batch_size is None:
            return fn(input_to_list(input_names, x))[0]
        else:
            return numpy_collate_concat([fn(input_to_list(input_names, batch))[0]
                                         for batch in nested_numpy_minibatch(x, batch_size=batch_size)])

    def contrib_score_all(self, seq, method='grad', batch_size=512, preact_only=True,
                          intp_pattern='*'):
        """Compute all contribution scores

        Args:
          seq: one-hot encoded DNA sequences
          method: 'grad' or deeplift'
          aggregate_strands: if True, the average contribution scores across strands will be returned
          batch_size: batch size when computing the contribution scores

        Returns:
          dictionary with keys: {task}/{head}/{interpretation_tensor}
          and values with the same shape as `seq` corresponding to contribution scores
        """
        intp_patterns = _listify(intp_pattern)  # make sure it's a list

        return {name: self.contrib_score(seq, name, method=method, batch_size=batch_size, preact_only=preact_only)
                for name, _ in self.get_intp_tensors(preact_only=preact_only)
                if fnmatch_any(name, intp_patterns)}
        # TODO - add a filter for name

    def predict(self, seq, batch_size=256):
        """Convert to dictionary
        """
        # Do we need bias?
        if not isinstance(seq, dict) and not isinstance(seq, list) and len(self.model.inputs) > 1:
            seq = {'seq': seq, **self.neutral_bias_inputs(len(seq), seqlen=seq.shape[1])}

        if batch_size is None:
            preds = self.model.predict_on_batch(seq)
        else:
            preds = self.model.predict(seq, batch_size=batch_size)
        return {k: postproc_fn(v) if postproc_fn is not None else v
                for k, v, postproc_fn in zip(self.target_names, preds, self.postproc_fns)}

    def evaluate(self, dataset,
                 eval_metric=None,
                 num_workers=8,
                 batch_size=256):
        lpreds = []
        llabels = []
        for inputs, targets in tqdm(dataset.batch_train_iter(cycle=False,
                                                             num_workers=num_workers,
                                                             batch_size=batch_size),
                                    total=len(dataset) // batch_size
                                    ):
            assert isinstance(targets, dict)
            target_keys = list(targets)
            llabels.append(deepcopy(targets))
            bpreds = {k: v for k, v in self.predict(inputs, batch_size=None).items()
                      if k in target_keys}  # keep only the target key predictions
            lpreds.append(bpreds)
            del inputs
            del targets
        preds = numpy_collate_concat(lpreds)
        labels = numpy_collate_concat(llabels)
        del lpreds
        del llabels

        if eval_metric is not None:
            return eval_metric(labels, preds)
        else:
            task_avg_tape = defaultdict(list)
            out = {}
            for task, heads in self.all_heads.items():
                for head_i, head in enumerate(heads):
                    target_name = head.get_target(task)
                    if target_name not in labels:
                        print(f"Target {target_name} not found. Skipping evaluation")
                        continue
                    res = head.metric(labels[target_name],
                                      preds[target_name])
                    out[target_name] = res
                    metrics_dict = flatten(res, separator='/')
                    for k, v in metrics_dict.items():
                        task_avg_tape[head.target_name.replace("{task}", "avg") + "/" + k].append(v)
            for k, v in task_avg_tape.items():
                # get the average
                out[k] = mean(v)

        # flatten everything
        out = flatten(out, separator='/')
        return out

    def save(self, file_path):
        """Save model to a file
        """
        from bpnet.utils import write_pkl, SerializableLock
        # fix the serialization of _OPERATIVE_CONFIG_LOCK
        gin.config._OPERATIVE_CONFIG_LOCK = SerializableLock()
        write_pkl(self, file_path)

    @classmethod
    def load(cls, file_path):
        """Load model from a file
        """
        from bpnet.utils import read_pkl
        return read_pkl(file_path)

    @classmethod
    def from_mdir(cls, model_dir):
        """Load the model from pkl
        """
        return cls.load(os.path.join(model_dir, 'seq_model.pkl'))


# avoid the decorator so that we can pickle it
config.external_configurable(SeqModel)

