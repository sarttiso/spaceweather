"""
Module for Bayesian deep learning in tensorflow.
"""

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from tensorflow.python.eager import context
from tensorflow.python.keras.layers.recurrent import LSTMCell
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.keras.layers import Layer
from tensorflow.python.keras import initializers, activations
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K



class Prior:
    """
    Prior function.

    Mixed Gaussians here.

    attributes:
        sig1: standard deviation of first Gaussian
        sig2: standard deviation of second Gaussian
        pi: Mixing coefficient between Gaussians.
        name: Name of prior, appended to above variables

    methods:
        __call__(w): probability of test weights given current parameters

    """
    def __init__(self, sig1=1.0, sig2=0.1, pi=0.2, name=''):
        self.sig1 = tf.Variable(sig1, name=name+'_sig1', trainable=True)
        self.sig2 = tf.Variable(sig2, name=name+'_sig2', trainable=True)
        self.pi = tf.Variable(pi, name=name+'_pi', trainable=True)
        self.prior_sig = tf.math.sqrt(self.pi*self.sig1**2 + (1-self.pi)*self.sig2**2).numpy()

    def __call__(self, w):
        """w: weights to evaluate prior on """
        self._dist1 = tfd.Normal(0.0, self.sig1)
        self._dist2 = tfd.Normal(0.0, self.sig2)
        P = self.pi*self._dist1.prob(w) + (1-self.pi)*self._dist2.prob(w)
        return P




class LSTMCellVariational(LSTMCell):
    """
    Extenstion of the LSTMCell defined in Tensorflow source. This cell contains two extra attributes; one that specifies a prior over weights and the other that specifies a weighting for the KL loss terms. Ultimately, it would be good to specify a general variational posterior form as another attribute, as is done for the Tensorflow implementation of a DenseVariational layer.

    attributes:
        units: dimensionality of hidden and cell states in the cell
        prior: callable prior object that returns probability of weights
        kl_weight: factor to weigh KL loss by; typically 1/nbatches

    methods:
        sample_weights(): samples weights from the variational posterior distribution
    """
    def __init__(self,
               units,
               prior,
               kl_weight,
               prior_trainable=True,
               **kwargs):
        super(LSTMCellVariational, self).__init__(units, **kwargs)
        self.prior = prior
        self.prior_trainable=prior_trainable
        self.kl_weight = kl_weight

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        """Instead of learning kernels, learn parameters that parameterize variational posterior distributions over the kernels."""
        input_dim = input_shape[-1]
        if self.prior_trainable:
            self._trainable_weights.append(self.prior.sig1)
            self._trainable_weights.append(self.prior.sig2)
            self._trainable_weights.append(self.prior.pi)

        self.kernel_mu = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel_mu',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.kernel_rho = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel_rho',
            initializer=initializers.Zeros,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.recurrent_kernel_mu = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel_mu',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        self.recurrent_kernel_rho = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel_rho',
            initializer=initializers.Zeros,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:

                def bias_initializer(_, *args, **kwargs):
                      return K.concatenate([
                          self.bias_initializer((self.units,), *args, **kwargs),
                          initializers.Ones()((self.units,), *args, **kwargs),
                          self.bias_initializer((self.units * 2,), *args, **kwargs),
                      ])
            else:
                bias_initializer = self.bias_initializer
            self.bias_mu = self.add_weight(
              shape=(self.units * 4,),
              name='bias_mu',
              initializer=bias_initializer,
              regularizer=self.bias_regularizer,
              constraint=self.bias_constraint)
            self.bias_rho = self.add_weight(
              shape=(self.units * 4,),
              name='bias_rho',
              initializer=initializers.Zeros,
              regularizer=self.bias_regularizer,
              constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def sample_weights(self):
        """Sample weights from variational posteriors"""
        self.kernel_sig = tf.math.softplus(self.kernel_rho)
        self.kernel = self.kernel_mu + \
            self.kernel_sig*tf.random.normal(self.kernel_mu.shape)

        self.recurrent_kernel_sig = tf.math.softplus(self.recurrent_kernel_rho)
        self.recurrent_kernel = self.recurrent_kernel_mu + \
          self.recurrent_kernel_sig*tf.random.normal(self.recurrent_kernel_mu.shape)

        if self.use_bias:
            self.bias_sig = tf.math.softplus(self.bias_rho)
            self.bias = self.bias_mu + \
                self.bias_sig*tf.random.normal(self.bias_mu.shape)

    def call(self, inputs, states, training=None):
        """Same call except that we sample weights from the variational posterior and add KL losses to the layer loss."""
        self.sample_weights()
        self.add_loss(self.kl_loss(self.kernel,
                                    self.kernel_mu,
                                    self.kernel_sig) +
                      self.kl_loss(self.recurrent_kernel,
                                    self.recurrent_kernel_mu,
                                    self.recurrent_kernel_sig))
        if self.use_bias:
            self.add_loss(self.kl_loss(self.bias, self.bias_mu, self.bias_sig))
        return super(LSTMCellVariational, self).call(inputs, states, training=None)

    def kl_loss(self, w, mu, sig):
        """Compute KL loss on variational posterior and prior."""
        # variational posterior term
        dist_variational = tfd.Normal(mu, sig)
        p_variational = dist_variational.log_prob(w)
        # prior term
        p_prior = K.log(self.prior(w))
        # now add up and sum over all terms
        KL = self.kl_weight * tf.reduce_sum(p_variational - p_prior)
        return KL


class LSTMVariational(LSTM):
    """Follow implementation of Tensorflow whereby LSTM objects containt LSTMCells, which I have altered here to both be of the variational form."""
    def __init__(self,
               units,
               prior,
               kl_weight,
               prior_trainable=True,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               **kwargs):
        if implementation == 0:
            logging.warning('`implementation=0` has been deprecated, '
                          'and now defaults to `implementation=1`.'
                          'Please update your layer call.')
        if context.executing_eagerly() and context.num_gpus() > 0:
            logging.warn('%s: Note that this layer is not optimized for performance. '
                       'Please use tf.keras.layers.CuDNNLSTM for better '
                       'performance on GPU.', self)
        cell = LSTMCellVariational(
            units,
            prior,
            kl_weight,
            prior_trainable=prior_trainable,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            unit_forget_bias=unit_forget_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            implementation=implementation)
        super(LSTM, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = [InputSpec(ndim=3)]

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell.reset_dropout_mask()
        self.cell.reset_recurrent_dropout_mask()
        return super(LSTM, self).call(
            inputs, mask=mask, training=training, initial_state=initial_state)


class DenseVariational(Layer):
    """Almost entirely taken from Krasser's tutorial on Bayes by Backprop. I've added my own prior formulation"""
    def __init__(self, output_dim, prior, kl_weight, prior_trainable=True, activation=None, **kwargs):
        self.output_dim = output_dim
        self.prior = prior
        self.kl_weight = kl_weight
        self.prior_trainable = prior_trainable
        self.activation = activations.get(activation)
        super().__init__(**kwargs)

    def build(self, input_shape):
        if self.prior_trainable:
            self._trainable_weights.append(self.prior.sig1)
            self._trainable_weights.append(self.prior.sig2)
            self._trainable_weights.append(self.prior.pi)

        self.kernel_mu = self.add_weight(name='kernel_mu',
                                    shape=(input_shape[1], self.output_dim),
                                    initializer=initializers.RandomNormal(stddev=self.prior.prior_sig),
                                    trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu',
                                    shape=(self.output_dim,),
                                    initializer=initializers.RandomNormal(stddev=self.prior.prior_sig),
                                    trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho',
                                    shape=(input_shape[1], self.output_dim),
                                    initializer=initializers.constant(0.0),
                                    trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho',
                                    shape=(self.output_dim,),
                                    initializer=initializers.constant(0.0),
                                    trainable=True)
        super().build(input_shape)

    def call(self, x):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)

        bias_sigma = tf.math.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)

        self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) +
                      self.kl_loss(bias, self.bias_mu, bias_sigma))

        return self.activation(K.dot(x, kernel) + bias)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def kl_loss(self, w, mu, sigma):
        dist_variational = tfd.Normal(mu, sigma)
        p_variational = dist_variational.log_prob(w)
        p_prior = K.log(self.prior(w))
        return self.kl_weight * tf.reduce_sum(p_variational - p_prior)
