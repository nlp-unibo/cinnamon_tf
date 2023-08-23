.. _model:

``TFNetwork``
*************************************

The ``TFNetwork`` is an implementation of ``Network`` for Tensorflow.

The following ``Network`` APIs are implemented:

- ``batch_loss``
- ``batch_train``
- ``batch_fit``
- ``batch_predict``
- ``batch_evaluate``
- ``save_model``
- ``load_model``
- ``fit``
- ``evaluate``
- ``predict``

These APIs are meant to be used with Tensorflow-specific data structures like `tf.data.Dataset <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_.
However, **no hardcoded types** are enforced: the ``TFNetwork`` works with ``FieldDict`` to ensure code flexibility.

.. note::
    The ``FieldDict`` is used to wrap data structures like ``tf.data.Dataset``

In particular, a ``FieldDict`` with the following keys **is required**:

- ``iterator``: the complete data iterator with both inputs and outputs.
- ``input_iterator``: the data iterator with only inputs.
- ``output_iterator``: the data iterator with only outputs (if any).
- ``steps``: the number of steps to take before exhausting the iterator.

For this reason, **it is always recommended** to define ad-hoc ``Processor`` components to transform input data into supported data formats.

For instance, the following ``TFTextTreeProcessor`` builds a ``tf.data.Dataset`` from input data.

.. code-block:: python

    class TFTextTreeProcessor(Processor):

        def get_input_data(
                self,
                data: FieldDict
        ):
            for x, y in zip(data.x, data.y):
                yield x, y

        def process(
                self,
                data: FieldDict,
                is_training_data: bool = False
        ) -> FieldDict:
            tf_data = tf.data.Dataset.from_generator(partial(self.get_input_data, data=data),
                                                       output_signature=(
                                                           tf.TensorSpec(shape=(None,), dtype=tf.int32),
                                                           tf.TensorSpec(shape=(), dtype=tf.int32)
                                                       ))

            if is_training_data:
                tf_data = tf_data.shuffle(buffer_size=self.buffer_size)

            tf_data = tf_data.padded_batch(batch_size=self.batch_size)

            if self.prefetch:
                tf_data = tf_data.prefetch(buffer_size=tf.data.AUTOTUNE)

            steps = math.ceil(len(data.nodes) / self.batch_size)

            return FieldDict({'iterator': lambda: iter(tf_data),
                              'input_iterator': lambda: iter(tf_data.map(lambda x, y: x)),
                              'output_iterator': lambda: iter(tf_data.map(lambda x, y: y).as_numpy_iterator()),
                              'steps': steps})