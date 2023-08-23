.. _callback:

``TFEarlyStopping``
*************************************

Stop training when a monitored quantity has stopped improving.

The ``TFEarlyStopping`` wraps the early stopping implementation of `Keras <https://keras.io/api/callbacks/early_stopping/>`_ to work with any ``Model`` component.

The ``TFEarlyStopping`` uses the ``TFEarlyStoppingConfig`` as default configuration template

.. code-block:: python

    class TFEarlyStoppingConfig(Configuration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()

            config.add(name='monitor',
                       value='val_loss',
                       type_hint=str,
                       is_required=True,
                       description='Metric/Loss to monitor for performing early stopping')

            config.add(name='patience',
                       value=5,
                       type_hint=int,
                       is_required=True,
                       allowed_range=lambda patience: patience > 0,
                       description='Number of epochs to wait for monitored value to improve before stopping training')

            config.add(name='baseline',
                       value=None,
                       type_hint=Optional[float],
                       description='Threshold value to consider for start monitoring')

            config.add(name='min_delta',
                       value=1e-06,
                       type_hint=float,
                       description='Minimum delta difference allowed in checking for new best monitor values')

            config.add(name='restore_best_weights',
                       value=True,
                       type_hint=bool,
                       description='If enabled, the best epoch weights will be restored before stopping training')

            config.add(name='mode',
                       value='min',
                       type_hint=str,
                       description='How to compute best monitor scores. '
                                   'If the monitored value gets better as it diminishes or viceversa',
                       allowed_range=lambda mode: mode.casefold() in ['min', 'max'])

            return config

***************************
Registered configurations
***************************

The ``cinnamon-tf`` package provides the following registered configurations:

- ``name='callback', tags={'early_stopping'}, namespace='tf'``: the default ``TFEarlyStoppingConfig``.

