from typing import Optional, Dict

import numpy as np

from cinnamon_core.utility import logging_utility
from cinnamon_generic.components.callback import Callback


class TFEarlyStopping(Callback):
    """Stop training when a monitored quantity has stopped improving.
    Arguments:
        monitor: Quantity to be monitored.
        min_delta: Minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: Number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: One of `{"auto", "min", "max"}`. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity.
            Training will stop if the model doesn't show improvement over the
            baseline.
        restore_best_weights: Whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    Example:
    ```python
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    # This callback will stop the training when there is no improvement in
    # the validation loss for three consecutive epochs.
    model.fit(data, labels, epochs=100, callbacks=[callback],
        validation_data=(val_data, val_labels))
    ```
    """

    def __init__(
            self,
            **kwargs
    ):
        super(TFEarlyStopping, self).__init__(**kwargs)

        self.wait: int = 0
        self.stopped_epoch: int = 0
        self.best_value: Optional[float] = None
        self.best_weights: Optional[int] = None

        if self.mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
            self.best_value = np.inf
        elif self.mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
            self.best_value = -np.inf

    def reset(
            self
    ):
        """
        Resets the ``EarlyStopping`` internal state for component re-use.
        """

        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        if self.baseline is not None:
            self.best_value = self.baseline
        else:
            self.best_value = np.Inf if self.monitor_op == np.less else -np.Inf

        self.component.model.stop_training = False

    def on_fit_begin(
            self,
            logs: Optional[Dict] = None
    ):
        # Allow instances to be re-used
        self.reset()

    def on_epoch_end(
            self,
            logs: Optional[Dict] = None
    ):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.monitor_op(current - self.min_delta, self.best_value):
            self.best_value = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.component.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = logs['epoch']
                self.component.model.stop_training = True
                if self.restore_best_weights:
                    logging_utility.logger.info('Restoring model weights from the end of the best epoch.')
                    self.component.model.set_weights(self.best_weights)

    def on_fit_end(
            self,
            logs: Optional[Dict] = None
    ):
        if self.stopped_epoch > 0:
            logging_utility.logger.info(f'Epoch {self.stopped_epoch + 1:.2f} early stopping')

        self.reset()

    def get_monitor_value(
            self,
            logs: Optional[Dict] = None
    ):
        logs = logs if logs is not None else {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging_utility.logger.warning(
                f'Early stopping conditioned on metric {self.monitor} which is not available.'
                f' Available metrics are: {",".join(list(logs.keys()))}')
        return monitor_value
