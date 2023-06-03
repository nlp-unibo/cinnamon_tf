import abc
import gc
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Dict, Union, AnyStr, Tuple, Iterator

import tensorflow as tf
from tqdm import tqdm

from cinnamon_core.core.data import FieldDict
from cinnamon_core.utility import logging_utility
from cinnamon_generic.components.callback import Callback, guard
from cinnamon_generic.components.metrics import Metric
from cinnamon_generic.components.model import Network
from cinnamon_generic.utility.printing_utility import prettify_statistics


class TFNetwork(Network):

    def batch_train(
            self,
            batch_x: Any,
            batch_y: Any,
            batch_args: Optional[Dict] = None) -> Any:
        with tf.GradientTape() as tape:
            loss, true_loss, loss_info, model_additional_info = self.batch_loss(batch_x,
                                                                                batch_y,
                                                                                batch_args=batch_args)
        grads = tape.gradient(loss, self.model.trainable_variables)
        return loss, loss_info, model_additional_info, grads

    @guard
    def batch_fit(
            self,
            batch_x: Any,
            batch_y: Any,
            batch_args: Optional[Dict] = None
    ) -> Any:
        loss, loss_info, model_additional_info, grads = self.batch_train(batch_x=batch_x,
                                                                         batch_y=batch_y,
                                                                         batch_args=batch_args)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        train_loss_info = {f'train_{key}': item for key, item in loss_info.items()}
        train_loss_info['train_loss'] = loss
        return train_loss_info, model_additional_info

    @guard
    def batch_predict(
            self,
            batch_x: Any,
            batch_args: Optional[Dict] = None
    ):
        batch_args = batch_args if batch_args is not None else {}
        batch_args['training'] = False
        predictions, model_additional_info = self.model(batch_x,
                                                        **batch_args)
        return predictions, model_additional_info

    @guard
    def batch_evaluate(
            self,
            batch_x: Any,
            batch_y: Any,
            batch_args: Optional[Dict] = None
    ):
        batch_args = batch_args if batch_args is not None else {}
        batch_args['training'] = False
        loss, loss_info = self.batch_loss(batch_x=batch_x,
                                          batch_y=batch_y,
                                          batch_args=batch_args)
        val_loss_info = {f'val_{key}': item for key, item in loss_info.items()}
        val_loss_info['val_loss'] = loss
        return val_loss_info

    @guard
    def batch_evaluate_and_predict(
            self,
            batch_x: Any,
            batch_y: Any,
            batch_args: Optional[Dict] = None
    ):
        batch_args = batch_args if batch_args is not None else {}
        batch_args['training'] = False
        batch_args['return_predictions'] = True
        loss, loss_info, predictions, model_additional_info = self.batch_loss(batch_x=batch_x,
                                                                              batch_y=batch_y,
                                                                              batch_args=batch_args)
        val_loss_info = {f'val_{key}': item for key, item in loss_info.items()}
        val_loss_info['val_loss'] = loss
        return val_loss_info, predictions, model_additional_info

    @guard
    def save_model(
            self,
            filepath: Union[AnyStr, Path]
    ):
        self.model.save_weights(filepath=filepath.joinpath('weights.h5'))

    @guard
    def load_model(self, filepath: Union[AnyStr, Path]):
        self.model.load_weights(filepath=filepath.joinpath('weights.h5'))

    @abc.abstractmethod
    def get_model_data(
            self,
            data: FieldDict,
            with_labels: bool = False
    ) -> Tuple[Iterator, int]:
        pass

    @guard
    def fit(
            self,
            train_data: FieldDict,
            val_data: Optional[FieldDict] = None,
            callbacks: Optional[Callback] = None,
            metrics: Optional[Metric] = None
    ) -> FieldDict:
        shuffled_train_iterator, train_steps = self.get_model_data(data=train_data,
                                                                   with_labels=True)

        logging_utility.logger.info('Training started...')
        logging_utility.logger.info(f'Total steps: {train_steps}')

        training_info = {}
        for epoch in range(self.epochs):

            if self.model.stop_training:
                break

            if callbacks:
                callbacks.run(hookpoint='on_epoch_begin',
                              logs={'epochs': self.epochs})

            epoch_info = defaultdict(float)
            batch_idx = 0

            with tqdm(total=train_steps, leave=True, position=0, desc=f'[Training] Epoch {epoch}') as pbar:
                while batch_idx < train_steps:
                    batch_info, model_additional_info = self.batch_fit(*next(shuffled_train_iterator))
                    batch_info = {key: item.numpy() for key, item in batch_info.items()}

                    for key, item in batch_info.items():
                        epoch_info[key] += item

                    batch_idx += 1
                    pbar.update(1)

            epoch_info = {key: item / train_steps for key, item in epoch_info.items()}

            if callbacks:
                callbacks.run(hookpoint='on_epoch_end',
                              logs={'epoch': epoch, 'epoch_info': epoch_info})

            epoch_info = {**epoch_info, **{'epoch': epoch + 1}}
            if val_data is not None:
                val_info = self.evaluate(data=val_data,
                                         callbacks=callbacks,
                                         metrics=metrics)

                epoch_info = {**epoch_info, **val_info.loss}
                if metrics is not None:
                    epoch_info = {**epoch_info, **val_info.metrics}

            logging_utility.logger.info(f'\n{prettify_statistics(epoch_info)}')

            if callbacks:
                callbacks.run(hookpoint='on_epoch_end',
                              logs=epoch_info)

            for key, value in epoch_info.items():
                training_info.setdefault(key, []).append(value)

            # Garbage collect
            gc.collect()

        return FieldDict(training_info)

    @guard
    def evaluate(
            self,
            data: FieldDict,
            callbacks: Optional[Callback] = None,
            metrics: Optional[Metric] = None
    ) -> FieldDict:
        loss = defaultdict(float)

        data_iterator, data_steps = self.get_model_data(data=data, with_labels=True)

        for _ in tqdm(range(data_steps), leave=True, position=0):
            batch_loss, \
                true_batch_loss, \
                batch_loss_info, \
                batch_predictions, \
                model_additional_info = self.batch_evaluate_and_predict(*next(data_iterator))

            batch_info = {f'val_{key}': item.numpy() for key, item in batch_loss_info.items()}
            batch_info['val_loss'] = batch_loss

            for key, item in batch_info.items():
                loss[key] += item

        loss = {key: item / data_steps for key, item in loss.items()}
        return FieldDict(loss)

    @guard
    def evaluate_and_predict(
            self,
            data: FieldDict,
            callbacks: Optional[Callback] = None,
            metrics: Optional[Metric] = None
    ) -> FieldDict:
        loss = defaultdict(float)
        predictions = []

        data_iterator, data_steps = self.get_model_data(data=data, with_labels=True)

        for _ in tqdm(range(data_steps), leave=True, position=0):
            batch_loss, \
                true_batch_loss, \
                batch_loss_info, \
                batch_predictions, \
                model_additional_info = self.batch_evaluate_and_predict(*next(data_iterator))

            batch_info = {f'val_{key}': item.numpy() for key, item in batch_loss_info.items()}
            batch_info['val_loss'] = batch_loss

            for key, item in batch_info.items():
                loss[key] += item

            predictions.append(self.parse_model_output(batch_predictions.numpy(), model_additional_info))

        loss = {key: item / data_steps for key, item in loss.items()}

        return FieldDict({**loss, **{'predictions': predictions}})

    @guard
    def predict(
            self,
            data: FieldDict,
            callbacks: Optional[Callback] = None,
            metrics: Optional[Metric] = None
    ) -> FieldDict:
        predictions = {}

        data_iterator, data_steps = self.get_model_data(data=data, with_labels=False)

        for _ in tqdm(range(data_steps), leave=True, position=0):
            output, model_additional_info = self.batch_predict(*next(data_iterator))
            output = {key: self.parse_model_output(value.numpy(), model_additional_info)
                      for key, value in output.items()}
            for key, value in output.items():
                predictions.setdefault(key, []).extend(value)

        return FieldDict(predictions)

    @abc.abstractmethod
    def parse_model_output(
            self,
            output: Any,
            model_additional_info: Dict
    ) -> Any:
        pass
