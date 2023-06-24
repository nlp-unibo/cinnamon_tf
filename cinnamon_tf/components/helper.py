import os
from typing import Optional, Any

import tensorflow as tf
from tensorflow.python.keras import backend as K

from cinnamon_core.utility import logging_utility
from cinnamon_generic.components.helper import Helper


class TFHelper(Helper):

    def set_seed(
            self,
            seed: int
    ):
        super().set_seed(seed=seed)
        tf.random.set_seed(seed)

        if self.deterministic:
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            try:
                tf.config.experimental.enable_op_determinism()
            except AttributeError:
                pass

    def clear_status(
            self
    ):
        K.clear_session()

    def limit_gpu_usage(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')

        # avoid other GPUs
        if self.limit_gpu_visibility:
            tf.config.set_visible_devices([gpu for idx, gpu in enumerate(gpus) if idx in self.gpu_indexes], "GPU")

        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logging_utility.logger.exception(e)

    def enable_eager_execution(self):
        assert tf.version.VERSION.startswith('2.'), \
            "Tensorflow version is not 2.X! This framework only supports >= 2.0 TF versions"
        tf.config.run_functions_eagerly(self.eager_execution)

    def run(
            self,
            seed: Optional[int] = None
    ) -> Any:
        self.set_seed(seed=seed)
        self.limit_gpu_usage()

        if self.eager_execution:
            self.enable_eager_execution()
