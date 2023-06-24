from typing import Type, Iterable

from cinnamon_core.core.configuration import Configuration, C
from cinnamon_core.core.registry import Registry, register
from cinnamon_tf.components.helper import TFHelper


class TFHelperConfig(Configuration):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add(name='deterministic',
                   value=False,
                   type_hint=bool,
                   description='If enabled, tensorflow will run operations in deterministic mode.'
                               'Note that this behaviour may seldom raise CUDA-related errors.')
        config.add(name='limit_gpu_visibility',
                   value=True,
                   type_hint=bool,
                   description='If enabled, it forces Tensorflow gpu visibility to the specified devices only')
        config.add(name='gpu_indexes',
                   value=[0],
                   type_hint=Iterable[int],
                   description='List of gpu indexes to make available to Tensorflow')
        config.add(name='eager_execution',
                   value=False,
                   type_hint=bool,
                   description='Whether to execute in eager or in graph mode.')

        # If limit_gpu_visibility, make sure we have some indexes
        config.add_condition(name='gpu_visibility',
                             condition=lambda params: (params.limit_gpu_visibility and params.gpu_indexes is not None)
                                                      or not params.limit_gpu_visibility)

        return config


@register
def register_helpers():
    Registry.add_and_bind(config_class=TFHelperConfig,
                          component_class=TFHelper,
                          name='helper',
                          is_default=True,
                          namespace='tf')
    Registry.add_and_bind(config_class=TFHelperConfig,
                          config_constructor=TFHelperConfig.get_delta_class_copy,
                          config_kwargs={
                              'params': {
                                  'eager_execution': True
                              }
                          },
                          component_class=TFHelper,
                          name='helper',
                          tags={'eager'},
                          namespace='tf')
