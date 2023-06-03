from typing import Type, Iterable

from cinnamon_core.core.configuration import Configuration, C
from cinnamon_core.core.registry import Registry, register
from components.helper import TFHelper


class TFHelperConfig(Configuration):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add_short(name='limit_gpu_visibility',
                         value=True,
                         type_hint=bool,
                         description='If enabled, it forces Tensorflow gpu visibility to the specified devices only')
        config.add_short(name='gpu_indexes',
                         value=[0],
                         type_hint=Iterable[int],
                         description='List of gpu indexes to make available to Tensorflow')
        config.add_short(name='eager_execution',
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
    Registry.register_and_bind(configuration_class=TFHelperConfig,
                               component_class=TFHelper,
                               name='helper',
                               is_default=True,
                               namespace='tf')
