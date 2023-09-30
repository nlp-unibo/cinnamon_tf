.. _catalog:

Available ``Configuration``
*************************************

Currently, ``cinnamon-tf`` provides the following registered ``Configuration``.


-------------------
Callback
-------------------

- ``name='callback', tags={'early_stopping'}, namespace='tf'``: the default ``TFEarlyStoppingConfig``.

-------------------
Helper
-------------------

- ``name='helper', tags={'default'}, namespace='tf'``: the default ``TFHelperConfig``.
- ``name='helper`, tags={'eager'}, namespace='tf'``: the ``TFHelperConfig`` with eager execution enabled.