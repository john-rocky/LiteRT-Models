"""transformers 4.57 <-> 5.x compat shims for rfdetr 1.9.x (vendored windowed-DINOv2).
Import this BEFORE importing rfdetr. No-op on transformers >= 5.1.
"""
import inspect
import transformers

if not hasattr(transformers, "BackboneConfigMixin"):
    from transformers.utils.backbone_utils import BackboneConfigMixin, BackboneMixin
    transformers.BackboneConfigMixin = BackboneConfigMixin
    transformers.BackboneMixin = BackboneMixin

from transformers.utils.backbone_utils import BackboneMixin as _BM
if hasattr(_BM, "_init_transformers_backbone"):
    _orig = _BM._init_transformers_backbone
    if "config" in inspect.signature(_orig).parameters:
        def _compat(self, config=None):
            return _orig(self, self.config if config is None else config)
        _BM._init_transformers_backbone = _compat
