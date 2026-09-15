from .device import get_device, configure_device, compile_model, wrap_model_for_multi_gpu
from .flops import estimate_model_flops, compare_models_flops, count_parameters