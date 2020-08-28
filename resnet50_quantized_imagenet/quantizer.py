import torch
import pdb


def asymmetric_linear_quantization_params(num_bits, saturation_min, saturation_max,
                                          integral_zero_point=True, signed=False):
    scalar_min, sat_min = _prep_saturation_val_tensor(saturation_min)
    scalar_max, sat_max = _prep_saturation_val_tensor(saturation_max)
    is_scalar = scalar_min and scalar_max

    if scalar_max and not scalar_min:
        sat_max = sat_max.to(sat_min.device)
    elif scalar_min and not scalar_max:
        sat_min = sat_min.to(sat_max.device)

    if any(sat_min > sat_max):
        raise ValueError('saturation_min must be smaller than saturation_max')

    n = 2 ** num_bits - 1

    # Make sure 0 is in the range
    sat_min = torch.min(sat_min, torch.zeros_like(sat_min))
    sat_max = torch.max(sat_max, torch.zeros_like(sat_max))

    diff = sat_max - sat_min
    # If float values are all 0, we just want the quantized values to be 0 as well. So overriding the saturation
    # value to 'n', so the scale becomes 1
    diff[diff == 0] = n

    scale = n / diff
    zero_point = scale * sat_min
    if integral_zero_point:
        zero_point = zero_point.round()
    if signed:
        zero_point += 2 ** (num_bits - 1)
    if is_scalar:
        return scale.item(), zero_point.item()
    return scale, zero_point


def clamp(input, min, max, inplace=False):
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)

def linear_quantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    return torch.round(scale * input - zero_point)


def linear_quantize_clamp(input, scale, zero_point, clamp_min, clamp_max, inplace=False):
    output = linear_quantize(input, scale, zero_point, inplace)
    return clamp(output, clamp_min, clamp_max, inplace)


def linear_dequantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale

class LinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, dequantize, inplace):
        if inplace:
            ctx.mark_dirty(input)
        output = linear_quantize(input, scale, zero_point, inplace)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None, None, None

def dorefa_quantize_param(param_fp, num_bits, dequantize=False):
    sat_min = torch.min(param_fp)
    sat_max = torch.max(param_fp)
    scale, zero_point = asymmetric_linear_quantization_params(num_bits, sat_min, sat_max, signed=False)
    out = LinearQuantizeSTE.apply(param_fp, scale, zero_point, dequantize, False)
    return scale, zero_point, out


class DorefaParamsBinarizationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, inplace=False):
        if inplace:
            ctx.mark_dirty(input)
        E = input.abs().mean()
        output = torch.where(input == 0, torch.ones_like(input), torch.sign(input)) * E
        return output
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def _prep_saturation_val_tensor(sat_val):
    is_scalar = not isinstance(sat_val, torch.Tensor)
    out = torch.tensor(sat_val) if is_scalar else sat_val.clone().detach()
    if not out.is_floating_point():
        out = out.to(torch.float32)
    if out.dim() == 0:
        out = out.unsqueeze(0)
    return is_scalar, out

# Testing 
if __name__ == '__main__':
    param_fp = torch.randn(5, 3, 3, 3)
    out = dorefa_quantize_param(param_fp, 16)
# class DorefaQuantizer(Quantizer):
#     """
#     Quantizer using the DoReFa scheme, as defined in:
#     Zhou et al., DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
#     (https://arxiv.org/abs/1606.06160)
#     Notes:
#         1. Gradients quantization not supported yet
#     """
#     def __init__(self, model, optimizer,
#                  bits_activations=32, bits_weights=32, bits_bias=None,
#                  overrides=None):
#         super(DorefaQuantizer, self).__init__(model, optimizer=optimizer, bits_activations=bits_activations,
#                                               bits_weights=bits_weights, bits_bias=bits_bias,
#                                               train_with_fp_copy=True, overrides=overrides)

#         def relu_replace_fn(module, name, qbits_map):
#             bits_acts = qbits_map[name].acts
#             if bits_acts is None:
#                 return module
#             return ClippedLinearQuantization(bits_acts, 1, dequantize=True, inplace=module.inplace)

#         self.param_quantization_fn = dorefa_quantize_param

#         self.replacement_factory[nn.ReLU] = relu_replace_fn
