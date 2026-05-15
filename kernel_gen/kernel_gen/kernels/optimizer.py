import triton
import triton.language as tl


@triton.jit
def adamw_step(
    param_ptr,
    grad_ptr,
    m_ptr,
    v_ptr,
    bias_correction1_ptr,
    bias_correction2_ptr,
    n_elements: tl.uint32,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    BLOCK_SIZE: tl.constexpr,
):
    out_dtype = param_ptr.dtype.element_ty
    bias_correction1 = tl.load(bias_correction1_ptr).to(tl.float32)
    bias_correction2 = tl.load(bias_correction2_ptr).to(tl.float32)

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets = offsets.to(tl.uint32)
    mask = offsets < n_elements

    param = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(m_ptr + offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + offsets, mask=mask, other=0.0)

    param = param - lr * weight_decay * param
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * grad * grad

    m_hat = m / bias_correction1
    v_hat = v / bias_correction2
    param = param - lr * (m_hat / (tl.sqrt(v_hat) + eps))

    tl.store(m_ptr + offsets, m, mask=mask)
    tl.store(v_ptr + offsets, v, mask=mask)
    tl.store(param_ptr + offsets, param.to(out_dtype), mask=mask)


@triton.jit
def sgd_step(
    param_ptr,
    grad_ptr,
    velocity_ptr,
    n_elements: tl.uint32,
    lr,
    momentum,
    weight_decay,
    nesterov: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    out_dtype = param_ptr.dtype.element_ty

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets = offsets.to(tl.uint32)
    mask = offsets < n_elements

    param = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    velocity = tl.load(velocity_ptr + offsets, mask=mask, other=0.0)

    grad = grad + weight_decay * param
    velocity = momentum * velocity + grad

    use_nesterov = nesterov != 0
    update = tl.where(use_nesterov, grad + momentum * velocity, velocity)
    param = param - lr * update

    tl.store(velocity_ptr + offsets, velocity, mask=mask)
    tl.store(param_ptr + offsets, param.to(out_dtype), mask=mask)
