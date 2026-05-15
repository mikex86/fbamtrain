import triton
import triton.language as tl


@triton.jit
def conv2d_dgrad_kernel(
        dout_ptr,
        weight_ptr,
        dinput_ptr,
        batch,
        in_channels,
        in_h,
        in_w,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        pad_h,
        pad_w,
        out_h,
        out_w,
        dout_stride_n,
        dout_stride_h,
        dout_stride_w,
        dout_stride_c,
        weight_stride_oc,
        weight_stride_kh,
        weight_stride_kw,
        weight_stride_ic,
        dinput_stride_n,
        dinput_stride_h,
        dinput_stride_w,
        dinput_stride_c,
        BLOCK_W: tl.constexpr,
        BLOCK_OC: tl.constexpr,
        BLOCK_K: tl.constexpr,
        KERNEL_H: tl.constexpr,
        KERNEL_W: tl.constexpr,
        IN_CHANNELS: tl.constexpr,
        ACCUMULATE_IN_FP32: tl.constexpr,
        OUT_DTYPE_CODE: tl.constexpr,
):
    out_dtype = tl.bfloat16 if OUT_DTYPE_CODE == 0 else tl.float16
    acc_dtype = tl.float32 if ACCUMULATE_IN_FP32 else tl.float16

    pid_ic = tl.program_id(0)
    pid_nh = tl.program_id(1)
    pid_w = tl.program_id(2)

    ic_block = pid_ic

    ic_offsets = ic_block * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_ic = ic_offsets < IN_CHANNELS

    iw_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_iw = iw_offsets < in_w

    n = pid_nh // in_h
    ih = pid_nh - n * in_h

    acc = tl.zeros((BLOCK_W, BLOCK_K), dtype=acc_dtype)

    num_oc_blocks = tl.cdiv(out_channels, BLOCK_OC)
    for oc_block in range(0, num_oc_blocks):
        oc_offsets = oc_block * BLOCK_OC + tl.arange(0, BLOCK_OC)
        mask_oc = oc_offsets < out_channels

        for kh in tl.static_range(KERNEL_H):
            oh_raw = ih + pad_h - kh * dilation_h
            mask_oh = (oh_raw >= 0) & (oh_raw < out_h * stride_h) & (oh_raw % stride_h == 0)
            oh = oh_raw // stride_h
            for kw in tl.static_range(KERNEL_W):
                ow_raw = iw_offsets + pad_w - kw * dilation_w
                mask_ow = (ow_raw >= 0) & (ow_raw < out_w * stride_w) & (ow_raw % stride_w == 0)
                ow = ow_raw // stride_w
                safe_oh = tl.where(mask_oh, oh, 0)
                safe_ow = tl.where(mask_ow, ow, 0)

                dout_ptrs = (
                        dout_ptr
                        + n * dout_stride_n
                        + safe_oh * dout_stride_h
                        + safe_ow[:, None] * dout_stride_w
                        + oc_offsets[None, :] * dout_stride_c
                )
                mask_dout = mask_ow[:, None] & mask_oc[None, :] & mask_oh
                dout_vals = tl.load(dout_ptrs, mask=mask_dout, other=0.0)

                weight_ptrs = (
                        weight_ptr
                        + oc_offsets[:, None] * weight_stride_oc
                        + kh * weight_stride_kh
                        + kw * weight_stride_kw
                        + ic_offsets[None, :] * weight_stride_ic
                )
                weight_mask = mask_oc[:, None] & mask_ic[None, :]
                weight_vals = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
                if ACCUMULATE_IN_FP32:
                    dout_vals = dout_vals.to(tl.float32)
                    weight_vals = weight_vals.to(tl.float32)
                acc += tl.dot(dout_vals, weight_vals, out_dtype=acc_dtype)

    dinput_ptrs = (
            dinput_ptr
            + n * dinput_stride_n
            + ih * dinput_stride_h
            + iw_offsets[:, None] * dinput_stride_w
            + ic_offsets[None, :] * dinput_stride_c
    )
    tl.store(dinput_ptrs, acc.to(out_dtype), mask=mask_iw[:, None] & mask_ic[None, :])


@triton.jit
def conv2d_wgrad_kernel(
        input_ptr,
        dout_ptr,
        dweight_ptr,
        batch,
        in_channels,
        in_h,
        in_w,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        pad_h,
        pad_w,
        out_h,
        out_w,
        input_stride_n,
        input_stride_h,
        input_stride_w,
        input_stride_c,
        dout_stride_n,
        dout_stride_h,
        dout_stride_w,
        dout_stride_c,
        dweight_stride_oc,
        dweight_stride_kh,
        dweight_stride_kw,
        dweight_stride_ic,
        accumulate_output,
        BLOCK_W: tl.constexpr,
        BLOCK_OC: tl.constexpr,
        BLOCK_K: tl.constexpr,
        KERNEL_H: tl.constexpr,
        KERNEL_W: tl.constexpr,
        IN_CHANNELS: tl.constexpr,
        ACCUMULATE_IN_FP32: tl.constexpr,
        OUT_DTYPE_CODE: tl.constexpr,
):
    out_dtype = tl.bfloat16 if OUT_DTYPE_CODE == 0 else tl.float16
    acc_dtype = tl.float32 if ACCUMULATE_IN_FP32 else tl.float16

    pid_oc = tl.program_id(0)
    pid_ic = tl.program_id(1)
    pid_k = tl.program_id(2)

    oc_offsets = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = oc_offsets < out_channels

    ic_offsets = pid_ic * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_ic = ic_offsets < IN_CHANNELS

    kh = pid_k // KERNEL_W
    kw = pid_k - kh * KERNEL_W

    total_p = batch * out_h * out_w
    num_p_blocks = tl.cdiv(total_p, BLOCK_W)

    acc = tl.zeros((BLOCK_OC, BLOCK_K), dtype=acc_dtype)
    for pid_p in range(0, num_p_blocks):
        p_offsets = pid_p * BLOCK_W + tl.arange(0, BLOCK_W)
        mask_p = p_offsets < total_p

        n = p_offsets // (out_h * out_w)
        rem = p_offsets - n * out_h * out_w
        oh = rem // out_w
        ow = rem - oh * out_w

        ih = oh * stride_h - pad_h + kh * dilation_h
        iw = ow * stride_w - pad_w + kw * dilation_w

        mask_hw = (ih >= 0) & (ih < in_h) & (iw >= 0) & (iw < in_w)
        mask_pixels = mask_p & mask_hw
        safe_n = tl.where(mask_pixels, n, 0)
        safe_oh = tl.where(mask_pixels, oh, 0)
        safe_ow = tl.where(mask_pixels, ow, 0)
        safe_ih = tl.where(mask_hw, ih, 0)
        safe_iw = tl.where(mask_hw, iw, 0)

        input_ptrs = (
                input_ptr
                + safe_n[:, None] * input_stride_n
                + safe_ih[:, None] * input_stride_h
                + safe_iw[:, None] * input_stride_w
                + ic_offsets[None, :] * input_stride_c
        )
        input_mask = mask_pixels[:, None] & mask_ic[None, :]
        input_vals = tl.load(input_ptrs, mask=input_mask, other=0.0)

        dout_ptrs = (
                dout_ptr
                + safe_n[:, None] * dout_stride_n
                + safe_oh[:, None] * dout_stride_h
                + safe_ow[:, None] * dout_stride_w
                + oc_offsets[None, :] * dout_stride_c
        )
        dout_mask = mask_pixels[:, None] & mask_oc[None, :]
        dout_vals = tl.load(dout_ptrs, mask=dout_mask, other=0.0)
        if ACCUMULATE_IN_FP32:
            input_vals = input_vals.to(tl.float32)
            dout_vals = dout_vals.to(tl.float32)
        acc += tl.dot(tl.trans(dout_vals), input_vals, out_dtype=acc_dtype)

    dweight_ptrs = (
            dweight_ptr
            + oc_offsets[:, None] * dweight_stride_oc
            + kh * dweight_stride_kh
            + kw * dweight_stride_kw
            + ic_offsets[None, :] * dweight_stride_ic
    )
    accumulate_mask = mask_oc[:, None] & mask_ic[None, :] & (accumulate_output != 0)
    existing = tl.load(dweight_ptrs, mask=accumulate_mask, other=0.0).to(acc_dtype)
    acc = acc + existing
    tl.store(dweight_ptrs, acc.to(out_dtype), mask=mask_oc[:, None] & mask_ic[None, :])
