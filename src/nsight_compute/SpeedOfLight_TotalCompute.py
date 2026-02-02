# Copyright (c) 2026, GENERATED
# This file adds a rule that reports total compute throughput (scalar FLOP/s and tensor OP/s).
import NvRules
from RequestedMetrics import Importance, MetricRequest, RequestedMetricsParser

# Scalar FP metrics (per-cycle achieved). Use Importance.OPTIONAL and default 0.
requested_metrics = [
    MetricRequest("sm__cycles_elapsed.avg.per_second", "sm_cycles_per_second", Importance.OPTIONAL, 0, False),

    # FP32
    MetricRequest("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed", "fadd", Importance.OPTIONAL, 0, False),
    MetricRequest("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed", "fmul", Importance.OPTIONAL, 0, False),
    MetricRequest("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed", "ffma", Importance.OPTIONAL, 0, False),
    # GB10x 2-wide (optional)
    MetricRequest("smsp__sass_thread_inst_executed_op_fadd2_pred_on.sum.per_cycle_elapsed", "fadd2", Importance.OPTIONAL, 0, False),
    MetricRequest("smsp__sass_thread_inst_executed_op_fmul2_pred_on.sum.per_cycle_elapsed", "fmul2", Importance.OPTIONAL, 0, False),
    MetricRequest("smsp__sass_thread_inst_executed_op_ffma2_pred_on.sum.per_cycle_elapsed", "ffma2", Importance.OPTIONAL, 0, False),

    # FP64
    MetricRequest("smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed", "dadd", Importance.OPTIONAL, 0, False),
    MetricRequest("smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed", "dmul", Importance.OPTIONAL, 0, False),
    MetricRequest("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed", "dfma", Importance.OPTIONAL, 0, False),

    # FP16 scalar
    MetricRequest("smsp__sass_thread_inst_executed_op_hadd_pred_on.sum.per_cycle_elapsed", "hadd", Importance.OPTIONAL, 0, False),
    MetricRequest("smsp__sass_thread_inst_executed_op_hmul_pred_on.sum.per_cycle_elapsed", "hmul", Importance.OPTIONAL, 0, False),
    MetricRequest("smsp__sass_thread_inst_executed_op_hfma_pred_on.sum.per_cycle_elapsed", "hfma", Importance.OPTIONAL, 0, False),

    # Some useful tensor paths (per-cycle). If missing they will be treated as 0.
    MetricRequest("sm__ops_path_tensor_src_fp16_bf16_tf32_dst_fp32.sum.per_cycle_elapsed", "tensor_fp16_bf16_tf32_fp32", Importance.OPTIONAL, 0, False),
    MetricRequest("sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_off.sum.per_cycle_elapsed", "tensor_bf16_fp32_off", Importance.OPTIONAL, 0, False),
    MetricRequest("sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_on.sum.per_cycle_elapsed", "tensor_bf16_fp32_on", Importance.OPTIONAL, 0, False),
    MetricRequest("sm__ops_path_tensor_src_fp16_dst_fp16_sparsity_off.sum.per_cycle_elapsed", "tensor_fp16_fp16_off", Importance.OPTIONAL, 0, False),
    MetricRequest("sm__ops_path_tensor_src_fp16_dst_fp16_sparsity_on.sum.per_cycle_elapsed", "tensor_fp16_fp16_on", Importance.OPTIONAL, 0, False),
    MetricRequest("sm__ops_path_tensor_src_tf32_dst_fp32_sparsity_off.sum.per_cycle_elapsed", "tensor_tf32_fp32_off", Importance.OPTIONAL, 0, False),
    MetricRequest("sm__ops_path_tensor_src_tf32_dst_fp32_sparsity_on.sum.per_cycle_elapsed", "tensor_tf32_fp32_on", Importance.OPTIONAL, 0, False),
    MetricRequest("sm__ops_path_tensor_src_fp64.sum.per_cycle_elapsed", "tensor_fp64", Importance.OPTIONAL, 0, False),
    MetricRequest("sm__ops_path_tensor_src_int8_sparsity_off.sum.per_cycle_elapsed", "tensor_int8_off", Importance.OPTIONAL, 0, False),
    MetricRequest("sm__ops_path_tensor_src_int8_sparsity_on.sum.per_cycle_elapsed", "tensor_int8_on", Importance.OPTIONAL, 0, False),
]


def get_identifier():
    return "SOLTotalCompute"


def get_name():
    return "Total Compute Throughput"


def get_description():
    return "Aggregates scalar FLOP/s (FP32/FP64/FP16) and tensor OP/s into a single report (Python-side)" 


def get_section_identifier():
    return "SpeedOfLight_RooflineChart"


def _metric_val(metrics, key):
    m = metrics.get(key)
    if m is None:
        return 0.0
    try:
        return float(m.value())
    except Exception:
        return 0.0


def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    metrics = RequestedMetricsParser(handle, action).parse(requested_metrics)

    sm_freq = _metric_val(metrics, "sm_cycles_per_second") or _metric_val(metrics, "smsp__cycles_elapsed.avg.per_second") or 0.0

    # Scalar ops per cycle
    fadd = _metric_val(metrics, "fadd")
    fmul = _metric_val(metrics, "fmul")
    ffma = _metric_val(metrics, "ffma")

    fadd2 = _metric_val(metrics, "fadd2")
    fmul2 = _metric_val(metrics, "fmul2")
    ffma2 = _metric_val(metrics, "ffma2")

    achieved_fp32_per_cycle = fadd + fmul + 2.0 * ffma + 2.0 * fadd2 + 2.0 * fmul2 + 4.0 * ffma2

    dadd = _metric_val(metrics, "dadd")
    dmul = _metric_val(metrics, "dmul")
    dfma = _metric_val(metrics, "dfma")
    achieved_fp64_per_cycle = dadd + dmul + 2.0 * dfma

    hadd = _metric_val(metrics, "hadd")
    hmul = _metric_val(metrics, "hmul")
    hfma = _metric_val(metrics, "hfma")
    achieved_fp16_per_cycle = hadd + hmul + 4.0 * hfma

    # Tensor ops per cycle (sum representative paths)
    tensor_sum_per_cycle = (
        _metric_val(metrics, "tensor_fp16_bf16_tf32_fp32")
        + _metric_val(metrics, "tensor_bf16_fp32_off")
        + _metric_val(metrics, "tensor_bf16_fp32_on")
        + _metric_val(metrics, "tensor_fp16_fp16_off")
        + _metric_val(metrics, "tensor_fp16_fp16_on")
        + _metric_val(metrics, "tensor_tf32_fp32_off")
        + _metric_val(metrics, "tensor_tf32_fp32_on")
        + _metric_val(metrics, "tensor_fp64")
        + _metric_val(metrics, "tensor_int8_off")
        + _metric_val(metrics, "tensor_int8_on")
    )

    # Convert to per-second
    achieved_fp32_per_sec = achieved_fp32_per_cycle * sm_freq
    achieved_fp64_per_sec = achieved_fp64_per_cycle * sm_freq
    achieved_fp16_per_sec = achieved_fp16_per_cycle * sm_freq
    tensor_ops_per_sec = tensor_sum_per_cycle * sm_freq

    scalar_flops_per_sec = achieved_fp32_per_sec + achieved_fp64_per_sec + achieved_fp16_per_sec
    combined_total = scalar_flops_per_sec + tensor_ops_per_sec

    # Emit a message and focus metrics
    msg = (
        f"Scalar FLOP/s (FP32+FP64+FP16): {scalar_flops_per_sec:.2f}; "
        f"Tensor OP/s: {tensor_ops_per_sec:.2f}; Total(combined): {combined_total:.2f}"
    )

    msg_id = fe.message(NvRules.MsgType.OK, msg, "Total Compute Throughput")

    fe.focus_metric(msg_id, "scalar_flops_per_sec", scalar_flops_per_sec, NvRules.IFrontend.Severity_SEVERITY_DEFAULT, "Sum of FP32/FP64/FP16 op/s (device)")
    fe.focus_metric(msg_id, "tensor_ops_per_sec", tensor_ops_per_sec, NvRules.IFrontend.Severity_SEVERITY_DEFAULT, "Sum of selected tensor OP/s paths (device)")
    fe.focus_metric(msg_id, "combined_total", combined_total, NvRules.IFrontend.Severity_SEVERITY_DEFAULT, "Scalar + Tensor combined throughput")

    # Also send dict to children for weighting if needed
    fe.send_dict_to_children({
        "total_compute_ops_per_second": combined_total,
        "scalar_flops_per_second": scalar_flops_per_sec,
        "tensor_ops_per_second": tensor_ops_per_sec,
    })
