# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import NvRules
from RequestedMetrics import Importance, MetricRequest, RequestedMetricsParser
from TableBuilder import OpcodeTableBuilder

requested_metrics = [  # metrics used to compute peak and achieved FP work
    MetricRequest("device__attribute_compute_capability_major", "cc_major"),  # device CC major
    MetricRequest("device__attribute_compute_capability_minor", "cc_minor"),  # device CC minor
    # This is currently collected in "SourceCounters" and "InstructionStatistics"
    # sections, do not warn if it is not available (as with the basic set).
    MetricRequest("inst_executed", None, Importance.OPTIONAL, None, False),  # optional executed instruction counter
    MetricRequest("sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained", "inst_executed_ffma_peak"),  # theoretical FFMA peak per-thread
    MetricRequest("sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained", "inst_executed_dfma_peak"),  # theoretical DFMA peak per-thread
    MetricRequest("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed", "inst_executed_fadd"),  # achieved FADD per-cycle
    MetricRequest("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed", "inst_executed_fmul"),  # achieved FMUL per-cycle
    MetricRequest("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed", "inst_executed_ffma"),  # achieved FFMA per-cycle
    MetricRequest("smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed", "inst_executed_dadd"),  # achieved DADD per-cycle
    MetricRequest("smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed", "inst_executed_dmul"),  # achieved DMUL per-cycle
    MetricRequest("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed", "inst_executed_dfma"),  # achieved DFMA per-cycle
    MetricRequest("launch__uses_green_context", "is_green_context", Importance.OPTIONAL, False, False),  # optional: green context flag
]

requested_metrics_gb10x = [  # extra metrics for GB10x arch variants
    MetricRequest("smsp__sass_thread_inst_executed_op_fadd2_pred_on.sum.per_cycle_elapsed", "inst_executed_fadd2"),  # FADD2 achieved per-cycle
    MetricRequest("smsp__sass_thread_inst_executed_op_fmul2_pred_on.sum.per_cycle_elapsed", "inst_executed_fmul2"),  # FMUL2 achieved per-cycle
    MetricRequest("smsp__sass_thread_inst_executed_op_ffma2_pred_on.sum.per_cycle_elapsed", "inst_executed_ffma2"),  # FFMA2 achieved per-cycle
]


def get_identifier():
    return "SOLFPRoofline"

def get_name():
    return "Roofline Analysis"

def get_description():
    return "Floating Point Roofline Analysis"

def get_section_identifier():
    return "SpeedOfLight_RooflineChart"

def get_parent_rules_identifiers():
    return ["HighPipeUtilization"]

def get_estimated_speedup(parent_weights, achieved_fp32, achieved_fp64, peak_fp32, peak_fp64):
    # Estimate the speedup achievable by replacing FP64 with FP32 where possible.
    # If peak FP64 per-thread exceeds FP32, no improvement is possible.
    if peak_fp64 / peak_fp32 > 1:
        return NvRules.IFrontend.SpeedupType_LOCAL, 0  # no local improvement

    improvement_local = (achieved_fp64 / (achieved_fp32 + achieved_fp64)) * (
        1 - peak_fp64 / peak_fp32
    )  # fraction of workload that's FP64 * relative peak difference

    if "fp64_pipeline_utilization_pct" in parent_weights:
        speedup_type = NvRules.IFrontend.SpeedupType_GLOBAL  # weighted global estimate
        improvement_percent = improvement_local * parent_weights["fp64_pipeline_utilization_pct"]  # apply utilization weight
    else:
        speedup_type = NvRules.IFrontend.SpeedupType_LOCAL  # local-only estimate
        improvement_percent = improvement_local * 100  # convert to percent

    return speedup_type, improvement_percent  # return type and estimated percent


def add_fp64_instructions_table_and_source_markers(
    message_id,
    frontend,
    action,
    metrics,
):
    if metrics["inst_executed"] is None:
        return  # nothing to correlate to source if inst_executed missing

    table_builder = OpcodeTableBuilder(
        workload=action,  # workload context
        instruction_metric=metrics["inst_executed"],  # instruction metric to correlate
        opcodes=["DADD", "DMUL", "DFMA"],  # FP64 opcodes of interest
    )
    header, data, config = table_builder.build(
        title="Most frequently executed FP64 instructions",  # table title
        description=(
            "Source lines with the highest number of executed"
            " 64-bit floating point instructions."
        ),  # table description
    )

    if len(data) == 0:
        return  # nothing to display

    frontend.generate_table(message_id, header, data, config)  # emit table to frontend

    source_marker_advice = (
        "This line executes many 64-bit floating-point instructions."
        " Consider converting them to their 32-bit equivalents"
        " to improve performance."
    )  # advice for source markers
    for aggregate in table_builder.get_aggregates():
        frontend.source_marker(
            source_marker_advice,  # advice text
            aggregate.source_location.line,  # source line number
            NvRules.MarkerKind.SOURCE,  # marker kind
            aggregate.source_location.path,  # file path
            NvRules.MsgType.OPTIMIZATION,  # message type
        )


def apply(handle):
    ctx = NvRules.get_context(handle)  # NvRules context
    action = ctx.range_by_idx(0).action_by_idx(0)  # current action/workload
    fe = ctx.frontend()  # frontend helpers
    metrics = RequestedMetricsParser(handle, action).parse(requested_metrics)  # retrieve needed metrics
    parent_weights = fe.receive_dict_from_parent("HighPipeUtilization")  # weights sent by parent rule

    peak_fp32 = 2 * metrics["inst_executed_ffma_peak"].value()  # FP32 peak: 2 ops per FFMA
    peak_fp64 = 2 * metrics["inst_executed_dfma_peak"].value()  # FP64 peak: 2 ops per DFMA

    fp32_add_achieved = metrics["inst_executed_fadd"].value()  # achieved FADD per-cycle
    fp32_mul_achieved = metrics["inst_executed_fmul"].value()  # achieved FMUL per-cycle
    fp32_fma_achieved = metrics["inst_executed_ffma"].value()  # achieved FFMA per-cycle
    achieved_fp32 = fp32_add_achieved + fp32_mul_achieved + 2 * fp32_fma_achieved  # total achieved FP32 ops per-cycle

    cc_major = metrics["cc_major"].value()  # compute capability major
    cc_minor = metrics["cc_minor"].value()  # compute capability minor

    if cc_major == 10 and (cc_minor == 0 or cc_minor == 3):
        metrics_gb10x = RequestedMetricsParser(handle, action).parse(requested_metrics_gb10x)  # parse GB10x extras
        fp32_add2_achieved = metrics_gb10x["inst_executed_fadd2"].value()  # achieved FADD2
        fp32_mul2_achieved = metrics_gb10x["inst_executed_fmul2"].value()  # achieved FMUL2
        fp32_fma2_achieved = metrics_gb10x["inst_executed_ffma2"].value()  # achieved FFMA2
        achieved_fp32 += fp32_add2_achieved *2 + fp32_mul2_achieved + 2 * fp32_fma2_achieved * 4  # include multi-op contributions

    fp64_add_achieved = metrics["inst_executed_dadd"].value()  # achieved DADD per-cycle
    fp64_mul_achieved = metrics["inst_executed_dmul"].value()  # achieved DMUL per-cycle
    fp64_fma_achieved = metrics["inst_executed_dfma"].value()  # achieved DFMA per-cycle
    achieved_fp64 = fp64_add_achieved + fp64_mul_achieved + 2 * fp64_fma_achieved  # total achieved FP64 ops per-cycle

    high_utilization_threshold = 0.60  # thresholds used for messaging
    low_utilization_threshold = 0.15

    resource_partition = "device"  # descriptor used in messages
    if metrics["is_green_context"].value():
        resource_partition = "green context"  # adjust descriptor if needed

    achieved_fp64_pct = achieved_fp64 / peak_fp64  # fraction of FP64 peak achieved
    fp64_prefix = "" if achieved_fp64_pct >= 0.01 or achieved_fp64_pct == 0.0 else " close to "  # wording
    achieved_fp32_pct = achieved_fp32 / peak_fp32  # fraction of FP32 peak achieved
    fp32_prefix = "" if achieved_fp32_pct >= 0.01 or achieved_fp32_pct == 0.0 else " close to "  # wording

    message = "The ratio of peak float (FP32) to double (FP64) performance on this device is {:.0f}:1.".format(peak_fp32 / peak_fp64)  # base summary
    message += " The workload achieved {}{:.0f}% of this {}'s FP32 peak performance and {}{:.0f}% of its FP64 peak performance.".format(fp32_prefix, 100.0 * achieved_fp32_pct, resource_partition, fp64_prefix, 100.0 * achieved_fp64_pct)  # add stats

    message_profiling_guide = " See the @url:Profiling Guide:https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline@ for more details on roofline analysis."  # link to docs

    if achieved_fp32_pct < high_utilization_threshold and achieved_fp64_pct > low_utilization_threshold:
        message += " If @section:ComputeWorkloadAnalysis:Compute Workload Analysis@ determines that this workload is FP64 bound, consider using 32-bit precision floating point operations to improve its performance."  # suggestion to lower precision
        message += message_profiling_guide
        msg_id = fe.message(NvRules.MsgType.OPTIMIZATION, message, "FP64/32 Utilization")  # emit optimization message

        speedup_type, speedup_value = get_estimated_speedup(parent_weights, achieved_fp32, achieved_fp64, peak_fp32, peak_fp64)  # compute estimated speedup
        fe.speedup(msg_id, speedup_type, speedup_value)  # attach speedup

        if speedup_value > 0:
            fe.focus_metric(msg_id, metrics["inst_executed_dadd"].name(), fp64_add_achieved, NvRules.IFrontend.Severity_SEVERITY_HIGH, "Decrease FP64 ADD instructions")  # focus on DADD
            fe.focus_metric(msg_id, metrics["inst_executed_dmul"].name(), fp64_mul_achieved, NvRules.IFrontend.Severity_SEVERITY_HIGH, "Decrease FP64 MUL instructions")  # focus on DMUL
            fe.focus_metric(msg_id, metrics["inst_executed_dfma"].name(), fp64_fma_achieved, NvRules.IFrontend.Severity_SEVERITY_HIGH, "Decrease FP64 FMA instructions")  # focus on DFMA

        add_fp64_instructions_table_and_source_markers(msg_id, fe, action, metrics)  # emit table and markers

    elif achieved_fp64_pct > high_utilization_threshold and achieved_fp32_pct > high_utilization_threshold:
        message += " If @section:SpeedOfLight:Speed Of Light@ analysis determines that this workload is compute bound, consider using integer arithmetic instead where applicable."  # generic suggestion
        message += message_profiling_guide
        msg_id = fe.message(NvRules.MsgType.OPTIMIZATION, message, "High FP Utilization")  # optimization message
    else:
        message += message_profiling_guide
        msg_id = fe.message(NvRules.MsgType.OK, message, "Roofline Analysis")  # OK status
