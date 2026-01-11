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
import NvRules  # NvRules API for rule/runtime interactions
from RequestedMetrics import Importance, MetricRequest, RequestedMetricsParser  # helpers to request metrics
from TableBuilder import OpcodeTableBuilder  # helper to build opcode-based tables

requested_metrics = [
    # This is currently collected in "SourceCounters" and "InstructionStatistics"
    # sections, warn if it is not available.
    MetricRequest("inst_executed", None, Importance.OPTIONAL, None, True),  # total executed instructions (optional)
    MetricRequest("sass__inst_executed_per_opcode", None, Importance.OPTIONAL, None, False),  # per-opcode executed counts (optional)
]


def get_identifier():
    return "FPInstructions"  # unique rule identifier

def get_name():
    return "FP32/64 Instructions"  # human-friendly rule name

def get_description():
    return "Floating-point instruction analysis."  # short description

def get_section_identifier():
    return "InstructionStats"  # UI section where results appear

def get_parent_rules_identifiers():
    return ["HighPipeUtilization"]  # parent rule for weighting

def get_estimated_speedup(pipeline_utilization_pct, fused_instructions, non_fused_instructions):
    # To calculate the speedup, assume we can convert non-fused to fused instructions,
    # which have double the throughput.
    # To get a global estimate weigh this with the FP pipeline utilization
    # (in terms of active cycles).
    all_instructions = non_fused_instructions + fused_instructions  # total FP-related instructions
    improvement_local = 0.5 * (non_fused_instructions / all_instructions)  # local relative improvement if all non-fused were fused

    if pipeline_utilization_pct is not None:
        speedup_type = NvRules.IFrontend.SpeedupType_GLOBAL  # report global speedup when we can weight it
        improvement_percent = improvement_local * pipeline_utilization_pct  # weight local improvement by utilization
    else:
        speedup_type = NvRules.IFrontend.SpeedupType_LOCAL  # fallback to local estimate
        improvement_percent = improvement_local * 100  # express as percent

    return speedup_type, improvement_percent  # return type and estimated percent


def add_non_fused_instructions_table_and_source_markers(
        message_id,
        frontend,
        action,
        metrics,
        fp_type,
):
    if metrics["inst_executed"] is None:
        return  # nothing to do if instruction count missing

    non_fused_opcodes = {
        32: ["FADD", "FMUL"],  # opcodes considered non-fused for FP32
        64: ["DADD", "DMUL"],  # opcodes considered non-fused for FP64
    }

    table_builder = OpcodeTableBuilder(
        workload=action,  # action/workload context
        instruction_metric=metrics["inst_executed"],  # metric used to correlate source lines
        opcodes=non_fused_opcodes[fp_type],  # opcodes of interest for this precision
    )
    header, data, config = table_builder.build(
        title=f"Most frequently executed non-fused FP{fp_type} instructions",  # table title
        description=(
            "Source lines with the highest number of executed"
            f" non-fused {fp_type}-bit floating point instructions."
        ),  # table description
    )

    if len(data) == 0:
        return  # no rows to report

    frontend.generate_table(message_id, header, data, config)  # emit the table to the frontend

    source_marker_advice = (
        "This line executes many non-fused floating-point instructions."
        " To improve performance, consider converting pairs of non-fused"
        " instructions to FMA instructions, and to enable NVCC's --use_fast_math"
        " or --fmad=true compiler flags."
    )  # actionable advice for source markers
    for aggregate in table_builder.get_aggregates():
        frontend.source_marker(
            source_marker_advice,
            aggregate.source_location.line,
            NvRules.MarkerKind.SOURCE,
            aggregate.source_location.path,
            NvRules.MsgType.OPTIMIZATION,
        )


def apply(handle):
    ctx = NvRules.get_context(handle)  # get the NvRules context
    action = ctx.range_by_idx(0).action_by_idx(0)  # get the current action (workload)
    fe = ctx.frontend()  # frontend interface

    metrics = RequestedMetricsParser(handle, action).parse(requested_metrics)
    if any(metric is None for metric in metrics.values()):
        # Not all SASS metrics are available for all supported workload/profile mode
        # combinations, so we skip the rule if any of them are missing.
        return

    parent_weights = fe.receive_dict_from_parent("HighPipeUtilization")

    fp_types = {
        32 : [ "FADD", "FMUL", "FFMA" ],
        64 : [ "DADD", "DMUL", "DFMA" ]
    }

    # the correlation IDs of sass__inst_executed_per_opcode are the opcode mnemonics
    inst_per_opcode = metrics["sass__inst_executed_per_opcode"]
    num_opcodes = inst_per_opcode.num_instances()
    opcodes = inst_per_opcode.correlation_ids()

    # analyze both 32 and 64 bit
    for fp_type in fp_types:
        fp_insts = dict()
        fp_opcodes = fp_types[fp_type]
        # get number of instructions by opcode
        for i in range(0,num_opcodes):
            op = opcodes.as_string(i).upper()
            if op in fp_opcodes:
                fp_insts[op] = inst_per_opcode.as_uint64(i)

        # calculate the sum of low- and high-throughput instructions
        non_fused = 0
        for i in range(0, 2):
            op = fp_opcodes[i]
            if op in fp_insts:
                non_fused += fp_insts[op]

        fused = 0
        op = fp_opcodes[2]
        if op in fp_insts:
            fused += fp_insts[op]

        if non_fused > 0 or fused > 0:
            # high-throughput/fused instructions have twice the throughput of non-fused ones
            ratio = (non_fused / (non_fused + fused)) / 2
            if ratio > 0.1:
                message = "This workload executes {} fused and {} non-fused FP{} instructions.".format(fused, non_fused, fp_type)
                message += " By converting pairs of non-fused instructions to their @url:fused:https://docs.nvidia.com/cuda/floating-point/#cuda-and-floating-point@, higher-throughput equivalent, the achieved FP{} performance could be increased by up to {:.0f}%"\
                    " (relative to its current performance).".format(fp_type, 100. * ratio)
                message_title = "FP{} Non-Fused Instructions".format(fp_type)
                msg_id = fe.message(NvRules.MsgType.OPTIMIZATION, message, message_title)

                pipeline_utilization_pct = None
                parent_weight_name = "fp{}_pipeline_utilization_pct".format(fp_type)
                if parent_weight_name in parent_weights:
                    pipeline_utilization_pct = parent_weights[parent_weight_name]
                speedup_type, speedup_value = get_estimated_speedup(pipeline_utilization_pct, fused, non_fused)
                fe.speedup(msg_id, speedup_type, speedup_value)

                add_non_fused_instructions_table_and_source_markers(msg_id, fe, action, metrics, fp_type)

                fe.focus_metric(msg_id, "sass__inst_executed_per_opcode", non_fused, NvRules.IFrontend.Severity_SEVERITY_HIGH, "Decrease the number of non-fused floating-point instructions (FADD, FMUL, DADD, DMUL)")
                if pipeline_utilization_pct is not None:
                    if fp_type == 32:
                        metric_name = "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"
                    else:
                        metric_name = "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active"
                    fe.focus_metric(msg_id, metric_name, pipeline_utilization_pct, NvRules.IFrontend.Severity_SEVERITY_LOW, "The higher the utilization of the pipeline the more severe the issue becomes")
