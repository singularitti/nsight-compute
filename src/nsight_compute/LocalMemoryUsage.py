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


def get_identifier():
    return "LocalMemoryUsage"


def get_name():
    return "Local Memory Usage"


def get_description():
    return "Detects local memory usage and register spilling and estimates its impact on performance."


def get_section_identifier():
    return "MemoryWorkloadAnalysis_Tables"


requested_metrics = [
    # L1 metrics
    # --- Sectors requested [sector]
    MetricRequest(
        "l1tex__t_sectors.sum", "l1_sectors", Importance.OPTIONAL, None, False
    ),
    MetricRequest("l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum", "l1_sectors_loads"),
    MetricRequest("l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum", "l1_sectors_stores"),
    # --- Hit rates [%]
    MetricRequest(
        "l1tex__t_sector_pipe_lsu_mem_local_op_ld_hit_rate.pct",
        "l1_sectors_loads_hit_pct",
    ),
    MetricRequest(
        "l1tex__t_sector_pipe_lsu_mem_local_op_st_hit_rate.pct",
        "l1_sectors_stores_hit_pct",
    ),
    # --- Lookup misses [sector]
    #     Estimate L2 sectors that are due to local memory accesses by
    #     the sectors not cached in L1TEX.
    MetricRequest(
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_miss.sum",
        "l2_sectors_loads",
    ),
    MetricRequest(
        "l1tex__t_sectors_pipe_lsu_mem_local_op_st_lookup_miss.sum",
        "l2_sectors_stores",
    ),
    # L2 metrics
    # --- Sectors requested [sector]
    MetricRequest("lts__t_sectors.sum", "l2_sectors"),
    # Memory bandwidth metrics [sector/s]
    MetricRequest(
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum.per_second",
        "l1_throughput_loads",
        Importance.OPTIONAL,
        None,
        False,
    ),
    MetricRequest(
        "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum.per_second",
        "l1_throughput_stores",
        Importance.OPTIONAL,
        None,
        False,
    ),
    MetricRequest(
        "lts__t_sectors_op_read.sum.per_second",
        "l2_throughput_loads",
        Importance.OPTIONAL,
        None,
        False,
    ),
    MetricRequest(
        "lts__t_sectors_op_write.sum.per_second",
        "l2_throughput_stores",
        Importance.OPTIONAL,
        None,
        False,
    ),
    # Executed instructions [inst]
    MetricRequest("inst_executed", "inst_executed", Importance.OPTIONAL, None, True),
    MetricRequest(
        "smsp__sass_inst_executed_op_local_ld.sum",
        "inst_loads",
        Importance.OPTIONAL,
        None,
        True,
    ),
    MetricRequest(
        "smsp__sass_inst_executed_op_local_st.sum",
        "inst_stores",
        Importance.OPTIONAL,
        None,
        True,
    ),
    # --- NOTE: The following metrics are only available through the SourceCounters
    # section and cannot be specified with --metrics. Hence, do not warn when missing.
    MetricRequest(
        "sass__inst_executed_register_spilling_mem_local",
        "inst_spills",
        Importance.OPTIONAL,
        None,
        False,
    ),
    MetricRequest(
        "sass__inst_executed_register_spilling_mem_local_op_read",
        "inst_spills_loads",
        Importance.OPTIONAL,
        None,
        False,
    ),
    MetricRequest(
        "sass__inst_executed_register_spilling_mem_local_op_write",
        "inst_spills_stores",
        Importance.OPTIONAL,
        None,
        False,
    ),
    # Kernel duration [ns]
    MetricRequest("gpu__time_duration.sum", "kernel_duration"),
]


def link_local_memory(text):
    return f"@url:{text}:https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses@"


def link_register_spilling(text):
    return f"@url:{text}:https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#variable-memory-space-specifiers@"


def calculate_estimated_speedup(metrics):
    """Estimate global speedup based on eliminating local memory traffic and instructions.

    Reducing local memory usage can lead to

    1. Reduced memory traffic to L1 and L2 caches.
    2. Reduced instruction count for LDL and STL instructions.

    We calculate these two speedups independently and take the maximum of the two
    to get an overall speedup estimate. All speedups can be calculated via

        speedup = (time_old - time_new) / time_old * 100
                = delta_time / time_old * 100 [%]

    Memory Traffic:

    To get a (very rough) estimate of the reduction in time one can achieve by
    eliminating local memory accesses, let's assume _all_ local memory accesses
    to L1 and L2 can be converted to register accesses. Let's further assume that
    register throughput is infinite (compared with L1 and L2).

    Then, for each cache level (L1 and L2, do not consider DRAM) and operation
    (loads and stores) we can estimate delta_t as:

        delta_t = sectors / throughput, [sector / (sector/s) = s]

    Instruction Count:

    To get a (equally rough) estimate for the speedup through the reduction
    of the instruction count, let's assume that all local memory instructions
    can be eliminated. Let's further assume that all instructions take roughly
    the same time and all non-local memory instructions are unaffected.

    Then, we can estimate the speedup as:

        speedup = (inst_old - inst_new) / inst_old * 100
                = inst_lmem / inst_executed * 100
    """

    # 1. Reducing memory traffic
    delta_time = 0
    for cache in ["l1", "l2"]:
        for operation in ["loads", "stores"]:
            sectors = metrics[f"{cache}_sectors_{operation}"].value()
            throughput = metrics[f"{cache}_throughput_{operation}"]  # [sector/s]
            if throughput and throughput.value() > 0:
                delta_time += sectors / throughput.value()

    relative_speedup_memory = 0
    time_old = metrics["kernel_duration"].value() * 10e-9  # [ns] -> [s]
    if time_old > 0:
        relative_speedup_memory = delta_time / time_old

    # 2. Reducing instruction count
    relative_speedup_instruction_count = 0
    inst_executed = metrics["inst_executed"]
    inst_loads = metrics["inst_loads"]
    inst_stores = metrics["inst_stores"]

    if inst_loads and inst_stores and inst_executed and inst_executed.value() > 0:
        relative_speedup_instruction_count = (
            inst_loads.value() + inst_stores.value()
        ) / inst_executed.value()

    # 3. Calculate the maximum speedup and convert to percentage
    improvement_percent = (
        max(
            relative_speedup_memory,
            relative_speedup_instruction_count,
        )
        * 100
    )

    return NvRules.IFrontend.SpeedupType_GLOBAL, improvement_percent


def add_focus_metrics(
    message_id,
    frontend,
    metrics,
):
    # 1. Register spilling
    inst_spills = metrics["inst_spills"]
    if inst_spills and inst_spills.value():
        frontend.focus_metric(
            message_id,
            inst_spills.name(),
            inst_spills.value(),
            NvRules.IFrontend.Severity_SEVERITY_HIGH,
            "Use the SASS view of the Source Page to check whether there are any unexpected"
            " register spills.",
        )

    # 2. Any local memory instructions
    inst_loads = metrics["inst_loads"]
    inst_stores = metrics["inst_stores"]
    if inst_loads and inst_loads.value():
        frontend.focus_metric(
            message_id,
            inst_loads.name(),
            inst_loads.value(),
            NvRules.IFrontend.Severity_SEVERITY_HIGH,
            "Decrease the number of local memory load instructions (LDL).",
        )
    if inst_stores and inst_stores.value():
        frontend.focus_metric(
            message_id,
            inst_stores.name(),
            inst_stores.value(),
            NvRules.IFrontend.Severity_SEVERITY_HIGH,
            "Decrease the number of local memory store instructions (STL).",
        )

    # 3. Caching local memory in L1TEX
    frontend.focus_metric(
        message_id,
        metrics["l1_sectors_loads_hit_pct"].name(),
        metrics["l1_sectors_loads_hit_pct"].value(),
        NvRules.IFrontend.Severity_SEVERITY_LOW,
        "Increase the L1TEX read hit rate, if local memory loads cannot be avoided.",
    )
    frontend.focus_metric(
        message_id,
        metrics["l1_sectors_stores_hit_pct"].name(),
        metrics["l1_sectors_stores_hit_pct"].value(),
        NvRules.IFrontend.Severity_SEVERITY_LOW,
        "Increase the L1TEX write hit rate, if local memory stores cannot be avoided.",
    )


def add_instructions_table_and_source_markers(
    message_id,
    frontend,
    action,
    metrics,
):
    if not metrics["inst_executed"]:
        return

    local_memory_opcodes = ["LDL", "STL"]

    table_builder = OpcodeTableBuilder(
        workload=action,
        instruction_metric=metrics["inst_executed"],
        opcodes=local_memory_opcodes,
    )
    header, data, config = table_builder.build(
        title="Most frequently executed local memory instructions (loads/stores)",
        description=(
            "Source lines with the highest number of executed"
            " local memory instructions."
        ),
        top_n=5,
    )

    if len(data) == 0:
        return

    frontend.generate_table(message_id, header, data, config)

    source_marker_advice = (
        "This line executes many local memory instructions."
        " Check the correlated SASS to see whether any of"
        " these might be due to register spilling."
    )
    for aggregate in table_builder.get_aggregates():
        frontend.source_marker(
            source_marker_advice,
            aggregate.source_location.line,
            NvRules.MarkerKind.SOURCE,
            aggregate.source_location.path,
            NvRules.MsgType.OPTIMIZATION,
        )


def add_speedup_estimates(
    message_id,
    frontend,
    metrics,
):
    speedup_type, speedup_value = calculate_estimated_speedup(metrics)
    frontend.speedup(message_id, speedup_type, speedup_value)


def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    parser = RequestedMetricsParser(handle, action)
    metrics = parser.parse(requested_metrics)

    # 1. Detect if there are any local memory accesses and register spilling
    # --- Check if local memory is used
    if (
        metrics["l1_sectors_loads"].value() == 0
        and metrics["l1_sectors_stores"].value() == 0
    ):
        return

    message_paragraphs = []

    paragraph1 = f"This workload accesses {link_local_memory('local memory')}"

    if metrics["l1_sectors"] and metrics["l1_sectors"].value() > 0:
        l1_sectors_local_pct = (
            (metrics["l1_sectors_loads"].value() + metrics["l1_sectors_stores"].value())
            / metrics["l1_sectors"].value()
            * 100
        )
        paragraph1 += f", accounting for {l1_sectors_local_pct:.2f}% of all sectors requested in L1TEX."
    else:
        paragraph1 += "."

    paragraph1 += (
        " Local memory refers to a memory space that is private to each thread."
        " Although it is logically distinct, local memory resides in global memory."
        " This can lead to higher memory latency compared to the"
        " usage of registers or shared memory, a decrease in available bandwidth"
        " to global memory, and an increase in executed instructions."
    )
    message_paragraphs.append(paragraph1)

    # --- Check if local memory usage is due to register spilling
    paragraph2 = ""
    inst_local = 0
    if (
        metrics["inst_loads"]
        and metrics["inst_stores"]
        and metrics["inst_spills"]
        and metrics["inst_spills"].value()
    ):
        inst_loads = metrics["inst_loads"].value()
        inst_stores = metrics["inst_stores"].value()
        inst_local = inst_loads + inst_stores
        inst_spills = metrics["inst_spills"].value()

        if inst_spills == inst_local:
            paragraph2 += (
                " All of the local memory accesses are due to"
                f" {link_register_spilling('register spilling')}."
            )
        else:
            paragraph2 += (
                " Some of the local memory accesses are due to"
                f" {link_register_spilling('register spilling')}"
            )

            # NOTE: We may have collected inst_spills, but not inst_spills_loads
            # and inst_spills_stores individually.
            if metrics["inst_spills_loads"] and metrics["inst_spills_stores"]:
                inst_spills_loads = metrics["inst_spills_loads"].value()
                inst_spills_stores = metrics["inst_spills_stores"].value()

                inst_spills_loads_pct = (
                    inst_spills_loads / inst_loads * 100 if inst_loads > 0 else 0
                )
                inst_spills_stores_pct = (
                    inst_spills_stores / inst_stores * 100 if inst_stores > 0 else 0
                )
                paragraph2 += (
                    f", accounting for {inst_spills_loads_pct:.2f}% of the"
                    " local memory load (LDL) and"
                    f" {inst_spills_stores_pct:.2f}% of the local store (STL) instructions."
                )
            else:
                inst_spills_pct = inst_spills / inst_local * 100
                paragraph2 += (
                    f", accounting for {inst_spills_pct:.2f}% of all local memory"
                    " instructions (LDL and STL)."
                )

    # 2. Detect if local memory usage impacts memory traffic and/or executed instructions
    # --- Detect impact on memory traffic
    threshold_l1_loads_hit_pct = 80
    threshold_l1_stores_hit_pct = 80
    l1_sectors_loads_hit_pct = metrics["l1_sectors_loads_hit_pct"].value()
    l1_sectors_stores_hit_pct = metrics["l1_sectors_stores_hit_pct"].value()

    if (
        l1_sectors_loads_hit_pct > threshold_l1_loads_hit_pct
        and l1_sectors_stores_hit_pct > threshold_l1_stores_hit_pct
    ):
        paragraph2 += (
            " The local memory accesses of this workload are largely cached in L1TEX"
            f" ({l1_sectors_loads_hit_pct:.2f}% of loads,"
            f" and {l1_sectors_stores_hit_pct:.2f}% of stores)."
            " However, even spilling to L1TEX might lead to high performance"
            " penalties compared with staying in registers."
        )
    else:
        l2_sectors_local_pct = (
            (metrics["l2_sectors_loads"].value() + metrics["l2_sectors_stores"].value())
            / metrics["l2_sectors"].value()
            * 100
        )
        paragraph2 += (
            f" {100-l1_sectors_loads_hit_pct:.2f}% of all local loads and"
            f" {100-l1_sectors_stores_hit_pct:.2f}% of all local stores spill"
            " into the L2 cache, accounting for"
            f" {l2_sectors_local_pct:.2f}% of all sectors requested in L2."
        )

    # --- Detect impact on executed instruction count
    threshold_inst_executed_pct = 10
    inst_local_pct = 0

    if metrics["inst_executed"] and metrics["inst_executed"].value() > 0:
        inst_local_pct = inst_local / metrics["inst_executed"].value() * 100

    if inst_local_pct > threshold_inst_executed_pct:
        paragraph2 += (
            " The local memory accesses of this workload amount to"
            f" {inst_local_pct:.2f}% of all executed instructions."
        )
    message_paragraphs.append(paragraph2.strip())

    # 3. Give some optimization advice
    paragraph3 = (
        "One common reason for the use of local memory is arrays being too large to fit"
        " into registers or not being indexed using compile time constants"
        " (see the \"Local Memory\" section of the"
        " @url:CUDA Programming Guide:https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses@."
        " For more advice on how to reduce the usage of local memory refer to the"
        " @url:documentation:https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-memory@."
    )
    message_paragraphs.append(paragraph3.strip())

    # 4. Issue rule result including focus metrics, instruction table, source markers
    # and speedup estimates
    rule_title = "Local Memory Usage"
    msg_type = NvRules.MsgType.OPTIMIZATION
    msg_id = fe.message(msg_type, "\n".join(message_paragraphs), rule_title)

    add_focus_metrics(msg_id, fe, metrics)
    add_instructions_table_and_source_markers(msg_id, fe, action, metrics)
    add_speedup_estimates(msg_id, fe, metrics)
