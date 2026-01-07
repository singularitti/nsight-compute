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

requested_metrics = [
    MetricRequest("launch__execution_model", "launch_execution_model", Importance.OPTIONAL, None, True),
    MetricRequest("launch__user_grid_size", "user_grid_size", Importance.OPTIONAL, None, False),
    MetricRequest("launch__grid_size", "grid_size", Importance.OPTIONAL, None, False),
    MetricRequest("launch__cluster_size", "cluster_size", Importance.OPTIONAL, None, False),
    MetricRequest("sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_elapsed", "tma_utilization", Importance.OPTIONAL, None, False),
    MetricRequest("sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed", "tensor_core_pipe_utilization", Importance.OPTIONAL, None, False),
]

def get_identifier():
    return "TileAnalysis"


def get_name():
    return "Tile Analysis"


def get_description():
    return "Analyze Tile execution and provide optimization recommendations"


def get_section_identifier():
    return "Tile"


def get_parent_rules_identifiers():
    return []


def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    parser = RequestedMetricsParser(handle, action)
    metrics = parser.parse(requested_metrics)

    execution_model_metric = metrics["launch_execution_model"]
    # Check if this is actually a Tile kernel
    if execution_model_metric is None or execution_model_metric.value() == "SIMT":
        return

    user_grid_size_metric = metrics["user_grid_size"]
    grid_size_metric = metrics["grid_size"]

    if user_grid_size_metric is None or grid_size_metric is None:
        return
    user_grid_size = user_grid_size_metric.value()
    grid_size = grid_size_metric.value()

    cluster_size_metric = metrics["cluster_size"]
    tma_utilization_metric = metrics["tma_utilization"]
    tensor_core_pipe_utilization_metric = metrics["tensor_core_pipe_utilization"]

    # Analyze Tile kernel configuration
    if cluster_size_metric is not None and user_grid_size > 0 and grid_size > 0:
        thread_blocks_per_tile = grid_size / user_grid_size

        name = "Tile Mapping"
        if thread_blocks_per_tile == 1 and cluster_size_metric.value() > 1:
            msg_type = NvRules.MsgType.OK
            message = "Multiple thread blocks ({:d}) are clustered together to form a CGA and mapped to a tile block to execute tile block's instructions.".format(cluster_size_metric.value())
            msg_id = fe.message(msg_type, message, name)

        elif thread_blocks_per_tile == 1:
            msg_type = NvRules.MsgType.OK
            message = "One thread block is mapped to execute one tile block instructions."
            msg_id = fe.message(msg_type, message, name)

        else:
            msg_type = NvRules.MsgType.OPTIMIZATION
            message = "Each tile block is mapped to {:.1f} thread blocks. This indicates inefficient resource utilization. Consider increasing tile block size for better performance.".format(thread_blocks_per_tile)
            msg_id = fe.message(msg_type, message, name)
            fe.focus_metric(msg_id, metrics["user_grid_size"].name(), user_grid_size, NvRules.IFrontend.Severity_SEVERITY_HIGH, "#Tile Blocks Launched")
            fe.focus_metric(msg_id, metrics["grid_size"].name(), grid_size, NvRules.IFrontend.Severity_SEVERITY_HIGH, "#Thread Blocks Launched")

    # Analyze GPU resource utilization for Tile kernels
    if tensor_core_pipe_utilization_metric is not None and tensor_core_pipe_utilization_metric.value() == 0:
        msg_type = NvRules.MsgType.OPTIMIZATION
        message = "Tensor core pipe utilization is 0%. Consider changing the tile size or using tensor core compatible data types for matmul or MMA to better utilize tensor core resources."
        name = "Tensor Core Unused"
        msg_id = fe.message(msg_type, message, name)
        fe.focus_metric(msg_id, tensor_core_pipe_utilization_metric.name(), tensor_core_pipe_utilization_metric.value(), NvRules.IFrontend.Severity_SEVERITY_HIGH, "Tensor Core Pipe Utilization")

    elif tma_utilization_metric is not None and tma_utilization_metric.value() == 0:
        msg_type = NvRules.MsgType.OPTIMIZATION
        message = "Tensor memory accelerator is not used. Consider changing tile size to better utilize tensor memory accelerator."
        name = "Tensor Memory Accelerator Unused"
        msg_id = fe.message(msg_type, message, name)
        fe.focus_metric(msg_id, tma_utilization_metric.name(), tma_utilization_metric.value(), NvRules.IFrontend.Severity_SEVERITY_HIGH, "TMA Utilization")


