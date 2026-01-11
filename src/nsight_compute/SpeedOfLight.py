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
import math

import NvRules
from RequestedMetrics import Importance, MetricRequest, RequestedMetricsParser

requested_metrics = [  # metrics driving the SOL (Speed of Light) analysis
    MetricRequest("sm__throughput.avg.pct_of_peak_sustained_elapsed", "sm_sol_pct"),  # SM compute throughput % of peak
    MetricRequest("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "mem_sol_pct", Importance.OPTIONAL, None, False),  # memory throughput % (optional)
    MetricRequest("breakdown:gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "breakdown_memory", Importance.OPTIONAL, None, False),  # memory breakdown metric (optional)
    MetricRequest("launch__waves_per_multiprocessor", "num_waves", Importance.OPTIONAL, None, False),  # waves per SM (optional)
    MetricRequest("launch__uses_nvlink_centric_scheduling", "nvlink_scheduling", Importance.OPTIONAL, None, False),  # NVLink scheduling flag (optional)
    MetricRequest("launch__uses_green_context", "is_green_context", Importance.OPTIONAL, False, False),  # green context flag (optional)
]


def get_identifier():
    return "SOLBottleneck"

def get_name():
    return "Bottleneck"

def get_description():
    return "High-level bottleneck detection"

def get_section_identifier():
    return "SpeedOfLight"

def get_max_pipe(breakdown_metrics):
    max_pipe = None  # name of the metric with maximum value
    max_pipe_value = 0  # value of the max metric

    for name, metric in breakdown_metrics.items():
        pipe_value = metric.value()  # get numeric value
        if pipe_value > max_pipe_value:
            max_pipe_value = pipe_value  # update best value
            max_pipe = name  # update best metric name

    if max_pipe:
        tokens = {
            "dram" : "DRAM",
            "l1tex" : "L1",
            "lts" : "L2",
            "ltc" : "L2",
            "fbp" : "DRAM",
            "fbpa" : "DRAM"
        }  # mapping token prefixes to human-readable names

        for token in tokens:
            if max_pipe.startswith(token):
                return tokens[token]  # return friendly name when a prefix matches

    return None  # no matching token found


def get_breakdown_metrics(parser, breakdown_metric):
    requests = []  # build MetricRequest list from comma-separated breakdown metric
    for metric_name in breakdown_metric.value().split(","):
        request = MetricRequest(metric_name)  # create a request per token
        requests.append(request)
    return parser.parse(requests)  # parse and return metric objects


def apply(handle):
    ctx = NvRules.get_context(handle)  # get NvRules context
    action = ctx.range_by_idx(0).action_by_idx(0)  # current action/workload
    fe = ctx.frontend()  # frontend helper

    parser = RequestedMetricsParser(handle, action)
    metrics = parser.parse(requested_metrics)  # request SOL-related metrics

    sm_sol_pct_metric = metrics["sm_sol_pct"]  # compute throughput metric object
    mem_sol_pct_metric = metrics["mem_sol_pct"]  # memory throughput metric object (may be optional)
    breakdown_memory_metric = metrics["breakdown_memory"]  # memory breakdown spec (optional)

    if mem_sol_pct_metric is None or breakdown_memory_metric is None:
        return  # cannot proceed without memory metrics

    breakdown_metrics_memory = get_breakdown_metrics(parser, breakdown_memory_metric)  # parse breakdown metrics

    sm_sol_pct_name = sm_sol_pct_metric.name()  # metric name for reporting
    mem_sol_pct_name = mem_sol_pct_metric.name()

    sm_sol_pct = sm_sol_pct_metric.value()  # achieved compute throughput % of peak
    mem_sol_pct = mem_sol_pct_metric.value()  # achieved memory throughput % of peak

    balanced_threshold = 10  # % threshold to call one side heavier
    latency_bound_threshold = 60  # % threshold below which we suspect latency
    no_bound_threshold = 80  # % threshold above which we say high throughput
    waves_threshold = 1  # threshold for small-grid detection

    msg_type = NvRules.MsgType.OK  # default message type
    resource_partition = "device"
    if metrics["is_green_context"].value():
        resource_partition = "green context"  # adjust messaging when profiling green contexts

    focus_metrics = []  # list of (name, value, severity, hint)

    if sm_sol_pct >= mem_sol_pct:
        bottleneck_section = "@section:ComputeWorkloadAnalysis:Compute Workload Analysis@"  # compute-focused guidance
    else:
        bottleneck_section = "@section:MemoryWorkloadAnalysis:Memory Workload Analysis@"  # memory-focused guidance

    if sm_sol_pct < no_bound_threshold and mem_sol_pct < no_bound_threshold:
        # Neither compute nor memory are near peak: could be latency or underutilization
        if sm_sol_pct < latency_bound_threshold and mem_sol_pct < latency_bound_threshold:
            msg_type = NvRules.MsgType.OPTIMIZATION  # suggest optimization actions
            num_waves_metric = metrics["num_waves"]  # check waves per SM
            if num_waves_metric and num_waves_metric.value() < waves_threshold:
                num_waves = num_waves_metric.value()
                focus_metrics.append((num_waves_metric.name(), num_waves, NvRules.IFrontend.Severity_SEVERITY_HIGH, "Increase the number of waves per SM towards {}".format(waves_threshold)))  # advise increasing waves
                if action.workload_type() == NvRules.IAction.WorkloadType_KERNEL:
                    message = "This kernel grid is too small to fill the available resources on this {}, resulting in only {:.2f} full waves across all SMs.".format(resource_partition, num_waves)  # kernel-specific message
                else:
                    # The aggregate value of num_waves is the max over all launches
                    message = "All launches of this workload use grids that are too small to fill the available resources on this {}, resulting in at most {:.2f} full waves across all SMs.".format(resource_partition, num_waves)  # workload-wide message
                message += " Look at @section:LaunchStats:Launch Statistics@ for more details."
                name = "Small Grid"
            else:
                focus_metrics.append((sm_sol_pct_name, sm_sol_pct, NvRules.IFrontend.Severity_SEVERITY_HIGH, "{:.3f} < {:.3f}".format(sm_sol_pct, no_bound_threshold)))  # point to compute metric
                focus_metrics.append((mem_sol_pct_name, mem_sol_pct, NvRules.IFrontend.Severity_SEVERITY_HIGH, "{:.3f} < {:.3f}".format(mem_sol_pct, no_bound_threshold)))  # point to memory metric
                message = "This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this {}. Achieved compute throughput and/or memory bandwidth below {:.1f}% of peak typically indicate latency issues. Look at @section:SchedulerStats:Scheduler Statistics@ and @section:WarpStateStats:Warp State Statistics@ for potential reasons.".format(resource_partition, latency_bound_threshold)
                name = "Latency Issue"
        elif math.fabs(sm_sol_pct - mem_sol_pct) >= balanced_threshold:
            msg_type = NvRules.MsgType.OPTIMIZATION  # suggest optimization when one side dominates
            if sm_sol_pct > mem_sol_pct:
                focus_metrics.append((sm_sol_pct_name, mem_sol_pct, NvRules.IFrontend.Severity_SEVERITY_LOW, "{:.3f} - {:.3f} >= {:.3f}".format(sm_sol_pct, mem_sol_pct, balanced_threshold)))  # compute dominated
                message = "Compute is more heavily utilized than Memory: Look at the {} section to see what the compute pipelines are spending their time doing. Also, consider whether any computation is redundant and could be reduced or moved to look-up tables.".format(bottleneck_section)
                name = "High Compute Throughput"
            else:
                focus_metrics.append((mem_sol_pct_name, mem_sol_pct, NvRules.IFrontend.Severity_SEVERITY_LOW, "{:.3f} - {:.3f} >= {:.3f}".format(mem_sol_pct, sm_sol_pct, balanced_threshold)))  # memory dominated
                pipe_name = get_max_pipe(breakdown_metrics_memory)  # best guess which memory pipe is hot
                pipe_msg = "to identify the {} bottleneck".format(pipe_name) if pipe_name else "to see where the memory system bottleneck is"
                message = "Memory is more heavily utilized than Compute: Look at the {} section {}. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the bytes transferred. Also consider whether it is possible to do more work per memory access (kernel fusion) or whether there are values you can (re)compute.".format(bottleneck_section, pipe_msg)
                name = "High Memory Throughput"
        else:
            message = "Compute and Memory are well-balanced: To reduce runtime, both computation and memory traffic must be reduced. Check both the @section:ComputeWorkloadAnalysis:Compute Workload Analysis@ and @section:MemoryWorkloadAnalysis:Memory Workload Analysis@ sections."
            name = "Balanced Throughput"
    else:
        pipe_name = None
        if mem_sol_pct > sm_sol_pct:
            pipe_name = get_max_pipe(breakdown_metrics_memory)  # identify the memory pipe if memory is dominant
        pipe_msg = pipe_name if pipe_name else "workloads"
        message = "This workload is utilizing greater than {:.1f}% of the available compute or memory performance of this {}. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing {} in the {} section.".format(no_bound_threshold, resource_partition, pipe_msg, bottleneck_section)
        name = "High Throughput"

    msg_id = fe.message(msg_type, message, name)  # emit the chosen message
    for focus_metric in focus_metrics:
        fe.focus_metric(msg_id, focus_metric[0], focus_metric[1], focus_metric[2], focus_metric[3])  # attach focus metrics

    if sm_sol_pct < no_bound_threshold:
        nvlink_scheduling_metric = metrics["nvlink_scheduling"]
        if nvlink_scheduling_metric and nvlink_scheduling_metric.value():
            message = "This workload uses NVLink-centric scheduling. Some SM resources may not be available to this workload, which can result in lower-than-expected measured utilization."
            msg_id = fe.message(NvRules.MsgType.WARNING, message, "NVLink-Centric Scheduling")
            fe.focus_metric(msg_id, nvlink_scheduling_metric.name(), nvlink_scheduling_metric.value(), NvRules.IFrontend.Severity_SEVERITY_HIGH, "Indicates NVLink-centric scheduling.")
            fe.focus_metric(msg_id, sm_sol_pct_metric.name(), sm_sol_pct_metric.value(), NvRules.IFrontend.Severity_SEVERITY_DEFAULT, "Achievable SM peak values may be lower than usual.")

    # Send weights to child rules (with values in [0,1])
    fe.send_dict_to_children(
        {
            "compute_throughput_normalized": sm_sol_pct / 100,
            "memory_throughput_normalized": mem_sol_pct / 100,
            "max_throughput_normalized": max(sm_sol_pct, mem_sol_pct) / 100,
        }
    )
