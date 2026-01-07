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
    MetricRequest("gpc__cycles_elapsed.max", "duration"),
    MetricRequest("smsp__pcsamp_sample_count", "count", Importance.OPTIONAL, None, False),
    MetricRequest("smsp__pcsamp_interval_cycles", "interval", Importance.OPTIONAL, None, False),
    MetricRequest("smsp__pcsamp_buffer_overflow", "buffer_overflow", Importance.OPTIONAL, None, False),
    MetricRequest("smsp__pcsamp_buffer_size_bytes", "buffer_size", Importance.OPTIONAL, None, False),
    MetricRequest("smsp__pcsamp_dropped_bytes", "dropped_bytes", Importance.OPTIONAL, None, False),
]


def get_identifier():
    return "PCSamplingData"


def get_name():
    return "PC sampling data"


def get_description():
    return "PC sampling data"


def get_section_identifier():
    return "SourceCounters"

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    if action.workload_type() != NvRules.IAction.WorkloadType_KERNEL:
        return

    metrics = RequestedMetricsParser(handle, action).parse(requested_metrics)
    interval = metrics["interval"]
    if not interval or not interval.value():
        # likely not supported
        return

    sample_count = metrics["count"].value()
    interval_cycles = interval.value()
    buffer_size = metrics["buffer_size"].value()
    kernel_duration = metrics["duration"].value()

    dropped_bytes = metrics["dropped_bytes"]
    buffer_overflow = metrics["buffer_overflow"]

    if dropped_bytes and dropped_bytes.value():
        message = "Some samples were dropped with a sampling interval of {} due to backpressure, leading to potentially inaccurate sampling data. It is recommended to increase the warp sampling interval using `--warp-sampling-interval` option to mitigate this issue.".format(int(interval_cycles))
        fe.message(NvRules.MsgType.WARNING, message, "Dropped Samples")

    if buffer_overflow and buffer_overflow.value():
        message = "A buffer overflow occurred, resulting in potentially inaccurate sampling data. It is recommended to increase the warp sampling buffer size using `--warp-sampling-buffer-size` option to address this issue. A buffer of size {} bytes was used.".format(int(buffer_size))
        fe.message(NvRules.MsgType.WARNING, message, "Buffer Overflow")

    if sample_count == 0:
        message = "Sampling metrics were enabled, but no samples could be collected for this kernel."

        if interval_cycles >= kernel_duration:
            message += " Note that the kernel duration is shorter than the sampling interval."
        fe.message(NvRules.MsgType.WARNING, message, "No Samples")
