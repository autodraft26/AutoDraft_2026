import copy
import bisect
import json
import random
import socket
import subprocess
import threading
from typing import List, Tuple
import time
import torch


from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


def timer(func):
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        result = func(*args, **kwargs)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f'{func.__name__} took {elapsed} seconds')
        return result

    return wrapper


def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
        return processor_list





def reset_past_key_values(passed_key_values: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values



def tree_decoding(
        model,
        draft_input_ids,
        past_key_values,
        draft_position_ids,
        tree_attention_mask,
        gpu_monitor=None,
        step_info=None,
):
    # GPU
    if gpu_monitor:
        gpu_monitor.start_monitoring()
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    outputs, tree_logits, hidden_state = model(
        draft_input_ids,
        tree_attention_mask=tree_attention_mask,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=draft_position_ids,
        init=False,
    )
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # GPU
    gpu_stats = None
    if gpu_monitor:
        gpu_monitor.stop_monitoring()
        gpu_stats = gpu_monitor.get_stats()
    
    if step_info is not None:
        step_info['tree_decoding'] = {
            'time_seconds': end_time - start_time,
            'gpu_stats': gpu_stats,
            'timestamp': time.time()
        }

    return tree_logits, hidden_state, outputs

def verify(input_ids,logits,draft,position_ids,tree_attention_mask,past_key_values_data,current_length_data,parent,model,nodes,threshold,max_depth,logits_processor,gpu_monitor=None,step_info=None):
    # 1 : Best candidate
    if gpu_monitor:
        gpu_monitor.start_monitoring()
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    if logits_processor is None:
        next = torch.argmax(logits, dim=-1)
    else:
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        next = torch.multinomial(probabilities, 1).view(1, -1)
    next=next.to(draft.device)

    parent = torch.where(parent == torch.arange(parent.size(0),device=parent.device), -1, parent)
    parent = torch.cat([torch.tensor([0],device=parent.device), parent + 1], dim=-1).to(draft.device)

    correct = torch.where(draft[0] != next[0][parent], 0, torch.ones(draft.size(1), device=draft.device))
    correct[0] = 1
    last_sum = torch.sum(correct)
    while True:
        correct = torch.where(correct[parent] == 0, 0, correct)
        if torch.sum(correct) == last_sum:
            break
        else:
            last_sum = torch.sum(correct)

    id = torch.argmax(correct * position_ids)
    best_candidate = []
    best_candidate_id = []
    max_id = id
    parent[0] = -1
    while id != -1:
        best_candidate.append(draft[0][id].item())
        best_candidate_id.append(id)
        id = parent[id].item()

    best_candidate.reverse()
    best_candidate_id.reverse()
    next_token = next[0][max_id].unsqueeze(0).unsqueeze(0)
    accept_length=len(best_candidate)-1
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # 1 GPU
    gpu_stats_1 = None
    if gpu_monitor:
        gpu_monitor.stop_monitoring()
        gpu_stats_1 = gpu_monitor.get_stats()
    
    # 1
    if step_info is not None:
        step_info['best_candidate_search'] = {
            'time_seconds': end_time - start_time,
            'gpu_stats': gpu_stats_1, 
            'timestamp': time.time()
        }
    
    # 2 : Past key values
    if gpu_monitor:
        gpu_monitor.start_monitoring()
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    start=current_length_data[0].item()-draft.size(1)
    select_indices=torch.tensor(best_candidate_id)+start

    for data in past_key_values_data:
        # select_indices=tensor([29, 30, 34, 44], device='cuda:0')
        tgt = data[..., select_indices.to(data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = data[..., start: start + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(start + tgt.shape[-2])
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # 2 GPU
    gpu_stats_2 = None
    if gpu_monitor:
        gpu_monitor.stop_monitoring()
        gpu_stats_2 = gpu_monitor.get_stats()
    
    # 2
    if step_info is not None:
        step_info['past_key_values_update'] = {
            'time_seconds': end_time - start_time,
            'gpu_stats': gpu_stats_2,
            'timestamp': time.time()
        }
    
    # 3 : Model.draft 
    if gpu_monitor:
        gpu_monitor.start_monitoring()
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    input_ids=torch.cat([input_ids,torch.tensor(best_candidate,device=input_ids.device).unsqueeze(0)],dim=-1)
    next_draft, next_position_ids, next_tree_attention_mask,parent,next_tree_depth = model.draft(torch.cat((input_ids, next_token.to(input_ids.device)), dim=1),nodes,threshold,max_depth)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # 3 GPU
    gpu_stats_3 = None
    if gpu_monitor:
        gpu_monitor.stop_monitoring()
        gpu_stats_3 = gpu_monitor.get_stats()
    
    # 3
    if step_info is not None:
        step_info['model_draft'] = {
            'time_seconds': end_time - start_time,
            'gpu_stats': gpu_stats_3,
            'timestamp': time.time()
        }
    
    # 4 : (GPU )
    torch.cuda.synchronize()
    start_time = time.time()
    
    next_draft=torch.cat([next_token, next_draft], dim=-1)
    next_position_ids = torch.cat([torch.tensor([next_position_ids[0] - 1],device=next_position_ids.device), next_position_ids], dim=-1)
    next_tree_attention_mask = torch.cat(
        [torch.zeros(1, next_tree_attention_mask.size(1), dtype=next_tree_attention_mask.dtype,device=next_tree_attention_mask.device), next_tree_attention_mask],
        dim=0)
    next_tree_attention_mask = torch.cat(
        [torch.ones(next_tree_attention_mask.size(0), 1, dtype=next_tree_attention_mask.dtype,device=next_tree_attention_mask.device), next_tree_attention_mask],
        dim=1)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # 4
    if step_info is not None:
        step_info['tensor_cleanup'] = {
            'time_seconds': end_time - start_time,
            'gpu_stats': None,
            'timestamp': time.time()
        }

    return input_ids,best_candidate,accept_length,next_draft, next_position_ids, next_tree_attention_mask,parent,next_tree_depth


class CPUPowerMonitor:
    """Class to monitor CPU power using powerstat"""

    def __init__(self, interval=0.1, debug=False):
        # powerstat 0.5 sampling
        self.interval = max(0.5, interval)
        if interval < 0.5 and debug:
            print(f"[draft] CPU power monitoring: requested interval ({interval}) is below the minimum (0.5), so it was adjusted to 0.5")
        self.monitoring = False
        self.data = []
        self.monitor_thread = None
        self.debug = debug
        self.monitor_call_count = 0  # start_monitoring()
        self.powerstat_process = None

    def get_cpu_power_info(self):
        """Get CPU power information using powerstat"""
        try:
            # powerstat -d 0 -R {interval}: interval , RAPL
            # 1 (timeout )
            cmd = ['sudo', 'powerstat', '-d', '0', '-R', str(self.interval), '1']
            if self.debug:
                print(f"[draft] Running CPU power measurement command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=self.interval + 2
            )

            if self.debug:
                print(f"[draft] powerstat return code: {result.returncode}")
                if result.stdout:
                    print(f"[draft] powerstat stdout (first 500 characters): {result.stdout[:500]}")
                if result.stderr:
                    print(f"[draft] powerstat stderr: {result.stderr}")

            if result.returncode == 0:
                # powerstat 
                lines = result.stdout.strip().split('\n')
                if self.debug:
                    print(f"[draft] powerstat output line count: {len(lines)}")

                for line in lines:
                    if not line.strip() or line.startswith('Time') or line.startswith('---'):
                        continue

                    parts = line.split()
                    if self.debug:
                        print(f"[draft] Trying to parse line: {line}, column count: {len(parts)}")

                    if len(parts) >= 10:
                        try:
                            # AvgPower (W )
                            # : "15.2W" "15.2"
                            avg_power_str = parts[-2] if len(parts) >= 2 else parts[-1]
                            # "W"
                            avg_power_str = avg_power_str.rstrip('W')
                            avg_power_w = float(avg_power_str)

                            if self.debug:
                                print(f"[draft] CPU power measurement succeeded: {avg_power_w}W")

                            return {
                                'cpu_power_w': avg_power_w,
                                'timestamp': time.time()
                            }
                        except (ValueError, IndexError) as e:
                            if self.debug:
                                print(f"[draft] powerstat parse error: {e}, line: {line}, parts: {parts}")
                            continue
                if self.debug:
                    print(f"[draft] powerstat parse failed: no valid data line found")
                return self._get_default_cpu_power_info()
            if self.debug:
                print(f"[draft] powerstat execution failed (return code: {result.returncode}): {result.stderr}")
            return self._get_default_cpu_power_info()
        except subprocess.TimeoutExpired:
            if self.debug:
                print(f"[draft] powerstat timeout (interval: {self.interval})")
            return self._get_default_cpu_power_info()
        except Exception as e:
            if self.debug:
                print(f"[draft] CPU power failed to get information: {e}", exc_info=True)
            return self._get_default_cpu_power_info()

    def _get_default_cpu_power_info(self):
        """Returns basic CPU power information"""
        return {
            'cpu_power_w': None,
            'timestamp': time.time()
        }

    def monitor_loop(self):
        """monitoring loop"""
        loop_count = 0
        while self.monitoring:
            loop_count += 1
            if self.debug:
                print(f"[draft] CPU power monitoring loop running (count: {loop_count})")

            cpu_power_info = self.get_cpu_power_info()
            if cpu_power_info:
                self.data.append(cpu_power_info)
                if self.debug:
                    print(f"[draft] CPU power data collected: {cpu_power_info}, total data count: {len(self.data)}")
            else:
                if self.debug:
                    print(f"[draft] CPU power data collection failed (None Returns)")
            time.sleep(self.interval)

    def start_monitoring(self):
        """Start monitoring"""
        self.monitor_call_count += 1
        if not self.monitoring:
            self.monitoring = True
            self.data = []
            self.monitor_thread = threading.Thread(target=self.monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop monitoring"""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join()

    def get_stats(self):
        """Return statistical information"""
        if self.debug:
            print(f"[draft] CPU power Statistics requested (data count: {len(self.data)})")

        if not self.data:
            if self.debug:
                print(f"[draft] CPU power Statistics: no data available, None Returns")
            return None

        # CPU
        cpu_power_values = [entry['cpu_power_w'] for entry in self.data if entry['cpu_power_w'] is not None]

        if self.debug:
            print(f"[draft] CPU power Statistics: valid value count: {len(cpu_power_values)} / total data count: {len(self.data)}")
            if cpu_power_values:
                print(f"[draft] CPU power value range: {min(cpu_power_values):.2f}W ~ {max(cpu_power_values):.2f}W")

        if not cpu_power_values:
            if self.debug:
                print(f"[draft] CPU power Statistics: no valid values available, None Returns")
            return None

        stats = {
            'cpu_power_w': {
                'avg': sum(cpu_power_values) / len(cpu_power_values),
                'max': max(cpu_power_values),
                'min': min(cpu_power_values),
                'count': len(cpu_power_values)
            }
        }

        if self.debug:
            print(f"[draft] CPU power Statistics calculation complete: {stats}")

        return stats


class GPUMonitor:
    """Class to monitor NVIDIA GPU memory and utilization"""

    def __init__(self, interval=0.1, fix_gpu_clock=False, graphics_clock=None, memory_clock=None, debug=False):
        self.interval = interval
        self.monitoring = False
        self.data = []
        self._timestamps = []
        self._lock = threading.RLock()
        self.retention_seconds = max(60.0, float(interval) * 4096.0, 300.0)
        self.monitor_thread = None
        self.fix_gpu_clock = fix_gpu_clock
        self.graphics_clock = graphics_clock
        self.memory_clock = memory_clock
        self.original_graphics_clock = None
        self.original_memory_clock = None
        self.debug = debug
        self.monitor_call_count = 0  # start_monitoring()
        self.backend = "nvidia-smi"
        self._nvml = None
        self._nvml_handles = []
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            device_count = pynvml.nvmlDeviceGetCount()
            self._nvml_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(device_count)
            ]
            if self._nvml_handles:
                self.backend = "nvml"
        except Exception as e:
            self._nvml = None
            self._nvml_handles = []
            if self.debug:
                print(f"[draft] NVML initialization failed, nvidia-smi using fallback: {e}")

        if self.fix_gpu_clock:
            self._set_gpu_clocks()

    def _set_gpu_clocks(self):
        """Set GPU clock to fixed"""
        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=clocks.current.graphics,clocks.current.memory',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0].strip():
                    parts = lines[0].split(', ')
                    if len(parts) >= 2:
                        self.original_graphics_clock = int(parts[0])
                        self.original_memory_clock = int(parts[1])
                        if self.debug:
                            print(f"[draft] Saved original GPU clocks: Graphics={self.original_graphics_clock}MHz, Memory={self.original_memory_clock}MHz")

            # GPU
            # nvidia-smi --applications-clocks "memory,graphics" .
            target_graphics = self.graphics_clock
            target_memory = self.memory_clock
            if target_graphics is None or target_memory is None:
                max_result = subprocess.run([
                    'nvidia-smi',
                    '--query-gpu=clocks.max.graphics,clocks.max.memory',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=5)
                if max_result.returncode == 0:
                    lines = max_result.stdout.strip().split('\n')
                    if lines and lines[0].strip():
                        parts = lines[0].split(', ')
                        if len(parts) >= 2:
                            try:
                                max_graphics = int(parts[0])
                            except Exception:
                                max_graphics = None
                            try:
                                max_memory = int(parts[1])
                            except Exception:
                                max_memory = None
                            if target_graphics is None:
                                target_graphics = max_graphics
                            if target_memory is None:
                                target_memory = max_memory

            if target_graphics is None or target_memory is None:
                if self.debug:
                    print("[draft] Skipping GPU clock fix: could not determine target graphics/memory clocks")
                return

            cmd = ['nvidia-smi', f'--applications-clocks={target_memory},{target_graphics}']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                if self.debug:
                    print(f"[draft] Set fixed GPU clocks: Graphics={target_graphics}MHz, Memory={target_memory}MHz")
            else:
                if self.debug:
                    print(f"[draft] Failed to set GPU clocks: {result.stderr}")
        except Exception as e:
            if self.debug:
                print(f"[draft] Error while setting GPU clocks: {e}")

    def _restore_gpu_clocks(self):
        """Restore GPU clocks to their original state"""
        try:
            if self.original_graphics_clock is not None and self.original_memory_clock is not None:
                cmd = ['nvidia-smi', '--applications-clocks=default']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

                if result.returncode == 0:
                    if self.debug:
                        print(f"[draft] GPU clocks restored")
                else:
                    if self.debug:
                        print(f"[draft] Failed to restore GPU clocks: {result.stderr}")
        except Exception as e:
            if self.debug:
                print(f"[draft] Error while restoring GPU clocks: {e}")

    def __del__(self):
        """Restoring GPU clock from destructor"""
        if self.fix_gpu_clock:
            self._restore_gpu_clocks()

    def get_gpu_info(self):
        """Get GPU information using nvidia-smi"""
        if self._nvml is not None and self._nvml_handles:
            try:
                gpu_info = []
                for gpu_id, handle in enumerate(self._nvml_handles):
                    mem = self._nvml.nvmlDeviceGetMemoryInfo(handle)
                    util = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                    try:
                        temperature_c = self._nvml.nvmlDeviceGetTemperature(
                            handle, self._nvml.NVML_TEMPERATURE_GPU
                        )
                    except Exception:
                        temperature_c = 0
                    try:
                        power_draw_w = self._nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    except Exception:
                        power_draw_w = None
                    try:
                        power_limit_w = self._nvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
                    except Exception:
                        power_limit_w = None
                    try:
                        graphics_clock_mhz = self._nvml.nvmlDeviceGetClockInfo(
                            handle, self._nvml.NVML_CLOCK_GRAPHICS
                        )
                    except Exception:
                        graphics_clock_mhz = None
                    try:
                        memory_clock_mhz = self._nvml.nvmlDeviceGetClockInfo(
                            handle, self._nvml.NVML_CLOCK_MEM
                        )
                    except Exception:
                        memory_clock_mhz = None
                    try:
                        max_graphics_clock_mhz = self._nvml.nvmlDeviceGetMaxClockInfo(
                            handle, self._nvml.NVML_CLOCK_GRAPHICS
                        )
                    except Exception:
                        max_graphics_clock_mhz = None
                    try:
                        max_memory_clock_mhz = self._nvml.nvmlDeviceGetMaxClockInfo(
                            handle, self._nvml.NVML_CLOCK_MEM
                        )
                    except Exception:
                        max_memory_clock_mhz = None
                    memory_total_mb = max(1, int(mem.total / (1024 * 1024)))
                    memory_used_mb = int(mem.used / (1024 * 1024))
                    power_usage_percent = None
                    if power_draw_w is not None and power_limit_w is not None and power_limit_w > 0:
                        power_usage_percent = (power_draw_w / power_limit_w) * 100.0
                    gpu_info.append({
                        'gpu_id': gpu_id,
                        'memory_used_mb': memory_used_mb,
                        'memory_total_mb': memory_total_mb,
                        'utilization_percent': int(util.gpu),
                        'temperature_c': int(temperature_c or 0),
                        'power_draw_w': power_draw_w,
                        'power_limit_w': power_limit_w,
                        'graphics_clock_mhz': graphics_clock_mhz,
                        'memory_clock_mhz': memory_clock_mhz,
                        'max_graphics_clock_mhz': max_graphics_clock_mhz,
                        'max_memory_clock_mhz': max_memory_clock_mhz,
                        'memory_used_percent': (memory_used_mb / memory_total_mb) * 100.0,
                        'power_usage_percent': power_usage_percent,
                        'timestamp': time.time()
                    })
                return gpu_info
            except Exception as e:
                if self.debug:
                    print(f"[draft] NVML Failed to get GPU information, nvidia-smi using fallback: {e}")
        try:
            def _to_int(value):
                if value is None:
                    return None
                s = str(value).strip()
                if s in {"N/A", "[N/A]", "[Not Supported]", "Not Supported", ""}:
                    return None
                try:
                    return int(float(s))
                except Exception:
                    return None

            def _to_float(value):
                if value is None:
                    return None
                s = str(value).strip()
                if s in {"N/A", "[N/A]", "[Not Supported]", "Not Supported", ""}:
                    return None
                try:
                    return float(s)
                except Exception:
                    return None

            # nvidia-smi GPU 
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,clocks.max.graphics,clocks.max.memory',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info = []
                for line in lines:
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 11:
                            gpu_id = _to_int(parts[0]) or 0
                            memory_used_mb = _to_int(parts[1]) or 0
                            memory_total_mb = _to_int(parts[2]) or 1
                            utilization_percent = _to_int(parts[3]) or 0
                            temperature_c = _to_int(parts[4]) or 0
                            power_draw_w = _to_float(parts[5])
                            power_limit_w = _to_float(parts[6])
                            graphics_clock_mhz = _to_int(parts[7])
                            memory_clock_mhz = _to_int(parts[8])
                            max_graphics_clock_mhz = _to_int(parts[9])
                            max_memory_clock_mhz = _to_int(parts[10])

                            memory_used_percent = (memory_used_mb / memory_total_mb) * 100 if memory_total_mb > 0 else 0
                            power_usage_percent = None
                            if power_draw_w is not None and power_limit_w is not None and power_limit_w > 0:
                                power_usage_percent = (power_draw_w / power_limit_w) * 100

                            gpu_info.append({
                                'gpu_id': gpu_id,
                                'memory_used_mb': memory_used_mb,
                                'memory_total_mb': memory_total_mb,
                                'utilization_percent': utilization_percent,
                                'temperature_c': temperature_c,
                                'power_draw_w': power_draw_w,
                                'power_limit_w': power_limit_w,
                                'graphics_clock_mhz': graphics_clock_mhz,
                                'memory_clock_mhz': memory_clock_mhz,
                                'max_graphics_clock_mhz': max_graphics_clock_mhz,
                                'max_memory_clock_mhz': max_memory_clock_mhz,
                                'memory_used_percent': memory_used_percent,
                                'power_usage_percent': power_usage_percent,
                                'timestamp': time.time()
                            })
                return gpu_info
            print(f"nvidia-smi execution failed: {result.stderr}")
            return self._get_default_gpu_info()
        except Exception as e:
            print(f"Failed to get GPU information: {e}")
            return self._get_default_gpu_info()

    def _get_default_gpu_info(self):
        """Returns basic GPU information"""
        return [{
            'gpu_id': 0,
            'memory_used_mb': 0,
            'memory_total_mb': 1,
            'utilization_percent': 0,
            'temperature_c': 0,
            'power_draw_w': None,
            'power_limit_w': None,
            'graphics_clock_mhz': None,
            'memory_clock_mhz': None,
            'max_graphics_clock_mhz': None,
            'max_memory_clock_mhz': None,
            'memory_used_percent': 0,
            'power_usage_percent': None,
            'timestamp': time.time()
        }]

    def monitor_loop(self):
        """monitoring loop"""
        while self.monitoring:
            gpu_info = self.get_gpu_info()
            if gpu_info:
                timestamp = time.time()
                new_entries = []
                new_timestamps = []
                for i, gpu in enumerate(gpu_info):
                    new_entries.append({
                        'timestamp': timestamp,
                        'gpu_id': i,
                        **gpu
                    })
                    new_timestamps.append(timestamp)
                with self._lock:
                    self.data.extend(new_entries)
                    self._timestamps.extend(new_timestamps)
                    self._prune_locked(time.time() - self.retention_seconds)
            time.sleep(self.interval)

    def _prune_locked(self, min_timestamp):
        """Remove old samples from self._lock holding state."""
        if not self._timestamps:
            return
        idx = bisect.bisect_left(self._timestamps, float(min_timestamp))
        if idx > 0:
            del self._timestamps[:idx]
            del self.data[:idx]

    def reset_data(self):
        """Clear accumulated GPU samples. Maintain a long-running monitor thread."""
        with self._lock:
            self.data = []
            self._timestamps = []

    def start_monitoring(self):
        """Start monitoring"""
        self.monitor_call_count += 1
        if not self.monitoring:
            self.monitoring = True
            self.reset_data()
            self.monitor_thread = threading.Thread(target=self.monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop monitoring"""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join()

    def get_stats(self, start_time=None, end_time=None):
        """Return statistical information"""
        if start_time is not None or end_time is not None:
            start = float(start_time) if start_time is not None else float("-inf")
            end = float(end_time) if end_time is not None else float("inf")
            with self._lock:
                if not self.data:
                    return None
                left = 0 if start == float("-inf") else bisect.bisect_left(self._timestamps, start)
                right = len(self.data) if end == float("inf") else bisect.bisect_right(self._timestamps, end)
                entries = list(self.data[left:right])
                if not entries and end_time is not None:
                    # interval( 0 ) end .
                    prior_idx = bisect.bisect_right(self._timestamps, end) - 1
                    if prior_idx >= 0:
                        entries = [dict(self.data[prior_idx])]
            if not entries and end_time is not None:
                return None
        else:
            with self._lock:
                if not self.data:
                    return None
                entries = list(self.data)

        # GPU
        gpu_stats = {}
        for entry in entries:
            gpu_id = entry['gpu_id']
            if gpu_id not in gpu_stats:
                gpu_stats[gpu_id] = {
                    'memory_used_mb': [],
                    'memory_total_mb': [],
                    'utilization_percent': [],
                    'temperature_c': [],
                    'power_draw_w': [],
                    'power_limit_w': [],
                    'power_usage_percent': [],
                    'graphics_clock_mhz': [],
                    'memory_clock_mhz': [],
                    'max_graphics_clock_mhz': [],
                    'max_memory_clock_mhz': [],
                    'memory_used_percent': []
                }

            gpu_stats[gpu_id]['memory_used_mb'].append(entry['memory_used_mb'])
            gpu_stats[gpu_id]['memory_total_mb'].append(entry['memory_total_mb'])
            gpu_stats[gpu_id]['utilization_percent'].append(entry['utilization_percent'])
            gpu_stats[gpu_id]['temperature_c'].append(entry['temperature_c'])
            gpu_stats[gpu_id]['memory_used_percent'].append(entry['memory_used_percent'])

            # (None )
            if entry['power_draw_w'] is not None:
                gpu_stats[gpu_id]['power_draw_w'].append(entry['power_draw_w'])
            if entry['power_limit_w'] is not None:
                gpu_stats[gpu_id]['power_limit_w'].append(entry['power_limit_w'])
            if entry['power_usage_percent'] is not None:
                gpu_stats[gpu_id]['power_usage_percent'].append(entry['power_usage_percent'])

            # (None )
            if entry['graphics_clock_mhz'] is not None:
                gpu_stats[gpu_id]['graphics_clock_mhz'].append(entry['graphics_clock_mhz'])
            if entry['memory_clock_mhz'] is not None:
                gpu_stats[gpu_id]['memory_clock_mhz'].append(entry['memory_clock_mhz'])
            if entry['max_graphics_clock_mhz'] is not None:
                gpu_stats[gpu_id]['max_graphics_clock_mhz'].append(entry['max_graphics_clock_mhz'])
            if entry['max_memory_clock_mhz'] is not None:
                gpu_stats[gpu_id]['max_memory_clock_mhz'].append(entry['max_memory_clock_mhz'])

        # , ,
        stats = {}
        for gpu_id, data in gpu_stats.items():
            stats[f'gpu_{gpu_id}'] = {
                'memory_used_mb': {
                    'avg': sum(data['memory_used_mb']) / len(data['memory_used_mb']),
                    'max': max(data['memory_used_mb']),
                    'min': min(data['memory_used_mb'])
                },
                'memory_total_mb': data['memory_total_mb'][0] if data['memory_total_mb'] else 0,
                'utilization_percent': {
                    'avg': sum(data['utilization_percent']) / len(data['utilization_percent']),
                    'max': max(data['utilization_percent']),
                    'min': min(data['utilization_percent'])
                },
                'temperature_c': {
                    'avg': sum(data['temperature_c']) / len(data['temperature_c']),
                    'max': max(data['temperature_c']),
                    'min': min(data['temperature_c'])
                },
                'memory_used_percent': {
                    'avg': sum(data['memory_used_percent']) / len(data['memory_used_percent']),
                    'max': max(data['memory_used_percent']),
                    'min': min(data['memory_used_percent'])
                }
            }

            if data['power_draw_w']:
                stats[f'gpu_{gpu_id}']['power_draw_w'] = {
                    'avg': sum(data['power_draw_w']) / len(data['power_draw_w']),
                    'max': max(data['power_draw_w']),
                    'min': min(data['power_draw_w'])
                }
            if data['power_limit_w']:
                stats[f'gpu_{gpu_id}']['power_limit_w'] = data['power_limit_w'][0]
            if data['power_usage_percent']:
                stats[f'gpu_{gpu_id}']['power_usage_percent'] = {
                    'avg': sum(data['power_usage_percent']) / len(data['power_usage_percent']),
                    'max': max(data['power_usage_percent']),
                    'min': min(data['power_usage_percent'])
                }

            if data['graphics_clock_mhz']:
                stats[f'gpu_{gpu_id}']['graphics_clock_mhz'] = {
                    'avg': sum(data['graphics_clock_mhz']) / len(data['graphics_clock_mhz']),
                    'max': max(data['graphics_clock_mhz']),
                    'min': min(data['graphics_clock_mhz'])
                }
            if data['memory_clock_mhz']:
                stats[f'gpu_{gpu_id}']['memory_clock_mhz'] = {
                    'avg': sum(data['memory_clock_mhz']) / len(data['memory_clock_mhz']),
                    'max': max(data['memory_clock_mhz']),
                    'min': min(data['memory_clock_mhz'])
                }
            if data['max_graphics_clock_mhz']:
                stats[f'gpu_{gpu_id}']['max_graphics_clock_mhz'] = data['max_graphics_clock_mhz'][0]
            if data['max_memory_clock_mhz']:
                stats[f'gpu_{gpu_id}']['max_memory_clock_mhz'] = data['max_memory_clock_mhz'][0]

        return stats

    def get_stats_between(self, start_time, end_time):
        """Only samples between start_time and end_time are counted."""
        return self.get_stats(start_time=start_time, end_time=end_time)


def send_json(sock: socket.socket, payload: dict) -> None:
    """Send JSON data"""
    data = (json.dumps(payload) + "\n").encode("utf-8")
    sock.sendall(data)


def recv_json(sock: socket.socket) -> dict:
    """Receive JSON data"""
    buffer = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Socket closed by peer")
        buffer += chunk
        if b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            return json.loads(line.decode("utf-8"))


def send_json_with_size(sock: socket.socket, payload: dict) -> int:
    """Send JSON data and return the number of bytes sent"""
    data = (json.dumps(payload) + "\n").encode("utf-8")
    sock.sendall(data)
    return len(data)


def recv_json_with_size(sock: socket.socket) -> Tuple[dict, int]:
    """Receives JSON data and returns the number of bytes received"""
    buffer = b""
    total_bytes = 0
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Socket closed by peer")
        buffer += chunk
        total_bytes += len(chunk)
        if b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            return json.loads(line.decode("utf-8")), total_bytes

