import argparse
import concurrent.futures
import gc
import json
import os
import random
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import importlib # added
import torch
from transformers import AutoTokenizer

from opt_classic.kv_cache import initialize_past_key_values
from opt_classic.utils import GPUMonitor as NVMLGPUMonitor
from opt_classic.utils import prepare_logits_processor, recv_json, send_json


def _resolve_data_root(parent_dir: str) -> str:
    """Resolve the directory used for ``data/profile`` caches. Honors
    ``AUTODRAFT_DATA_DIR`` so PyPI-installed users don't write into
    site-packages."""
    override = os.environ.get("AUTODRAFT_DATA_DIR")
    if override:
        return override
    return os.path.join(parent_dir, "data")


def set_seed(seed: int):
    """Fixed torch, random, and numpy seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def set_deterministic():
    """Disable PyTorch non-deterministic algorithm for identical results even on GPU operations (some operations may be slower)."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


PROFILE_WARMUP_RUNS = 2
PROFILE_BURNIN_RUNS = 1
TARGET_PROFILE_FIXED_WIDTH = 50
TARGET_PROFILE_FIXED_DEPTH = 10


def get_kv_llama_class(base_model_path: str):
    """
    Select the KV implementation class based on the model family.
    Supported: LLaMA2, LLaMA3, Qwen2.5, Qwen3
    """
    model_path_l = (base_model_path or "").lower()
    is_llama3_family = ("llama-3" in model_path_l) or ("llama3" in model_path_l)
    is_qwen3_family = ("qwen3" in model_path_l) or ("qwen-3" in model_path_l) or ("qwen/qwen3" in model_path_l)
    is_qwen2_family = (not is_qwen3_family) and (
        ("qwen2" in model_path_l) or ("qwen-2" in model_path_l) or ("qwen/qwen2" in model_path_l)
    )

    if is_qwen3_family:
        module_name, class_name = "opt_classic.modeling_qwen3_kv", "Qwen3ForCausalLM"
    elif is_qwen2_family:
        module_name, class_name = "opt_classic.modeling_qwen2_kv", "Qwen2ForCausalLM"
    elif is_llama3_family:
        module_name, class_name = "opt_classic.modeling_llama3_kv", "LlamaForCausalLM"
    else:
        module_name, class_name = "opt_classic.modeling_llama_kv", "LlamaForCausalLM"

    print(f'importing {module_name}.{class_name}')
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, class_name)
    except Exception as e:
        raise RuntimeError(f"Failed to import {module_name}.{class_name}: {e}")
    





def get_available_gpu_clocks():
    """Returns the available GPU clock combinations"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-supported-clocks=graphics,memory',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            clock_combinations = []
            for line in lines:
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 2:
                        graphics = int(parts[0])
                        memory = int(parts[1])
                        clock_combinations.append((graphics, memory))
            return clock_combinations
        else:
            print(f"Failed to query GPU clocks: {result.stderr}")
            return []
    except Exception as e:
        print(f"Error while querying GPU clocks: {e}")
        return []


class GPUMonitor:
    """Class to monitor NVIDIA GPU memory and utilization"""
    
    def __init__(self, interval=0.1, fix_gpu_clock=False, graphics_clock=None, memory_clock=None, debug=False):
        self.interval = interval
        self.monitoring = False
        self.data = []
        self.monitor_thread = None
        self.fix_gpu_clock = fix_gpu_clock
        self.graphics_clock = graphics_clock
        self.memory_clock = memory_clock
        self.original_graphics_clock = None
        self.original_memory_clock = None
        self.debug = debug
        self.monitor_call_count = 0  # start_monitoring()
        
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
                            print(f"[target] Saved original GPU clocks: Graphics={self.original_graphics_clock}MHz, Memory={self.original_memory_clock}MHz")
            
            # GPU
            cmd = ['nvidia-smi', '--applications-clocks=']
            if self.graphics_clock is not None:
                cmd[1] += f"{self.graphics_clock}"
            else:
                cmd[1] += "0"
            
            if self.memory_clock is not None:
                cmd[1] += f",{self.memory_clock}"
            else:
                cmd[1] += ",0"
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                if self.debug:
                    print(f"[target] Set fixed GPU clocks: Graphics={self.graphics_clock}MHz, Memory={self.memory_clock}MHz")
            else:
                    if self.debug:
                        print(f"[target] Failed to set GPU clocks: {result.stderr}")
                
        except Exception as e:
            if self.debug:
                print(f"[target] Error while setting GPU clocks: {e}")
    
    def _restore_gpu_clocks(self):
        """Restore GPU clocks to their original state"""
        try:
            if self.original_graphics_clock is not None and self.original_memory_clock is not None:
                cmd = ['nvidia-smi', '--applications-clocks=default']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    if self.debug:
                        print(f"[target] GPU clocks restored")
                else:
                    if self.debug:
                        print(f"[target] Failed to restore GPU clocks: {result.stderr}")
        except Exception as e:
            if self.debug:
                print(f"[target] Error while restoring GPU clocks: {e}")
    
    def __del__(self):
        """Restoring GPU clock from destructor"""
        if self.fix_gpu_clock:
            self._restore_gpu_clocks()
        
    def get_gpu_info(self):
        """Get GPU information using nvidia-smi"""
        try:
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
                            gpu_info.append({
                                'gpu_id': int(parts[0]),
                                'memory_used_mb': int(parts[1]),
                                'memory_total_mb': int(parts[2]),
                                'utilization_percent': int(parts[3]),
                                'temperature_c': int(parts[4]),
                                'power_draw_w': float(parts[5]) if parts[5] != '[Not Supported]' and parts[5] != 'N/A' else None,
                                'power_limit_w': float(parts[6]) if parts[6] != '[Not Supported]' and parts[6] != 'N/A' else None,
                                'graphics_clock_mhz': int(parts[7]) if parts[7] != '[Not Supported]' else None,
                                'memory_clock_mhz': int(parts[8]) if parts[8] != '[Not Supported]' else None,
                                'max_graphics_clock_mhz': int(parts[9]) if parts[9] != '[Not Supported]' else None,
                                'max_memory_clock_mhz': int(parts[10]) if parts[10] != '[Not Supported]' else None,
                                'memory_used_percent': (int(parts[1]) / int(parts[2])) * 100 if int(parts[2]) > 0 else 0,
                                'power_usage_percent': (float(parts[5]) / float(parts[6])) * 100 if parts[5] != '[Not Supported]' and parts[6] != '[Not Supported]' and parts[5] != 'N/A' and parts[6] != 'N/A' and float(parts[6]) > 0 else None,
                                'timestamp': time.time()
                            })
                return gpu_info
            else:
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
                for i, gpu in enumerate(gpu_info):
                    self.data.append({
                        'timestamp': timestamp,
                        'gpu_id': i,
                        **gpu
                    })
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
            # print("[target] GPU ")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join()
            # print("[target] GPU ")
    
    def get_stats(self):
        """Return statistical information"""
        if not self.data:
            return None
        
        # GPU
        gpu_stats = {}
        for entry in self.data:
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
    
    def save_data(self, filename=None):
        """Save data as JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gpu_monitor_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=2)
        if self.debug:
            print(f"[target] Saved GPU monitoring data: {filename}")
        return filename


def save_performance_stats(timing_data, gpu_data, output_file, step_count, args=None, accept_lengths=None):
    """Save performance statistics as JSON file"""
    # output_file
    if not output_file:
        # result
        script_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(script_dir)
        result_dir = os.path.join(parent_dir, "result")
        os.makedirs(result_dir, exist_ok=True)
        
        # ( _ )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(result_dir, f"target_gpu_results_{timestamp}.json")
    
    times = [data["total_time_seconds"] for data in timing_data]
    timing_stats = {
        "count": len(times),
        "min_time_seconds": min(times) if times else 0,
        "max_time_seconds": max(times) if times else 0,
        "avg_time_seconds": sum(times) / len(times) if times else 0,
        "total_time_seconds": sum(times),
        "num_steps": len(times),
        "times": times
    }
    
    # Accept length
    accept_stats = {}
    if accept_lengths:
        accept_stats = {
            "total_accepted_tokens": sum(accept_lengths),
            "avg_accept_length": sum(accept_lengths) / len(accept_lengths) if accept_lengths else 0,
            "min_accept_length": min(accept_lengths) if accept_lengths else 0,
            "max_accept_length": max(accept_lengths) if accept_lengths else 0,
            "num_steps": len(accept_lengths)
        }
    
    # GPU 
    gpu_summary = {}
    if gpu_data:
        # GPU
        all_gpu_stats = {}
        for step_data in gpu_data:
            for gpu_id, stats in step_data["gpu_stats"].items():
                if gpu_id not in all_gpu_stats:
                    all_gpu_stats[gpu_id] = {
                        'memory_used_mb': [],
                        'utilization_percent': [],
                        'temperature_c': [],
                        'memory_used_percent': [],
                        'power_draw_w': [],
                        'power_limit_w': [],
                        'power_usage_percent': []
                    }
                
                all_gpu_stats[gpu_id]['memory_used_mb'].extend([stats['memory_used_mb']['avg']])
                all_gpu_stats[gpu_id]['utilization_percent'].extend([stats['utilization_percent']['avg']])
                all_gpu_stats[gpu_id]['temperature_c'].extend([stats['temperature_c']['avg']])
                all_gpu_stats[gpu_id]['memory_used_percent'].extend([stats['memory_used_percent']['avg']])
                
                if 'power_draw_w' in stats:
                    all_gpu_stats[gpu_id]['power_draw_w'].extend([stats['power_draw_w']['avg']])
                if 'power_limit_w' in stats:
                    all_gpu_stats[gpu_id]['power_limit_w'].extend([stats['power_limit_w']])
                if 'power_usage_percent' in stats:
                    all_gpu_stats[gpu_id]['power_usage_percent'].extend([stats['power_usage_percent']['avg']])
        
        for gpu_id, data in all_gpu_stats.items():
            gpu_summary[gpu_id] = {
                'memory_used_mb': {
                    'min': min(data['memory_used_mb']),
                    'max': max(data['memory_used_mb']),
                    'avg': sum(data['memory_used_mb']) / len(data['memory_used_mb'])
                },
                'utilization_percent': {
                    'min': min(data['utilization_percent']),
                    'max': max(data['utilization_percent']),
                    'avg': sum(data['utilization_percent']) / len(data['utilization_percent'])
                },
                'temperature_c': {
                    'min': min(data['temperature_c']),
                    'max': max(data['temperature_c']),
                    'avg': sum(data['temperature_c']) / len(data['temperature_c'])
                },
                'memory_used_percent': {
                    'min': min(data['memory_used_percent']),
                    'max': max(data['memory_used_percent']),
                    'avg': sum(data['memory_used_percent']) / len(data['memory_used_percent'])
                }
            }
            
            if data['power_draw_w']:
                gpu_summary[gpu_id]['power_draw_w'] = {
                    'min': min(data['power_draw_w']),
                    'max': max(data['power_draw_w']),
                    'avg': sum(data['power_draw_w']) / len(data['power_draw_w'])
                }
            if data['power_limit_w']:
                # power_limit_w
                gpu_summary[gpu_id]['power_limit_w'] = data['power_limit_w'][0]
            if data['power_usage_percent']:
                gpu_summary[gpu_id]['power_usage_percent'] = {
                    'min': min(data['power_usage_percent']),
                    'max': max(data['power_usage_percent']),
                    'avg': sum(data['power_usage_percent']) / len(data['power_usage_percent'])
                }
            
            # GPU 
            # step power_draw_w  total_time_seconds
            if data['power_draw_w'] and timing_data:
                total_energy_joules = 0.0
                step_energies = []
                
                # gpu_data timing_data step
                for step_idx, step_data in enumerate(gpu_data):
                    if step_idx < len(timing_data):
                        step_num = step_data.get('step', step_idx + 1)
                        # step GPU power_draw_w
                        if gpu_id in step_data.get('gpu_stats', {}):
                            step_power_avg = step_data['gpu_stats'][gpu_id].get('power_draw_w', {}).get('avg', 0.0)
                            step_time = timing_data[step_idx].get('total_time_seconds', 0.0)
                            step_energy = step_power_avg * step_time  # = (Joule)
                            step_energies.append(step_energy)
                            total_energy_joules += step_energy
                
                # start_monitoring() (target call )
                # gpu_data monitor_call_count
                target_call_count = step_count  # : step_count
                if gpu_data:
                    last_step_data = gpu_data[-1]
                    if last_step_data.get('monitor_call_count') is not None:
                        target_call_count = last_step_data.get('monitor_call_count')
                
                if step_energies and target_call_count > 0:
                    gpu_summary[gpu_id]['energy'] = {
                        'total_energy_joules': float(total_energy_joules),
                        'total_energy_kwh': float(total_energy_joules / 3600000.0),  # kWh (1 kWh = 3,600,000 J)
                        'avg_energy_per_call_joules': float(total_energy_joules / target_call_count),
                        'avg_energy_per_call_kwh': float((total_energy_joules / target_call_count) / 3600000.0),
                        'min_energy_per_call_joules': float(min(step_energies)) if step_energies else 0.0,
                        'max_energy_per_call_joules': float(max(step_energies)) if step_energies else 0.0,
                        'num_calls': int(target_call_count)  # start_monitoring()
                    }
    
    # Experiment info
    experiment_info = {
        "timestamp": datetime.now().isoformat(),
        "total_steps": step_count,
        "description": "Target model verification performance statistics"
    }
    
    # args
    if args:
        experiment_info.update({
            "base_model_path": getattr(args, 'base_model_path', None),
            "temperature": getattr(args, 'temperature', None),
            "quantization": getattr(args, 'quantization', None),
            "load_in_8bit": getattr(args, 'load_in_8bit', False),
            "load_in_4bit": getattr(args, 'load_in_4bit', False),
            "enable_gpu_monitor": getattr(args, 'enable_gpu_monitor', False),
            "gpu_monitor_interval": getattr(args, 'gpu_monitor_interval', 0.1),
            "fix_gpu_clock": getattr(args, 'fix_gpu_clock', False),
            "graphics_clock": getattr(args, 'graphics_clock', None),
            "memory_clock": getattr(args, 'memory_clock', None),
            "device_map": getattr(args, 'device_map', None),
            "seed": getattr(args, 'seed', None),
            "deterministic": getattr(args, 'deterministic', False),
        })
    
    result_data = {
        "experiment_info": experiment_info,
        "timing_stats": timing_stats,
        "accept_stats": accept_stats if accept_stats else {},
        "gpu_summary": gpu_summary,
        "raw_timing_data": timing_data,
        "raw_gpu_data": gpu_data
    }
    
    # JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    # (debug )
    if args and getattr(args, 'debug', False):
        print("\n" + "="*60)
        print("[target] === Performance Statistics Summary ===")
        print("="*60)
        print(f"Total processing steps: {step_count}steps")
        print(f"Total processing time: {timing_stats['total_time_seconds']:.4f}s")
        print(f"Average processing time: {timing_stats['avg_time_seconds']:.4f}s")
        print(f"Minimum processing time: {timing_stats['min_time_seconds']:.4f}s")
        print(f"Maximum processing time: {timing_stats['max_time_seconds']:.4f}s")
        
        # Accept length
        if accept_stats:
            print(f"\n--- Accept Length Statistics ---")
            print(f"Total accepted tokens: {accept_stats['total_accepted_tokens']}")
            print(f"average Accept Length: {accept_stats['avg_accept_length']:.2f}")
            print(f"minimum Accept Length: {accept_stats['min_accept_length']}")
            print(f"maximum Accept Length: {accept_stats['max_accept_length']}")
        
        # GPU 
        if gpu_summary:
            print("\n--- GPU usage statistics ---")
            for gpu_id, stats in gpu_summary.items():
                print(f"{gpu_id}:")
                print(f"  Memory used: {stats['memory_used_mb']['avg']:.1f}MB (average), {stats['memory_used_mb']['min']:.1f}MB (minimum), {stats['memory_used_mb']['max']:.1f}MB (maximum)")
                print(f"  Memory usage: {stats['memory_used_percent']['avg']:.1f}% (average), {stats['memory_used_percent']['min']:.1f}% (minimum), {stats['memory_used_percent']['max']:.1f}% (maximum)")
                print(f"  GPU utilization: {stats['utilization_percent']['avg']:.1f}% (average), {stats['utilization_percent']['min']:.1f}% (minimum), {stats['utilization_percent']['max']:.1f}% (maximum)")
                print(f"  Temperature: {stats['temperature_c']['avg']:.1f}°C (average), {stats['temperature_c']['min']:.1f}°C (minimum), {stats['temperature_c']['max']:.1f}°C (maximum)")
                
                if 'power_draw_w' in stats:
                    print(f"  Power draw: {stats['power_draw_w']['avg']:.1f}W (average), {stats['power_draw_w']['min']:.1f}W (minimum), {stats['power_draw_w']['max']:.1f}W (maximum)")
                if 'power_limit_w' in stats:
                    print(f"  Power limit: {stats['power_limit_w']:.1f}W")
                if 'power_usage_percent' in stats:
                    print(f"  Power usage: {stats['power_usage_percent']['avg']:.1f}% (average), {stats['power_usage_percent']['min']:.1f}% (minimum), {stats['power_usage_percent']['max']:.1f}% (maximum)")
                
                if 'energy' in stats:
                    energy = stats['energy']
                    print(f"  Total GPU energy: {energy['total_energy_kwh']:.6f} kWh ({energy['total_energy_joules']:.2f} J)")
                    print(f"  Average energy per call: {energy['avg_energy_per_call_kwh']:.6f} kWh ({energy['avg_energy_per_call_joules']:.2f} J)")
                    print(f"  Minimum energy per call: {energy['min_energy_per_call_joules']:.2f} J")
                    print(f"  Maximum energy per call: {energy['max_energy_per_call_joules']:.2f} J")
                    print(f"  Total call count: {energy['num_calls']}")
        else:
            print("\n--- GPU monitoring disabled ---")
        
        print("="*60)
        print(f"[target] Saved performance statistics: {output_file}")
        print("="*60)


class TargetWorker:
    def __init__(self, base_model_path: str, temperature: float, quantization: str = "4bit", int8_cpu_offload: bool = False, device_map: str = "auto", enable_gpu_monitor: bool = False, gpu_monitor_interval: float = 0.1, fix_gpu_clock: bool = False, graphics_clock: int = None, memory_clock: int = None, debug: bool = False, model_family: str = "llama2"):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        KVLlamaForCausalLM = get_kv_llama_class(base_model_path)

        self.temperature = temperature
        self.logits_processor = prepare_logits_processor(temperature=temperature) if temperature > 1e-5 else None
        self.enable_gpu_monitor = enable_gpu_monitor
        self.gpu_monitor = None
        self.debug = debug
        
        if self.enable_gpu_monitor:
            self.gpu_monitor = NVMLGPUMonitor(
                interval=gpu_monitor_interval,
                fix_gpu_clock=fix_gpu_clock,
                graphics_clock=graphics_clock,
                memory_clock=memory_clock,
                debug=debug
            )
            self.gpu_monitor.start_monitoring()
            if debug:
                print(
                    "[target] Starting GPU monitoring "
                    f"(backend={getattr(self.gpu_monitor, 'backend', 'unknown')}, "
                    f"interval={gpu_monitor_interval}s)"
                )

        quantization_config = None
        if quantization == "8bit":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=["lm_head"],
                llm_int8_enable_fp32_cpu_offload=int8_cpu_offload,
            )
        elif quantization == "4bit":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        self.base_model = KVLlamaForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device_map,
            quantization_config=quantization_config,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None,
        )

        (
            self.past_key_values,
            self.past_key_values_data,
            self.current_length_data,
        ) = initialize_past_key_values(self.base_model)

    def close(self):
        if self.gpu_monitor is not None:
            try:
                self.gpu_monitor.stop_monitoring()
            except Exception:
                pass

    def reset_kv(self):
        self.current_length_data.zero_()

    @torch.inference_mode()
    def handle_init(self, input_ids: List[int]) -> int:
        self.reset_kv()
        device = self.base_model.lm_head.weight.device
        ids = torch.tensor([input_ids], device=device, dtype=torch.long)
        outputs = self.base_model.model(
            input_ids=ids,
            position_ids=torch.arange(0, ids.shape[1], device=device, dtype=torch.long),
            past_key_values=self.past_key_values,
        )
        self.ar_next_position = int(ids.shape[1])
        logits = self.base_model.lm_head(outputs[0])
        if self.logits_processor is not None:
            last_logits = logits[:, -1]
            last_logits = self.logits_processor(None, last_logits)
            probs = torch.nn.functional.softmax(last_logits, dim=1)
            next_token = torch.multinomial(probs, 1).item()
        else:
            next_token = torch.argmax(logits[:, -1]).item()
        return next_token

    @torch.inference_mode()
    def handle_autoregressive_prefill(self, input_ids: List[int]) -> Tuple[int, bool, Optional[float], dict]:
        """Run prompt prefill and return the first greedy/token-sampled AR token."""
        self.ar_past_key_values = None
        device = self.base_model.lm_head.weight.device
        ids = torch.tensor([input_ids], device=device, dtype=torch.long)
        seq_len = int(ids.shape[1])
        if seq_len <= 0:
            raise RuntimeError("Empty input_ids are not supported.")
        position_ids = torch.arange(0, seq_len, device=device, dtype=torch.long)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        gpu_monitor_interval_start = time.time()

        outputs = self.base_model.model(
            input_ids=ids,
            position_ids=position_ids,
            past_key_values=self.ar_past_key_values,
            return_kv=True,
            is_draft=True,
        )
        self.ar_past_key_values = outputs[1]
        self.ar_next_position = seq_len
        logits = self.base_model.lm_head(outputs[0][:, -1, :])
        if self.logits_processor is not None:
            last_logits = self.logits_processor(None, logits)
            probs = torch.nn.functional.softmax(last_logits, dim=1)
            next_token = int(torch.multinomial(probs, 1).item())
        else:
            next_token = int(torch.argmax(logits, dim=-1).item())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()

        gpu_stats = self.gpu_monitor.get_stats_between(gpu_monitor_interval_start, end_time) if self.gpu_monitor else None
        target_energy_rate_per_sec = self._extract_target_energy_rate_per_sec(gpu_stats)
        timing_stats = {
            "total_time_seconds": float(end_time - start_time),
            "timestamp": datetime.now().isoformat(),
            "target_verification_time_ms": float((end_time - start_time) * 1000.0),
            "gpu_stats": gpu_stats,
            "gpu_monitor_backend": (
                str(getattr(self.gpu_monitor, "backend", "none"))
                if self.gpu_monitor is not None else "none"
            ),
            "gpu_monitor_long_running": bool(self.gpu_monitor is not None),
        }
        eos_reached = self.tokenizer.eos_token_id is not None and next_token == int(self.tokenizer.eos_token_id)
        return next_token, bool(eos_reached), target_energy_rate_per_sec, timing_stats

    @torch.inference_mode()
    def handle_autoregressive_next(self, token_id: int) -> Tuple[int, bool, Optional[float], dict]:
        """Advance target KV by one generated token and return the next AR token."""
        device = self.base_model.lm_head.weight.device
        token = torch.tensor([[int(token_id)]], device=device, dtype=torch.long)
        next_position = int(getattr(self, "ar_next_position", int(self.current_length_data[0].item())))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        gpu_monitor_interval_start = time.time()

        outputs = self.base_model.model(
            input_ids=token,
            position_ids=torch.tensor([next_position], device=device, dtype=torch.long),
            past_key_values=self.ar_past_key_values,
            return_kv=True,
            is_draft=True,
        )
        self.ar_past_key_values = outputs[1]
        self.ar_next_position = next_position + 1
        logits = self.base_model.lm_head(outputs[0][:, -1, :])
        if self.logits_processor is not None:
            last_logits = self.logits_processor(None, logits)
            probs = torch.nn.functional.softmax(last_logits, dim=1)
            next_token = int(torch.multinomial(probs, 1).item())
        else:
            next_token = int(torch.argmax(logits, dim=-1).item())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()

        gpu_stats = self.gpu_monitor.get_stats_between(gpu_monitor_interval_start, end_time) if self.gpu_monitor else None
        target_energy_rate_per_sec = self._extract_target_energy_rate_per_sec(gpu_stats)
        timing_stats = {
            "total_time_seconds": float(end_time - start_time),
            "timestamp": datetime.now().isoformat(),
            "target_verification_time_ms": float((end_time - start_time) * 1000.0),
            "gpu_stats": gpu_stats,
            "gpu_monitor_backend": (
                str(getattr(self.gpu_monitor, "backend", "none"))
                if self.gpu_monitor is not None else "none"
            ),
            "gpu_monitor_long_running": bool(self.gpu_monitor is not None),
        }
        eos_reached = self.tokenizer.eos_token_id is not None and next_token == int(self.tokenizer.eos_token_id)
        return next_token, bool(eos_reached), target_energy_rate_per_sec, timing_stats

    def _compute_logits(self, draft: torch.Tensor, position_ids: torch.Tensor, tree_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.base_model.model(
            input_ids=draft,
            tree_attention_mask=tree_mask,
            past_key_values=self.past_key_values,
            position_ids=position_ids,
        )
        logits = self.base_model.lm_head(outputs[0])
        return logits

    @staticmethod
    def _extract_target_energy_rate_per_sec(gpu_stats: dict) -> Optional[float]:
        """Reduce GPU monitor output to the only field the draft side needs."""
        if not isinstance(gpu_stats, dict) or not gpu_stats:
            return None
        total_power_w = 0.0
        found = False
        for gpu_entry in gpu_stats.values():
            if not isinstance(gpu_entry, dict):
                continue
            power_info = gpu_entry.get("power_draw_w", {})
            power_avg_w = power_info.get("avg") if isinstance(power_info, dict) else None
            if power_avg_w is None:
                continue
            try:
                power_avg_w = float(power_avg_w)
            except Exception:
                continue
            if power_avg_w <= 0:
                continue
            total_power_w += power_avg_w
            found = True
        if not found or total_power_w <= 0:
            return None
        return total_power_w / 3_600_000.0

    @torch.inference_mode()
    def handle_tree_step(
        self,
        draft_input_ids: List[int],
        draft_position_ids: List[int],
        tree_attention_mask: List[List[int]],
        parent: List[int],
    ) -> Tuple[List[int], int, int, bool, Optional[float], dict, List[int], int, int]:
        device = self.base_model.lm_head.weight.device

        torch.cuda.synchronize()
        start_time = time.time()

        gpu_monitor_interval_start = time.time()

        draft = torch.tensor([draft_input_ids], device=device, dtype=torch.long)
        position_ids = torch.tensor(draft_position_ids, device=device, dtype=torch.long)
        tree_mask = torch.tensor(tree_attention_mask, device=device, dtype=torch.int8)
        parent_t = torch.tensor(parent, device=device, dtype=torch.long)

        logits = self._compute_logits(draft, position_ids, tree_mask)

        # verify (without calling model.draft)
        if self.logits_processor is None:
            next_tok = torch.argmax(logits, dim=-1)
        else:
            proc_logits = self.logits_processor(None, logits)
            probabilities = torch.nn.functional.softmax(proc_logits, dim=-1)[0]
            next_tok = torch.multinomial(probabilities, 1).view(1, -1)
        next_tok = next_tok.to(draft.device)

        parent_work = torch.where(parent_t == torch.arange(parent_t.size(0), device=parent_t.device), -1, parent_t)
        parent_work = torch.cat([torch.tensor([0], device=parent_t.device), parent_work + 1], dim=-1).to(draft.device)

        correct = torch.where(draft[0] != next_tok[0][parent_work], 0, torch.ones(draft.size(1), device=draft.device))
        correct[0] = 1
        last_sum = torch.sum(correct)
        while True:
            correct = torch.where(correct[parent_work] == 0, 0, correct)
            if torch.sum(correct) == last_sum:
                break
            else:
                last_sum = torch.sum(correct)

        id_t_tensor = torch.argmax(correct * position_ids)
        id_t = int(id_t_tensor.item())
        best_candidate: List[int] = []
        best_ids: List[int] = []
        max_id = id_t
        parent_work[0] = -1
        while id_t != -1:
            token_val = int(draft[0][id_t].item())
            best_candidate.append(token_val)
            best_ids.append(id_t)
            id_t = int(parent_work[id_t].item())

        best_candidate.reverse()
        best_ids.reverse()
        next_token = int(next_tok[0][max_id].item())
        accept_length = len(best_candidate) - 1

        # KV cache ( current_length_data )
        start = int(self.current_length_data[0].item() - draft.size(1))
        base_input_len = int(self.current_length_data[0].item() - draft.size(1))  # input_ids
        
        select_indices = (torch.tensor(best_ids, device=device) + start).to(device)

        for data in self.past_key_values_data:
            tgt = data[..., select_indices.to(data.device), :]
            dst = data[..., start: start + tgt.shape[-2], :]
            dst.copy_(tgt, non_blocking=True)
        self.current_length_data.fill_(start + tgt.shape[-2])

        eos_id = self.tokenizer.eos_token_id
        # Check whether best_candidate reached EOS.
        # eval_autodraft_draft.py Line 766: tokenizer.eos_token_id in input_ids[0, input_len:].tolist()
        # best_candidate stores generated input_ids.
        eos_reached = eos_id in best_candidate if len(best_candidate) > 0 else False

        torch.cuda.synchronize()
        end_time = time.time()
        total_time = end_time - start_time

        # GPU : long-running NVML monitor verification
        gpu_stats = None
        if self.gpu_monitor:
            gpu_stats = self.gpu_monitor.get_stats_between(gpu_monitor_interval_start, end_time)
        target_energy_rate_per_sec = self._extract_target_energy_rate_per_sec(gpu_stats)

        verification_time_ms = (end_time - start_time) * 1000.0  # ms
        timing_stats = {
            "total_time_seconds": total_time,
            "timestamp": datetime.now().isoformat(),
            "target_verification_time_ms": verification_time_ms,  # (ms)
            "gpu_stats": gpu_stats,
            "gpu_monitor_backend": (
                str(getattr(self.gpu_monitor, "backend", "none"))
                if self.gpu_monitor is not None else "none"
            ),
            "gpu_monitor_long_running": bool(self.gpu_monitor is not None),
        }

        return best_candidate, accept_length, next_token, eos_reached, target_energy_rate_per_sec, timing_stats, best_ids, start, base_input_len

def serve(host: str, port: int, base_model_path: str, temperature: float, quantization: str, int8_cpu_offload: bool, device_map: str, enable_gpu_monitor: bool = False, gpu_monitor_interval: float = 0.1, output_file: str = None, fix_gpu_clock: bool = False, graphics_clock: int = None, memory_clock: int = None, args=None, debug: bool = False, model_family: str = "llama2", draft_model_path: str = None, preload_model_on_start: bool = False):
    # quantization args (experiment_info )
    if args:
        args.quantization = quantization
    configured_base_model_path = str(base_model_path or "").strip()
    current_base_model_path = ""
    current_quantization = str(quantization)
    current_device_map = str(device_map)
    current_int8_cpu_offload = bool(int8_cpu_offload)

    def _build_target_worker(model_path: str, quant_mode: str, dev_map: str, offload: bool) -> TargetWorker:
        return TargetWorker(
            base_model_path=model_path,
            temperature=temperature,
            quantization=quant_mode,
            int8_cpu_offload=offload,
            device_map=dev_map,
            enable_gpu_monitor=enable_gpu_monitor,
            gpu_monitor_interval=gpu_monitor_interval,
            fix_gpu_clock=fix_gpu_clock,
            graphics_clock=graphics_clock,
            memory_clock=memory_clock,
            debug=debug,
            model_family=model_family,
        )

    server_runner = None
    current_server_draft_model_path = ""
    current_server_draft_quantization = ""
    server_draft_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
    build_tree_with_next_token = None
    _select_proactive_path = None
    build_proactive_tree_from_path = None
    from evaluation.eval_autodraft_draft import (
        DraftRunner,
        _normalize_objective_metric,
        build_tree_with_next_token as _build_tree_with_next_token,
        _select_proactive_path as _select_proactive_path_fn,
        build_proactive_tree_from_path as _build_proactive_tree_from_path,
        profile_width_timing as _profile_width_timing,
    )
    
    def _build_server_only_runner(active_worker: TargetWorker, draft_path: str, quant_mode: str, dev_map: str, offload: bool):
        draft_path = str(draft_path or "").strip()
        if not draft_path:
            return None
        KVDraft = get_kv_llama_class(draft_path)
        draft_quantization_config = None
        if quant_mode == "8bit":
            from transformers import BitsAndBytesConfig
            draft_quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=["lm_head"],
                llm_int8_enable_fp32_cpu_offload=offload,
            )
        elif quant_mode == "4bit":
            from transformers import BitsAndBytesConfig
            draft_quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        draft_model = KVDraft.from_pretrained(
            draft_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=dev_map,
            quantization_config=draft_quantization_config,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None,
        )
        runner = DraftRunner(
            draft_model=draft_model,
            tokenizer=active_worker.tokenizer,
            debug=debug,
            enable_gpu_monitor=bool(getattr(args, "enable_gpu_monitor", False)),
            enable_cpu_monitor=False,
        )
        setattr(runner, "server_draft_quantization", str(quant_mode))
        setattr(runner, "server_draft_model_path", str(draft_path))
        return runner
    build_tree_with_next_token = _build_tree_with_next_token
    _select_proactive_path = _select_proactive_path_fn
    build_proactive_tree_from_path = _build_proactive_tree_from_path

    def _estimate_profile_energy_rate_per_sec(profile_data: Dict[str, Any]) -> float:
        """Estimate draft kWh/s from server-side draft profile power samples."""
        rates: List[float] = []
        if isinstance(profile_data, dict):
            for row in profile_data.values():
                if not isinstance(row, dict):
                    continue
                try:
                    power_w = float(row.get("gpu_power_avg_w", 0.0) or 0.0)
                except Exception:
                    power_w = 0.0
                if power_w > 0:
                    rates.append(power_w / 3_600_000.0)
        return float(sum(rates) / len(rates)) if rates else 0.0
    
    worker = None
    if preload_model_on_start:
        if not configured_base_model_path:
            raise ValueError("--base-model-path is required when --eager-load is enabled")
        worker = _build_target_worker(
            model_path=configured_base_model_path,
            quant_mode=current_quantization,
            dev_map=current_device_map,
            offload=current_int8_cpu_offload,
        )
        current_base_model_path = configured_base_model_path
    else:
        print("[target] lazy load mode enabled: listening without preloaded model")
    
    timing_data = []
    gpu_data = []
    accept_lengths = []  # Accept length
    step_count = 0
    def _target_log(message: str):
        ts = time.strftime("%H:%M:%S")
        print(f"[target][{ts}] {message}", flush=True)

    def _log_cuda_memory_state(stage: str):
        if not torch.cuda.is_available():
            _target_log(f"{stage}: cuda unavailable")
            return
        try:
            allocated = int(torch.cuda.memory_allocated())
            reserved = int(torch.cuda.memory_reserved())
            max_allocated = int(torch.cuda.max_memory_allocated())
            max_reserved = int(torch.cuda.max_memory_reserved())
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            used_bytes = int(total_bytes - free_bytes)
            _target_log(
                f"{stage}: cuda_mem "
                f"allocated={allocated}B reserved={reserved}B "
                f"used_by_driver={used_bytes}B free={int(free_bytes)}B total={int(total_bytes)}B "
                f"peak_allocated={max_allocated}B peak_reserved={max_reserved}B"
            )
        except Exception as e:
            _target_log(f"{stage}: cuda_mem_log failed: {e}")

    def _dispose_runtime_object(obj: Any):
        if obj is None:
            return
        try:
            close_fn = getattr(obj, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            pass
        # Explicitly break strong references to large tensors/models before gc.
        for attr_name in (
            "base_model",
            "tokenizer",
            "past_key_values",
            "past_key_values_data",
            "current_length_data",
            "draft_model",
            "draft_stable_kv",
            "proactive_kv",
            "gpu_monitor",
            "cpu_power_monitor",
            "logits_processor",
        ):
            try:
                if hasattr(obj, attr_name):
                    setattr(obj, attr_name, None)
            except Exception:
                pass
    profile_session_lock = threading.Lock()
    profile_session: Dict[str, Any] = {
        "running": False,
        "pending": False,
        "status": "idle",
        "error": None,
        "output_file": None,
        "timing_summary": None,
        "started_at": None,
        "finished_at": None,
        "total_combinations": 0,
        "completed_combinations": 0,
        "current_combination": None,
    }
    profile_session_thread: Optional[threading.Thread] = None

    def _sanitize_key_component(component: str) -> str:
        return str(component).strip().lower().replace("/", "_").replace(" ", "_")

    def _parse_int_list(v: Any, default_values: List[int]) -> List[int]:
        if v is None:
            return list(default_values)
        if isinstance(v, list):
            out = []
            for item in v:
                try:
                    out.append(int(item))
                except Exception:
                    continue
            return out or list(default_values)
        text = str(v).strip()
        if not text:
            return list(default_values)
        out = []
        for tok in text.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.append(int(tok))
            except Exception:
                continue
        return out or list(default_values)

    def _build_target_profile_output_path(model_path: str, quant_mode: str) -> str:
        model_name = model_path.split("/")[-1] if "/" in model_path else model_path
        script_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(script_dir)
        profile_dir = os.path.join(_resolve_data_root(parent_dir), "profile")
        os.makedirs(profile_dir, exist_ok=True)
        server_name = _sanitize_key_component(getattr(args, "server_name", "target") or "target")
        tq_name = _sanitize_key_component(str(quant_mode or "8bit").lower())
        return os.path.join(
            profile_dir,
            f"profile_target_{server_name}_{_sanitize_key_component(model_name)}_tq-{tq_name}.json",
        )

    def _build_server_draft_profile_output_path(
        model_path: str,
        quant_mode: str,
        server_name_value: Optional[str] = None,
    ) -> str:
        model_name = model_path.split("/")[-1] if "/" in model_path else model_path
        script_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(script_dir)
        profile_dir = os.path.join(_resolve_data_root(parent_dir), "profile")
        os.makedirs(profile_dir, exist_ok=True)
        raw_server_name = (
            server_name_value
            if server_name_value is not None
            else getattr(args, "server_name", "target")
        )
        server_name = _sanitize_key_component(raw_server_name or "target")
        dq_name = _sanitize_key_component(str(quant_mode or "none").lower())
        return os.path.join(
            profile_dir,
            f"profile_draft_{server_name}_{_sanitize_key_component(model_name)}_dq-{dq_name}.json",
        )

    def _parse_width_csv(v: Any, default_widths: List[int]) -> List[int]:
        # width=1 can trip Tree/profile indexing and poison the CUDA context with
        # a device-side assert. Server-side draft profiling only needs widths >= 2.
        min_safe_width = 2
        if v is None:
            return [int(w) for w in default_widths if int(w) >= min_safe_width]
        if isinstance(v, list):
            out = []
            for item in v:
                try:
                    width = int(item)
                    if width >= min_safe_width:
                        out.append(width)
                except Exception:
                    continue
            return sorted(set(out)) if out else [int(w) for w in default_widths if int(w) >= min_safe_width]
        text = str(v).strip()
        if not text:
            return [int(w) for w in default_widths if int(w) >= min_safe_width]
        out = []
        for tok in text.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                width = int(tok)
                if width >= min_safe_width:
                    out.append(width)
            except Exception:
                continue
        return sorted(set(out)) if out else [int(w) for w in default_widths if int(w) >= min_safe_width]

    def _ensure_server_draft_executor() -> concurrent.futures.ThreadPoolExecutor:
        nonlocal server_draft_executor
        if server_draft_executor is None:
            server_draft_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="server_draft_worker")
        return server_draft_executor

    def _shutdown_server_draft_executor() -> None:
        nonlocal server_draft_executor
        if server_draft_executor is not None:
            try:
                server_draft_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            server_draft_executor = None

    def _ensure_server_draft_profile(
        active_runner,
        active_worker: TargetWorker,
        *,
        quant_mode: str,
        bench_name: str,
        question_file: str,
        force_refresh: bool,
        model_calls_per_count: int,
        width_list_csv: str,
    ) -> Dict[str, Any]:
        profile_prepare_start = time.time()
        effective_server_name = _sanitize_key_component(
            str(getattr(active_runner, "server_name_for_profile", "") or getattr(args, "server_name", "target") or "target")
        )
        profile_file = _build_server_draft_profile_output_path(
            draft_model_path or "",
            quant_mode,
            server_name_value=effective_server_name,
        )
        source = "missing"
        profile_data = None
        resolved_question_file = str(question_file or "").strip()
        if resolved_question_file and (not os.path.exists(resolved_question_file)):
            _target_log(
                "[server_draft_profile][warn] question_file not found on target; "
                f"fallback to bench default path. requested={resolved_question_file}"
            )
            resolved_question_file = ""
        if (not force_refresh) and os.path.exists(profile_file):
            try:
                with open(profile_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict) and loaded:
                    profile_data = loaded
                    source = "loaded"
            except Exception as e:
                _target_log(f"[server_draft_profile][warn] failed to load cached profile: {e}")
        if profile_data is None:
            _target_log(
                f"[server_draft_profile] generate start force_refresh={bool(force_refresh)} "
                f"calls={int(max(1, int(model_calls_per_count)))} width_list={width_list_csv}"
            )
            widths = _parse_width_csv(width_list_csv, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
            original_gpu_monitor = getattr(active_runner, "gpu_monitor", None)
            original_enable_gpu_monitor = bool(getattr(active_runner, "enable_gpu_monitor", False))
            requires_draft_energy = bool(
                hasattr(active_runner, "uses_draft_energy_profile")
                and active_runner.uses_draft_energy_profile()
            )
            try:
                # Server-only target-side draft profiling is latency-only unless
                # the objective explicitly needs draft energy samples.
                if not requires_draft_energy:
                    setattr(active_runner, "gpu_monitor", None)
                    setattr(active_runner, "enable_gpu_monitor", False)
                generated = _profile_width_timing(
                    runner=active_runner,
                    tokenizer=active_worker.tokenizer,
                    max_depth=10,
                    draft_model_path=draft_model_path,
                    device_name=str(effective_server_name),
                    draft_quantization=str(quant_mode or "none"),
                    question_file=resolved_question_file,
                    bench_name=str(bench_name or "mt_bench"),
                    width_list=widths,
                    target_model_calls_per_width=int(max(1, int(model_calls_per_count))),
                    fixed_depth=False,
                    force_refresh=bool(force_refresh),
                )
            finally:
                setattr(active_runner, "gpu_monitor", original_gpu_monitor)
                setattr(active_runner, "enable_gpu_monitor", original_enable_gpu_monitor)
            if isinstance(generated, dict) and generated:
                profile_data = generated
                source = "generated"
            elif os.path.exists(profile_file):
                with open(profile_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict) and loaded:
                    profile_data = loaded
                    source = "loaded_after_generate"
        if not isinstance(profile_data, dict) or (not profile_data):
            raise RuntimeError("server draft profile is empty after ensure")
        active_runner.profile_data = profile_data
        setattr(active_runner, "draft_profile_file", profile_file)
        return {
            "file": str(profile_file),
            "source": str(source),
            "rows": int(len(profile_data)),
            "prepare_wall_sec": float(max(0.0, time.time() - profile_prepare_start)),
        }

    def _set_profile_state(**kwargs):
        with profile_session_lock:
            profile_session.update(kwargs)

    def _derive_tree_depth_from_parent(parent: List[int]) -> int:
        if not parent:
            return 0
        max_depth = 0
        for i in range(len(parent)):
            depth = 1
            cur = i
            seen = set()
            while True:
                if cur in seen:
                    break
                seen.add(cur)
                p = int(parent[cur])
                if p == cur or p < 0 or p >= len(parent):
                    break
                depth += 1
                cur = p
            if depth > max_depth:
                max_depth = depth
        return int(max_depth)

    def _slice_profile_base_tree(base_tree: Dict[str, Any], desired_nodes: int) -> Dict[str, Any]:
        desired_nodes = max(1, int(desired_nodes))
        base_draft_ids = [int(x) for x in base_tree.get("draft_input_ids", [])]
        base_draft_pos = [int(x) for x in base_tree.get("draft_position_ids", [])]
        base_mask = base_tree.get("tree_attention_mask", [])
        base_parent = [int(x) for x in base_tree.get("parent", [])]
        keep = desired_nodes + 1
        if (
            len(base_draft_ids) < keep
            or len(base_draft_pos) < keep
            or (not isinstance(base_mask, list))
            or len(base_mask) < keep
            or len(base_parent) < desired_nodes
        ):
            raise RuntimeError(
                f"base tree is smaller than requested nnodes={desired_nodes} "
                f"(ids={len(base_draft_ids)}, pos={len(base_draft_pos)}, parent={len(base_parent)})"
            )
        sliced_parent = [int(x) for x in base_parent[:desired_nodes]]
        parent_valid = all(0 <= p < desired_nodes for p in sliced_parent)
        if not parent_valid:
            raise RuntimeError(f"invalid parent closure while slicing nnodes={desired_nodes}")
        sliced_mask = [row[:keep] for row in base_mask[:keep]]
        return {
            "draft_input_ids": base_draft_ids[:keep],
            "draft_position_ids": base_draft_pos[:keep],
            "tree_attention_mask": sliced_mask,
            "parent": sliced_parent,
            "tree_depth": _derive_tree_depth_from_parent(sliced_parent),
        }

    def _run_target_profile_session(session_args: Dict[str, Any]):
        nonlocal worker
        try:
            if worker is None:
                raise RuntimeError("target model is not loaded. Call reload_model first.")

            node_list = _parse_int_list(session_args.get("node_list"), [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            runs_per_combination = max(1, int(session_args.get("runs_per_combination", 10)))
            profile_warmup_runs = max(
                0, int(session_args.get("profile_warmup_runs", PROFILE_WARMUP_RUNS))
            )
            profile_burnin_runs = max(
                0, int(session_args.get("profile_burnin_runs", PROFILE_BURNIN_RUNS))
            )
            profile_deterministic = bool(session_args.get("profile_deterministic", False))
            base_input_ids_raw = session_args.get("base_input_ids", [])
            base_tree = session_args.get("base_tree", {})
            if not isinstance(base_input_ids_raw, list) or not base_input_ids_raw:
                raise RuntimeError("profile_target_start requires base_input_ids")
            if not isinstance(base_tree, dict) or not base_tree:
                raise RuntimeError("profile_target_start requires base_tree payload")
            base_input_ids = [int(x) for x in base_input_ids_raw]
            base_draft_time_sec = max(0.0, float(base_tree.get("draft_profile_time_sec", 0.0) or 0.0))
            base_tree_next_token = int(base_tree.get("next_token", -1))
            fixed_width = int(base_tree.get("width", TARGET_PROFILE_FIXED_WIDTH) or TARGET_PROFILE_FIXED_WIDTH)
            fixed_depth = int(base_tree.get("depth", TARGET_PROFILE_FIXED_DEPTH) or TARGET_PROFILE_FIXED_DEPTH)

            max_nodes_cap = int(TARGET_PROFILE_FIXED_WIDTH * TARGET_PROFILE_FIXED_DEPTH)
            valid_nodes = sorted({int(max_nnodes) for max_nnodes in node_list if int(max_nnodes) > 0 and int(max_nnodes) <= max_nodes_cap})
            if not valid_nodes:
                raise RuntimeError("no valid nnodes combinations to profile")
            _target_log(
                "[profile] start requested: "
                f"fixed_width={fixed_width} fixed_depth={fixed_depth} "
                f"node={len(node_list)} valid={len(valid_nodes)} runs={runs_per_combination} "
                f"warmup={profile_warmup_runs} burnin={profile_burnin_runs} "
                f"deterministic={profile_deterministic} base_tree_draft_time_sec={base_draft_time_sec:.6f}"
            )

            total_samples = int(len(valid_nodes) * runs_per_combination)
            _set_profile_state(
                running=True,
                status="running",
                pending=False,
                error=None,
                started_at=time.time(),
                finished_at=None,
                timing_summary=None,
                total_combinations=total_samples,
                completed_combinations=0,
                current_combination=None,
            )

            wall_time_buckets: Dict[int, List[float]] = {}
            verification_time_buckets: Dict[int, List[float]] = {}
            server_draft_time_buckets: Dict[int, List[float]] = {}
            total_runs_per_combo = int(profile_warmup_runs + profile_burnin_runs + runs_per_combination)
            for max_nnodes in valid_nodes:
                sliced_tree = _slice_profile_base_tree(base_tree, int(max_nnodes))
                for run_idx in range(total_runs_per_combo):
                    if run_idx < profile_warmup_runs:
                        profile_phase = "warmup"
                    elif run_idx < (profile_warmup_runs + profile_burnin_runs):
                        profile_phase = "burnin"
                    else:
                        profile_phase = "measure"
                    worker.reset_kv()
                    run_next_token = int(worker.handle_init(base_input_ids))
                    if base_tree_next_token >= 0 and run_next_token != base_tree_next_token:
                        _target_log(
                            f"[profile][warn] init next_token mismatch: tree={base_tree_next_token} run={run_next_token}"
                        )
                    _target_log(
                        "[profile] cached_tree_step "
                        f"phase={profile_phase} width={fixed_width} depth={fixed_depth} nnodes={max_nnodes}"
                    )
                    _set_profile_state(
                        current_combination={
                            "width": int(fixed_width),
                            "depth": int(fixed_depth),
                            "max_nnodes": int(max_nnodes),
                            "phase": profile_phase,
                            "run_index": int(run_idx + 1),
                        }
                    )
                    (
                        _best_candidate,
                        _accept_length,
                        _next_token,
                        _eos_reached,
                        _target_energy_rate_per_sec,
                        timing_stats,
                        _best_ids,
                        _kv_start,
                        _base_input_len,
                    ) = worker.handle_tree_step(
                        sliced_tree["draft_input_ids"],
                        sliced_tree["draft_position_ids"],
                        sliced_tree["tree_attention_mask"],
                        sliced_tree["parent"],
                    )
                    verification_sec = max(
                        0.0,
                        float(timing_stats.get("target_verification_time_ms", 0.0) or 0.0) / 1000.0,
                    )
                    server_draft_sec = float(base_draft_time_sec)
                    wall_sec = verification_sec + server_draft_sec
                    if profile_phase == "measure":
                        wall_time_buckets.setdefault(int(max_nnodes), []).append(wall_sec)
                        verification_time_buckets.setdefault(int(max_nnodes), []).append(verification_sec)
                        server_draft_time_buckets.setdefault(int(max_nnodes), []).append(server_draft_sec)
                        _set_profile_state(
                            completed_combinations=int(sum(len(v) for v in wall_time_buckets.values()))
                        )

            out_data = {}
            all_wall_vals: List[float] = []
            all_verification_vals: List[float] = []
            all_server_draft_vals: List[float] = []
            for max_nnodes, wall_vals in wall_time_buckets.items():
                if not wall_vals:
                    continue
                verification_vals = verification_time_buckets.get(int(max_nnodes), [])
                server_draft_vals = server_draft_time_buckets.get(int(max_nnodes), [])
                if len(verification_vals) != len(wall_vals) or len(server_draft_vals) != len(wall_vals):
                    raise RuntimeError("target profile timing bucket size mismatch")
                all_wall_vals.extend(wall_vals)
                all_verification_vals.extend(verification_vals)
                all_server_draft_vals.extend(server_draft_vals)
                key = f"nnodes_{int(max_nnodes)}"
                out_data[key] = {
                    "max_nnodes": int(max_nnodes),
                    "count": len(wall_vals),
                    # nodes verification-only
                    "total_time_ms": float(sum(verification_vals) * 1000.0),
                    "avg_time_ms": float((sum(verification_vals) / len(verification_vals)) * 1000.0),
                    "min_time_ms": float(min(verification_vals) * 1000.0),
                    "max_time_ms": float(max(verification_vals) * 1000.0),
                    "total_wall_time_ms": float(sum(wall_vals) * 1000.0),
                    "avg_wall_time_ms": float((sum(wall_vals) / len(wall_vals)) * 1000.0),
                    "min_wall_time_ms": float(min(wall_vals) * 1000.0),
                    "max_wall_time_ms": float(max(wall_vals) * 1000.0),
                    "total_verification_time_ms": float(sum(verification_vals) * 1000.0),
                    "avg_verification_time_ms": float((sum(verification_vals) / len(verification_vals)) * 1000.0),
                    "min_verification_time_ms": float(min(verification_vals) * 1000.0),
                    "max_verification_time_ms": float(max(verification_vals) * 1000.0),
                    "total_server_draft_time_ms": float(sum(server_draft_vals) * 1000.0),
                    "avg_server_draft_time_ms": float((sum(server_draft_vals) / len(server_draft_vals)) * 1000.0),
                    "min_server_draft_time_ms": float(min(server_draft_vals) * 1000.0),
                    "max_server_draft_time_ms": float(max(server_draft_vals) * 1000.0),
                    "tree_source": "draft_cached_base_tree",
                    "base_tree_build_time_ms": float(base_draft_time_sec * 1000.0),
                }
            if not out_data:
                raise RuntimeError("target profile session produced no timing samples")
            timing_summary = {
                "target_profile_wall_sec": float(sum(all_wall_vals)),
                "target_verification_profile_sec": float(sum(all_verification_vals)),
                "target_server_draft_profile_sec": float(sum(all_server_draft_vals)),
                "sample_count": int(len(all_wall_vals)),
                "tree_source": "draft_cached_base_tree",
                "base_tree_build_time_sec": float(base_draft_time_sec),
            }

            out_file = _build_target_profile_output_path(current_base_model_path, current_quantization)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(out_data, f, indent=2)
            _set_profile_state(
                running=False,
                pending=False,
                status="done",
                error=None,
                output_file=out_file,
                timing_summary=timing_summary,
                finished_at=time.time(),
                current_combination=None,
            )
            _target_log(f"[profile] completed: {out_file}")
        except Exception as e:
            _set_profile_state(
                running=False,
                pending=False,
                status="error",
                error=repr(e),
                output_file=None,
                timing_summary=None,
                finished_at=time.time(),
                current_combination=None,
            )
            _target_log(f"[profile][error] {e}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(1)
        # if debug:
        print(f"[target] listening on {host}:{port}")
        server_running = True
        while server_running:
            conn, addr = s.accept()
            with conn:
                # if debug:
                print(f"[target] connection from {addr}")
                while True:
                    try:
                        msg = recv_json(conn)
                    except ConnectionError:
                        _target_log(f"client disconnected: {addr}")
                        # (output_file None )
                        if timing_data or gpu_data:
                            save_performance_stats(timing_data, gpu_data, output_file, step_count, args, accept_lengths)
                        break
                    # Draft Target : Draft
                    target_recv_end_time = time.time()
                    mtype = msg.get("type")
                    _target_log(f"rpc recv from {addr}: type={mtype}")
                    if mtype == "status":
                        with profile_session_lock:
                            profile_snapshot = dict(profile_session)
                        send_json(conn, {
                            "type": "status_ok",
                            "loaded_model": current_base_model_path if worker is not None else "",
                            "configured_model": configured_base_model_path,
                            "loaded": bool(worker is not None),
                            "quantization": current_quantization,
                            "device_map": current_device_map,
                            "int8_cpu_offload": current_int8_cpu_offload,
                            "target_profile_session": profile_snapshot,
                        })
                        continue
                    elif mtype == "profile_target_start":
                        if not bool(getattr(args, "enable_auto_target_profile", True)):
                            send_json(conn, {"type": "error", "message": "auto target profiling is disabled on target"})
                            continue
                        with profile_session_lock:
                            if profile_session.get("running", False) or profile_session.get("pending", False):
                                send_json(conn, {
                                    "type": "profile_target_started",
                                    "already_running": True,
                                    "status": dict(profile_session),
                                })
                                continue
                        if worker is None:
                            send_json(conn, {"type": "error", "message": "target model is not loaded. Call reload_model first."})
                            continue
                        force_refresh = bool(msg.get("force_refresh", False))
                        expected_out = _build_target_profile_output_path(current_base_model_path, current_quantization)
                        if (not force_refresh) and os.path.exists(expected_out):
                            _set_profile_state(
                                running=False,
                                pending=False,
                                status="done",
                                error=None,
                                output_file=expected_out,
                                timing_summary=None,
                                started_at=time.time(),
                                finished_at=time.time(),
                            )
                            send_json(conn, {
                                "type": "profile_target_started",
                                "already_running": False,
                                "status": dict(profile_session),
                                "skipped_existing": True,
                            })
                            continue
                        _set_profile_state(
                            running=True,
                            pending=True,
                            status="starting",
                            error=None,
                            output_file=None,
                            timing_summary=None,
                            started_at=time.time(),
                            finished_at=None,
                            total_combinations=0,
                            completed_combinations=0,
                            current_combination=None,
                        )
                        profile_session_thread = threading.Thread(
                            target=_run_target_profile_session,
                            args=(dict(msg),),
                            daemon=True,
                        )
                        try:
                            profile_session_thread.start()
                        except Exception as e:
                            _set_profile_state(
                                running=False,
                                pending=False,
                                status="error",
                                error=f"profile thread start failed: {e}",
                                output_file=None,
                                timing_summary=None,
                                finished_at=time.time(),
                            )
                            send_json(conn, {"type": "error", "message": f"failed to start profile thread: {e}"})
                            continue
                        send_json(conn, {
                            "type": "profile_target_started",
                            "already_running": False,
                            "status": dict(profile_session),
                        })
                        continue
                    elif mtype == "profile_target_status":
                        with profile_session_lock:
                            snapshot = dict(profile_session)
                        send_json(conn, {"type": "profile_target_status_ok", "status": snapshot})
                        continue
                    elif mtype == "profile_target_get_result":
                        with profile_session_lock:
                            snapshot = dict(profile_session)
                        profile_data = None
                        error_message = None
                        if str(snapshot.get("status", "")) == "done":
                            output_path = snapshot.get("output_file")
                            if output_path and os.path.exists(str(output_path)):
                                try:
                                    with open(str(output_path), "r", encoding="utf-8") as f:
                                        loaded = json.load(f)
                                    if isinstance(loaded, dict):
                                        profile_data = loaded
                                    else:
                                        error_message = "profile output JSON is not an object"
                                except Exception as e:
                                    error_message = f"failed to load output_file for transfer: {e}"
                                    _target_log(f"[profile][warn] {error_message}")
                            else:
                                error_message = f"profile output file missing: {output_path}"
                        elif str(snapshot.get("status", "")) == "error":
                            error_message = str(snapshot.get("error") or "target profile session failed")
                        else:
                            error_message = f"profile session is not done (status={snapshot.get('status')})"
                        success = bool(isinstance(profile_data, dict) and profile_data)
                        send_json(
                            conn,
                            {
                                "type": "profile_target_result_ok",
                                "success": success,
                                "error": error_message,
                                "status": snapshot,
                                "profile_data": profile_data,
                                "timing_summary": snapshot.get("timing_summary"),
                            },
                        )
                        continue
                    elif mtype == "reload_model":
                        requested_model = str(msg.get("base_model_path", "")).strip()
                        requested_quant = str(msg.get("quantization", current_quantization)).strip().lower()
                        if requested_quant not in {"none", "4bit", "8bit"}:
                            requested_quant = current_quantization
                        requested_device_map = str(msg.get("device_map", current_device_map)).strip() or current_device_map
                        requested_int8_offload = bool(msg.get("int8_cpu_offload", current_int8_cpu_offload))
                        if not requested_model:
                            send_json(conn, {"type": "error", "message": "reload_model requires base_model_path"})
                            continue
                        # Fast path: already loaded with same runtime options.
                        if (
                            worker is not None
                            and
                            requested_model == current_base_model_path
                            and requested_quant == current_quantization
                            and requested_device_map == current_device_map
                            and requested_int8_offload == current_int8_cpu_offload
                        ):
                            worker.reset_kv()
                            _target_log(
                                f"reload fast-path model={current_base_model_path} "
                                f"quant={current_quantization} device_map={current_device_map}"
                            )
                            send_json(conn, {
                                "type": "reload_ok",
                                "loaded_model": current_base_model_path,
                                "changed": False,
                            })
                            continue
                        prev_worker = worker
                        try:
                            new_worker = _build_target_worker(
                                model_path=requested_model,
                                quant_mode=requested_quant,
                                dev_map=requested_device_map,
                                offload=requested_int8_offload,
                            )
                            worker = new_worker
                            current_base_model_path = requested_model
                            current_quantization = requested_quant
                            current_device_map = requested_device_map
                            current_int8_cpu_offload = requested_int8_offload
                            _target_log(
                                f"reload success model={current_base_model_path} quant={current_quantization} "
                                f"device_map={current_device_map} int8_cpu_offload={current_int8_cpu_offload}"
                            )
                            # Keep server-only draft runner lazy to avoid extra VRAM pressure
                            # during target reload/profile flows.
                            server_runner = None
                            _shutdown_server_draft_executor()
                            try:
                                _dispose_runtime_object(prev_worker)
                                del prev_worker
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except Exception:
                                pass
                            send_json(conn, {
                                "type": "reload_ok",
                                "loaded_model": current_base_model_path,
                                "quantization": current_quantization,
                                "device_map": current_device_map,
                                "int8_cpu_offload": current_int8_cpu_offload,
                                "changed": True,
                            })
                        except Exception as e:
                            # Keep previous worker alive on reload failure.
                            send_json(
                                conn,
                                {
                                    "type": "error",
                                    "message": f"reload_model failed: {e}",
                                },
                            )
                        continue
                    elif mtype == "unload_model":
                        try:
                            prev_worker = worker
                            prev_server_runner = server_runner
                            worker = None
                            server_runner = None
                            _shutdown_server_draft_executor()
                            current_base_model_path = ""
                            _log_cuda_memory_state("unload/before_cleanup")
                            _dispose_runtime_object(prev_server_runner)
                            _dispose_runtime_object(prev_worker)
                            try:
                                del prev_server_runner
                                del prev_worker
                            except Exception:
                                pass
                            # Force Python-side reference collection first.
                            gc.collect()
                            gc.collect()
                            try:
                                if torch.cuda.is_available():
                                    try:
                                        torch.cuda.synchronize()
                                    except Exception:
                                        pass
                                    torch.cuda.empty_cache()
                                    try:
                                        torch.cuda.ipc_collect()
                                    except Exception:
                                        pass
                                    # One more GC/cache sweep after allocator compaction.
                                    gc.collect()
                                    torch.cuda.empty_cache()
                            except Exception:
                                pass
                            # Give allocator/driver a brief chance to settle.
                            time.sleep(0.25)
                            _log_cuda_memory_state("unload/after_cleanup")
                            send_json(conn, {"type": "unload_ok"})
                            _target_log("unload_model success")
                        except Exception as e:
                            send_json(conn, {"type": "error", "message": f"unload_model failed: {e}"})
                            _target_log(f"unload_model failed: {e}")
                        continue
                    if mtype in {"init", "tree_step", "server_only_init", "server_only_ar_init", "server_only_ar_turn"}:
                        with profile_session_lock:
                            if profile_session.get("running", False):
                                send_json(conn, {
                                    "type": "error",
                                    "message": "target profiling in progress; inference RPCs are temporarily blocked",
                                })
                                continue
                    if mtype == "init":
                        if worker is None:
                            send_json(conn, {"type": "error", "message": "target model is not loaded. Call reload_model first."})
                            continue
                        if worker.gpu_monitor is not None:
                            try:
                                worker.gpu_monitor.reset_data()
                            except Exception:
                                pass
                        input_ids = msg["input_ids"]
                        _target_log(f"init request len(input_ids)={len(input_ids)}")
                        next_token = worker.handle_init(input_ids)
                        target_send_start_time = time.time()
                        send_json(conn, {
                            "type": "init_ok", 
                            "next_token": next_token,
                            "target_recv_end_time": target_recv_end_time,
                            "target_send_start_time": target_send_start_time,
                        })
                    elif mtype == "tree_step":                        
                        if worker is None:
                            send_json(conn, {"type": "error", "message": "target model is not loaded. Call reload_model first."})
                            continue
                        _target_log(
                            "tree_step request "
                            f"draft_tokens={len(msg.get('draft_input_ids', []))} "
                            f"tree_nodes={len(msg.get('parent', []))}"
                        )
                        best_candidate, accept_length, next_token, eos_reached, target_energy_rate_per_sec, timing_stats, best_ids, kv_start, base_input_len = worker.handle_tree_step(
                            msg["draft_input_ids"],
                            msg["draft_position_ids"],
                            msg["tree_attention_mask"],
                            msg["parent"],
                        )
                        
                        step_count += 1
                        timing_data.append(timing_stats)
                        accept_lengths.append(accept_length)  # Accept length
                        # Keep detailed GPU stats only for target-side local summaries.
                        if worker.gpu_monitor:
                            current_gpu_stats = timing_stats.get("gpu_stats")
                            if current_gpu_stats:
                                gpu_data.append({
                                    "step": step_count,
                                    "timestamp": timing_stats["timestamp"],
                                    "gpu_stats": current_gpu_stats,
                                    "monitor_call_count": worker.gpu_monitor.monitor_call_count if worker.gpu_monitor else None
                                })
                        
                        # Target Draft : Target
                        target_send_start_time = time.time()
                        send_json(conn, {
                            "type": "verify_result",
                            "accepted_tokens": best_candidate,
                            "accepted_plus_next_tokens": list(best_candidate) + [int(next_token)],
                            "accept_length": accept_length,
                            "next_token": next_token,
                            "eos_reached": eos_reached,
                            "best_ids": best_ids,
                            "base_input_len": base_input_len,
                            "kv_start": kv_start,  # KV cache
                            "target_recv_end_time": target_recv_end_time,  # Target
                            "target_send_start_time": target_send_start_time,  # Target
                            "target_verification_time_ms": timing_stats.get("target_verification_time_ms", None),  # Target (ms)
                            "target_energy_rate_per_sec": target_energy_rate_per_sec,
                        })
                    elif mtype == "server_only_ar_turn":
                        _target_log("server_only_ar_turn start")
                        if worker is None:
                            send_json(conn, {"type": "error", "message": "target model is not loaded. Call reload_model first."})
                            continue
                        if worker.gpu_monitor is not None:
                            try:
                                worker.gpu_monitor.reset_data()
                            except Exception:
                                pass
                        input_ids = msg["input_ids"]
                        max_new_tokens = int(max(1, int(msg.get("max_new_tokens", 1024))))
                        send_json(conn, {"type": "server_only_ok", "mode": "autoregressive_turn_model_forward"})

                        generated_tokens: List[int] = []
                        total_target_ms = 0.0
                        energy_rate_weighted_sum = 0.0
                        energy_rate_weight_total = 0.0
                        eos_reached = False
                        next_token = None
                        final_timing_stats = {}
                        target_energy_rate_per_sec = None
                        try:
                            _target_log(
                                "server_only_ar_turn prefill begin "
                                f"prompt_tokens={len(input_ids)} max_new_tokens={max_new_tokens}"
                            )
                            next_token, eos_reached, target_energy_rate_per_sec, timing_stats = worker.handle_autoregressive_prefill(input_ids)
                            final_timing_stats = timing_stats
                            step_ms = float(timing_stats.get("target_verification_time_ms", 0.0) or 0.0)
                            total_target_ms += step_ms
                            if target_energy_rate_per_sec is not None and step_ms > 0:
                                energy_rate_weighted_sum += float(target_energy_rate_per_sec) * step_ms
                                energy_rate_weight_total += step_ms
                            generated_tokens.append(int(next_token))
                            _target_log(
                                "server_only_ar_turn prefill done "
                                f"first_token={int(next_token)} eos={bool(eos_reached)} "
                                f"prefill_ms={step_ms:.3f}"
                            )
                            while (not bool(eos_reached)) and len(generated_tokens) < max_new_tokens:
                                prev_token = int(generated_tokens[-1])
                                next_token, eos_reached, target_energy_rate_per_sec, timing_stats = worker.handle_autoregressive_next(prev_token)
                                final_timing_stats = timing_stats
                                step_ms = float(timing_stats.get("target_verification_time_ms", 0.0) or 0.0)
                                total_target_ms += step_ms
                                if target_energy_rate_per_sec is not None and step_ms > 0:
                                    energy_rate_weighted_sum += float(target_energy_rate_per_sec) * step_ms
                                    energy_rate_weight_total += step_ms
                                generated_tokens.append(int(next_token))
                                if len(generated_tokens) == 1 or len(generated_tokens) % 32 == 0 or bool(eos_reached):
                                    _target_log(
                                        "server_only_ar_turn progress "
                                        f"tokens={len(generated_tokens)}/{max_new_tokens} "
                                        f"eos={bool(eos_reached)}"
                                    )
                            if energy_rate_weight_total > 0:
                                target_energy_rate_per_sec = energy_rate_weighted_sum / energy_rate_weight_total
                            verify_msg = {
                                "type": "verify_result",
                                "accepted_tokens": generated_tokens,
                                "accepted_plus_next_tokens": generated_tokens,
                                "accept_length": max(0, len(generated_tokens) - 1),
                                "next_token": int(generated_tokens[-1]) if generated_tokens else int(next_token),
                                "eos_reached": bool(eos_reached),
                                "best_ids": list(range(len(generated_tokens))),
                                "base_input_len": int(len(input_ids)),
                                "kv_start": int(len(input_ids)),
                                "target_verification_time_ms": float(total_target_ms),
                                "target_energy_rate_per_sec": target_energy_rate_per_sec,
                                "tree_build_time_ms": 0.0,
                                "draft_to_target_time_ms": 0.0,
                                "target_to_draft_time_ms": 0.0,
                                "final_nnodes": 1,
                                "tree_depth": 1,
                                "depth_widths": [1],
                                "tree_model_forward_ms": 0.0,
                                "tree_width_algo_ms": 0.0,
                                "tree_nnodes_algo_ms": 0.0,
                                "tree_mask_build_ms": 0.0,
                                "tree_finalize_ms": 0.0,
                                "server_only_ar": True,
                                "server_only_ar_turn": True,
                                "server_only_done": True,
                                "generation_limit_reached": bool(len(generated_tokens) >= max_new_tokens),
                            }
                            send_json(conn, verify_msg)
                            _target_log(
                                "server_only_ar_turn done inline "
                                f"tokens={len(generated_tokens)} eos={bool(eos_reached)} "
                                f"limit={bool(len(generated_tokens) >= max_new_tokens)} "
                                f"target_ms={float(total_target_ms):.3f}"
                            )
                        except Exception as e:
                            _target_log(f"server_only_ar_turn failed: {e}\n{traceback.format_exc()}")
                            try:
                                send_json(conn, {"type": "error", "message": f"server_only_ar_turn failed: {e}"})
                            except Exception:
                                pass
                            continue
                    elif mtype == "server_only_ar_init":
                        _target_log("server_only_ar_init start")
                        if worker is None:
                            send_json(conn, {"type": "error", "message": "target model is not loaded. Call reload_model first."})
                            continue
                        if worker.gpu_monitor is not None:
                            try:
                                worker.gpu_monitor.reset_data()
                            except Exception:
                                pass
                        input_ids = msg["input_ids"]
                        max_new_tokens = int(max(1, int(msg.get("max_new_tokens", 1024))))
                        ar_debug_interval = int(
                            max(0, int(os.environ.get("SERVER_ONLY_AR_DEBUG_INTERVAL", "32")))
                        )
                        ar_progress_interval = int(
                            max(1, int(os.environ.get("SERVER_ONLY_AR_PROGRESS_INTERVAL", "32")))
                        )
                        token_count = 0
                        client_disconnected_during_ar = False
                        server_only_done_sent_inline = False
                        send_json(conn, {"type": "server_only_ok", "mode": "autoregressive_model_forward"})

                        try:
                            _target_log(
                                "server_only_ar prefill begin "
                                f"prompt_tokens={len(input_ids)} max_new_tokens={max_new_tokens}"
                            )
                            next_token, eos_reached, target_energy_rate_per_sec, timing_stats = worker.handle_autoregressive_prefill(input_ids)
                            _target_log(
                                "server_only_ar prefill done "
                                f"first_token={int(next_token)} "
                                f"prefill_ms={float(timing_stats.get('target_verification_time_ms', 0.0) or 0.0):.3f}"
                            )
                        except Exception as e:
                            _target_log(f"server_only_ar prefill failed: {e}\n{traceback.format_exc()}")
                            try:
                                send_json(conn, {"type": "error", "message": f"server_only_ar prefill failed: {e}"})
                            except Exception:
                                pass
                            continue
                        while True:
                            accepted_token = int(next_token)
                            will_stop_after_this_token = bool(eos_reached or (token_count + 1) >= max_new_tokens)
                            verify_msg = {
                                "type": "verify_result",
                                "accepted_tokens": [accepted_token],
                                "accepted_plus_next_tokens": [accepted_token],
                                # AR has no drafted tokens to accept. The one emitted
                                # token is counted by the draft side as accept_length + 1.
                                "accept_length": 0,
                                "next_token": accepted_token,
                                "eos_reached": bool(eos_reached),
                                "best_ids": [0],
                                "base_input_len": int(len(input_ids) + token_count),
                                "kv_start": int(len(input_ids) + token_count),
                                "target_verification_time_ms": timing_stats.get("target_verification_time_ms", None),
                                "target_energy_rate_per_sec": target_energy_rate_per_sec,
                                "tree_build_time_ms": 0.0,
                                "draft_to_target_time_ms": 0.0,
                                "target_to_draft_time_ms": 0.0,
                                "final_nnodes": 1,
                                "tree_depth": 1,
                                "depth_widths": [1],
                                "tree_model_forward_ms": 0.0,
                                "tree_width_algo_ms": 0.0,
                                "tree_nnodes_algo_ms": 0.0,
                                "tree_mask_build_ms": 0.0,
                                "tree_finalize_ms": 0.0,
                                "server_only_ar": True,
                                "server_only_done": bool(will_stop_after_this_token),
                                "generation_limit_reached": bool((token_count + 1) >= max_new_tokens),
                            }
                            if will_stop_after_this_token:
                                _target_log(
                                    "server_only_ar final verify_result send begin "
                                    f"tokens={token_count + 1}/{max_new_tokens} "
                                    f"eos={bool(eos_reached)} "
                                    f"limit={bool((token_count + 1) >= max_new_tokens)} "
                                    f"token={accepted_token}"
                                )
                            try:
                                send_json(conn, verify_msg)
                            except (BrokenPipeError, ConnectionError, OSError) as e:
                                _target_log(
                                    "server_only_ar verify_result send failed "
                                    f"(client likely disconnected): {e}"
                                )
                                client_disconnected_during_ar = True
                                break

                            token_count += 1
                            if will_stop_after_this_token:
                                server_only_done_sent_inline = True
                            if token_count == 1 or token_count % ar_progress_interval == 0 or bool(eos_reached) or bool(will_stop_after_this_token):
                                _target_log(
                                    "server_only_ar progress "
                                    f"tokens={token_count}/{max_new_tokens} "
                                    f"eos={bool(eos_reached)} "
                                    f"done={bool(will_stop_after_this_token)}"
                                )
                            if eos_reached or token_count >= max_new_tokens:
                                break
                            try:
                                should_log_next_step = bool(
                                    ar_debug_interval > 0
                                    and (
                                        ar_debug_interval == 1
                                        or token_count == 1
                                        or token_count % ar_debug_interval == 0
                                    )
                                )
                                if should_log_next_step:
                                    _target_log(
                                        "server_only_ar next begin "
                                        f"prev_tokens={token_count}/{max_new_tokens} "
                                        f"accepted_token={accepted_token} "
                                        f"next_position={getattr(worker, 'ar_next_position', None)}"
                                    )
                                next_token, eos_reached, target_energy_rate_per_sec, timing_stats = worker.handle_autoregressive_next(accepted_token)
                                if should_log_next_step:
                                    _target_log(
                                        "server_only_ar next done "
                                        f"prev_tokens={token_count}/{max_new_tokens} "
                                        f"next_token={int(next_token)} "
                                        f"eos={bool(eos_reached)} "
                                        f"forward_ms={float(timing_stats.get('target_verification_time_ms', 0.0) or 0.0):.3f} "
                                        f"next_position={getattr(worker, 'ar_next_position', None)}"
                                    )
                            except Exception as e:
                                _target_log(f"server_only_ar next failed: {e}\n{traceback.format_exc()}")
                                try:
                                    send_json(conn, {"type": "error", "message": f"server_only_ar next failed: {e}"})
                                except Exception:
                                    pass
                                break

                        if client_disconnected_during_ar:
                            _target_log("server_only_ar_init aborted: client disconnected during streaming")
                            continue
                        if server_only_done_sent_inline:
                            _target_log(f"server_only_ar_init done inline tokens={token_count}")
                            continue
                        try:
                            send_json(conn, {"type": "server_only_done"})
                            _target_log(f"server_only_ar_init done tokens={token_count}")
                        except (BrokenPipeError, ConnectionError, OSError) as e:
                            _target_log(
                                "server_only_ar_done send failed "
                                f"(client likely disconnected): {e}"
                            )
                            continue
                    elif mtype == "server_only_init":
                        _target_log("server_only_init start")
                        if worker is None:
                            err_msg = "server-only mode requires loaded target model and --draft-model-path on target"
                            print(f"[target][ERROR] {err_msg}")
                            send_json(conn, {"type": "error", "message": err_msg})
                            continue
                        if worker.gpu_monitor is not None:
                            try:
                                worker.gpu_monitor.reset_data()
                            except Exception:
                                pass
                        requested_server_draft_model_path = str(msg.get("draft_model_path", draft_model_path or "")).strip()
                        requested_server_draft_quantization = str(msg.get("draft_quantization", "none") or "none").strip().lower()
                        if requested_server_draft_quantization not in {"none", "4bit", "8bit"}:
                            requested_server_draft_quantization = "none"
                        server_draft_changed = (
                            server_runner is not None
                            and (
                                str(current_server_draft_model_path).strip() != requested_server_draft_model_path
                                or str(current_server_draft_quantization).strip().lower() != requested_server_draft_quantization
                            )
                        )
                        if server_draft_changed:
                            _target_log(
                                "server-only draft reload requested "
                                f"old_model={current_server_draft_model_path or '<none>'} "
                                f"old_quant={current_server_draft_quantization or '<none>'} "
                                f"new_model={requested_server_draft_model_path} "
                                f"new_quant={requested_server_draft_quantization}"
                            )
                            prev_server_runner = server_runner
                            server_runner = None
                            current_server_draft_model_path = ""
                            current_server_draft_quantization = ""
                            _shutdown_server_draft_executor()
                            try:
                                _dispose_runtime_object(prev_server_runner)
                                del prev_server_runner
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except Exception:
                                pass
                        if server_runner is None and requested_server_draft_model_path:
                            try:
                                server_runner = _build_server_only_runner(
                                    active_worker=worker,
                                    draft_path=requested_server_draft_model_path,
                                    quant_mode=requested_server_draft_quantization,
                                    dev_map=current_device_map,
                                    offload=current_int8_cpu_offload,
                                )
                                current_server_draft_model_path = requested_server_draft_model_path
                                current_server_draft_quantization = requested_server_draft_quantization
                            except Exception as e:
                                print(f"[target][ERROR] failed to build server-only draft runner: {e}")
                                server_runner = None
                        if server_runner is None:
                            err_msg = "server-only mode requires loaded target model and draft_model_path"
                            print(f"[target][ERROR] {err_msg}")
                            send_json(conn, {"type": "error", "message": err_msg})
                            continue
                        # server-only session init
                        input_ids = msg["input_ids"]
                        max_new_tokens = int(max(1, int(msg.get("max_new_tokens", 1024))))
                        nodes = int(msg.get("nodes", 50))
                        max_depth = int(msg.get("max_depth", 10))
                        per_token_probability_bound = float(msg.get("per_token_probability_bound", 0.0))
                        per_path_probability_bound = float(msg.get("per_path_probability_bound", 0.0))
                        min_width = int(msg.get("min_width", 1))
                        fixed_width = bool(msg.get("fixed_width", False))
                        fixed_width_value = msg.get("fixed_width_value", None)
                        fixed_nnodes = bool(msg.get("fixed_nnodes", False))
                        fixed_depth = bool(msg.get("fixed_depth", False))
                        proactive_drafting = bool(msg.get("proactive_drafting", True))
                        proactive_threshold = float(msg.get("proactive_threshold", 0.0))
                        adaptive_proactive_threshold = bool(msg.get("adaptive_proactive_threshold", False))
                        bill_draft_as_target_gpu = bool(msg.get("bill_draft_as_target_gpu", False))
                        server_draft_profile_auto = bool(msg.get("server_draft_profile_auto", True)) and bool(getattr(args, "enable_auto_server_draft_profile", True))
                        server_draft_profile_force_refresh = bool(msg.get("server_draft_profile_force_refresh", False))
                        server_draft_profile_model_calls_per_count = int(max(1, int(msg.get("server_draft_profile_model_calls_per_count", 100))))
                        server_draft_profile_width_list = str(msg.get("server_draft_profile_width_list", "10,20,30,40,50,60,70,80,90,100,110,120,130,140,150"))
                        server_name_for_profile = _sanitize_key_component(
                            str(msg.get("server_name", "")).strip()
                            or str(getattr(args, "server_name", "target") or "target")
                        )
                        server_bench_name = str(msg.get("bench_name", "mt_bench"))
                        server_question_file = str(msg.get("question_file", ""))
                        server_runner.cost_sensitivity = float(msg.get("cost_sensitivity", 0.0))
                        server_runner.draft_per_sec_cost = float(msg.get("draft_per_sec_cost", 0.0))
                        server_runner.target_per_sec_cost = float(msg.get("target_per_sec_cost", 0.0))
                        server_runner.reference_tps = float(msg.get("reference_tps", 1.0))
                        server_runner.reference_objective_per_token = float(msg.get("reference_objective_per_token", 1.0))
                        server_runner.objective_metric = _normalize_objective_metric(msg.get("objective_metric", "cost"))
                        server_runner.no_draft_cost = bool(msg.get("no_draft_cost", False))
                        server_runner.server_only_mode = True
                        effective_bill_draft_as_target_gpu = bool(bill_draft_as_target_gpu)
                        if server_runner.objective_metric in {"total_cost", "api_cost"} and (not server_runner.no_draft_cost):
                            if not effective_bill_draft_as_target_gpu:
                                _target_log(
                                    "server-only cost objective detected; force-enabling draft billing as target GPU "
                                    "for time-based cost aggregation."
                                )
                            effective_bill_draft_as_target_gpu = True
                        server_runner.bill_draft_as_target_gpu = bool(effective_bill_draft_as_target_gpu)
                        if server_runner.bill_draft_as_target_gpu and (not server_runner.no_draft_cost):
                            server_runner.draft_per_sec_cost = float(server_runner.target_per_sec_cost)
                        if server_runner.no_draft_cost:
                            server_runner.draft_objective_rate_per_sec = 0.0
                        elif (
                            server_runner.objective_metric == "total_cost"
                            or (
                                server_runner.objective_metric == "api_cost"
                                and bool(server_runner.bill_draft_as_target_gpu)
                            )
                        ):
                            server_runner.draft_objective_rate_per_sec = float(server_runner.draft_per_sec_cost)
                        else:
                            server_runner.draft_objective_rate_per_sec = 0.0
                        server_runner.target_objective_rate_per_sec = (
                            float(server_runner.target_per_sec_cost)
                            if server_runner.objective_metric in {"total_cost", "api_cost"}
                            else 0.0
                        )
                        setattr(server_runner, "server_name_for_profile", str(server_name_for_profile))
                        profile_status = {"source": "disabled", "file": None, "rows": 0}
                        if server_draft_profile_auto:
                            try:
                                profile_status = _ensure_server_draft_profile(
                                    server_runner,
                                    worker,
                                    quant_mode=str(getattr(server_runner, "server_draft_quantization", current_quantization)),
                                    bench_name=server_bench_name,
                                    question_file=server_question_file,
                                    force_refresh=bool(server_draft_profile_force_refresh),
                                    model_calls_per_count=server_draft_profile_model_calls_per_count,
                                    width_list_csv=server_draft_profile_width_list,
                                )
                            except Exception as e:
                                send_json(conn, {"type": "error", "message": f"server draft profile prepare failed: {e}"})
                                continue
                        if (
                            server_runner.objective_metric in {"draft_energy", "target_energy"}
                            and not server_runner.no_draft_cost
                        ):
                            profile_energy_rate = _estimate_profile_energy_rate_per_sec(
                                getattr(server_runner, "profile_data", None)
                            )
                            if profile_energy_rate > 0:
                                server_runner.draft_objective_rate_per_sec = float(profile_energy_rate)
                        server_runner.reset_kv()
                        server_runner.reset_proactive_kv()
                        worker.reset_kv()
                        current_next_token = worker.handle_init(input_ids)
                        send_json(conn, {"type": "server_only_ok", "server_draft_profile": profile_status})
                        input_ids_t = torch.as_tensor([input_ids]).to(server_runner.draft_model.lm_head.weight.device)
                        use_proactive_tree = False
                        proactive_tree = None
                        pending_proactive = None
                        last_target_verification_ms = None
                        last_draft_to_target_ms = None
                        last_target_to_draft_ms = None
                        turn_steps = 0
                        new_token_count = 0
                        client_disconnected_during_server_only = False
                        server_only_direct_draft = str(os.environ.get("AUTODRAFT_SERVER_ONLY_DIRECT_DRAFT", "1")).strip() != "0"
                        while True:
                            tree_build_start_time = time.time()
                            proactive_elapsed_sec = 0.0
                            if use_proactive_tree and proactive_tree is not None:
                                draft_ids = proactive_tree["draft_ids"]
                                draft_pos = proactive_tree["draft_pos"]
                                tree_mask = proactive_tree["tree_mask"]
                                parent = proactive_tree["parent"]
                                tree_depth = proactive_tree["tree_depth"]
                                final_nnodes = proactive_tree["final_nnodes"]
                                depth_widths = proactive_tree["depth_widths"]
                                node_meta = proactive_tree.get("node_meta")
                                server_runner.last_sum_expected_accepted_length = proactive_tree.get("expected_accept_length")
                                server_runner.last_accept_length_scale_used = float(proactive_tree.get("accept_length_scale_used", 1.0))
                                server_runner.last_tree_timing_breakdown = dict(proactive_tree.get("timing_breakdown", {}) or {})
                                use_proactive_tree = False
                                proactive_tree = None
                            else:
                                if server_only_direct_draft:
                                    draft_ids, draft_pos, tree_mask, parent, tree_depth, final_nnodes, depth_widths, node_meta = build_tree_with_next_token(
                                        server_runner,
                                        input_ids_t,
                                        nodes,
                                        max_depth,
                                        current_next_token,
                                        worker.tokenizer,
                                        debug,
                                        False,
                                        per_token_probability_bound=per_token_probability_bound,
                                        per_path_probability_bound=per_path_probability_bound,
                                        min_width=min_width,
                                        fixed_width=fixed_width,
                                        fixed_width_value=fixed_width_value,
                                        fixed_nnodes=fixed_nnodes,
                                        fixed_depth=fixed_depth,
                                    )
                                else:
                                    draft_executor = _ensure_server_draft_executor()
                                    draft_future = draft_executor.submit(
                                        lambda: build_tree_with_next_token(
                                            server_runner,
                                            input_ids_t,
                                            nodes,
                                            max_depth,
                                            current_next_token,
                                            worker.tokenizer,
                                            debug,
                                            False,
                                            per_token_probability_bound=per_token_probability_bound,
                                            per_path_probability_bound=per_path_probability_bound,
                                            min_width=min_width,
                                            fixed_width=fixed_width,
                                            fixed_width_value=fixed_width_value,
                                            fixed_nnodes=fixed_nnodes,
                                            fixed_depth=fixed_depth,
                                        )
                                    )
                                    draft_ids, draft_pos, tree_mask, parent, tree_depth, final_nnodes, depth_widths, node_meta = draft_future.result()
                            tree_build_time = (time.time() - tree_build_start_time) * 1000.0
                            proactive_path_prob = None
                            proactive_future = None
                            proactive_result = {"tree": None, "error": None, "elapsed_sec": None, "head_token": None}
                            proactive_stop_flag = None
                            draft_executor = _ensure_server_draft_executor()
                            if proactive_drafting:
                                proactive_stop_flag = threading.Event()
                                node_tokens = draft_ids[0].tolist()[1:]
                                parent_list = parent.tolist() if isinstance(parent, torch.Tensor) else list(parent)
                                proactive_path, proactive_path_prob = _select_proactive_path(node_tokens, parent_list, node_meta)
                                should_start_proactive = False
                                if proactive_path and proactive_path_prob is not None:
                                    path_prob = float(proactive_path_prob)
                                    has_adaptive_stats = (
                                        last_target_verification_ms is not None
                                        and last_draft_to_target_ms is not None
                                        and last_target_to_draft_ms is not None
                                    )
                                    if adaptive_proactive_threshold and has_adaptive_stats:
                                        overlap_ms = min(
                                            float(tree_build_time),
                                            float(last_target_verification_ms) + float(last_draft_to_target_ms) + float(last_target_to_draft_ms),
                                        )
                                        overlap_sec = max(0.0, overlap_ms / 1000.0)
                                        expected_latency_gain_sec = path_prob * overlap_sec
                                        cost_per_sec = server_runner.get_draft_objective_rate_per_sec()
                                        expected_cost_loss = (1.0 - path_prob) * (overlap_sec * cost_per_sec)
                                        alpha = server_runner.get_sensitivity_alpha()
                                        latency_gain_norm = expected_latency_gain_sec / max(1e-9, server_runner.get_reference_latency_per_token())
                                        cost_loss_norm = expected_cost_loss / server_runner.get_reference_objective_per_token()
                                        if ((1.0 - alpha) * latency_gain_norm - alpha * cost_loss_norm) >= 0:
                                            should_start_proactive = True
                                    else:
                                        if path_prob >= float(proactive_threshold):
                                            should_start_proactive = True
                                if should_start_proactive:
                                    proactive_path = [current_next_token] + proactive_path
                                    def _build_proactive_tree_local() -> Dict[str, Any]:
                                        st = time.time()
                                        try:
                                            tree_payload = build_proactive_tree_from_path(
                                                runner=server_runner,
                                                base_input_ids=input_ids_t,
                                                path_tokens=proactive_path,
                                                nodes=nodes,
                                                max_depth=max_depth,
                                                tokenizer=worker.tokenizer,
                                                debug=debug,
                                                print_tree=False,
                                                per_token_probability_bound=per_token_probability_bound,
                                                per_path_probability_bound=per_path_probability_bound,
                                                min_width=min_width,
                                                fixed_width=fixed_width,
                                                fixed_width_value=fixed_width_value,
                                                fixed_nnodes=fixed_nnodes,
                                                fixed_depth=fixed_depth,
                                                stop_flag=proactive_stop_flag,
                                                head_token_holder=proactive_result,
                                            )
                                            return {
                                                "tree": tree_payload,
                                                "error": None,
                                                "elapsed_sec": float(time.time() - st),
                                                "head_token": proactive_result.get("head_token"),
                                            }
                                        except Exception as e:
                                            return {
                                                "tree": None,
                                                "error": str(e),
                                                "elapsed_sec": float(time.time() - st),
                                                "head_token": proactive_result.get("head_token"),
                                            }
                                    proactive_future = draft_executor.submit(_build_proactive_tree_local)
                            best_candidate, accept_length, next_token, eos_reached, target_energy_rate_per_sec, timing_stats, best_ids, kv_start, base_input_len = worker.handle_tree_step(
                                draft_ids[0].tolist(),
                                draft_pos.tolist(),
                                tree_mask.tolist(),
                                parent.tolist(),
                            )
                            server_runner.update_target_objective_rate(target_energy_rate_per_sec)
                            if proactive_future is not None:
                                canceled = False
                                if accept_length != tree_depth:
                                    canceled = True
                                else:
                                    proactive_head = proactive_result.get("head_token")
                                    if proactive_head is None or next_token != proactive_head:
                                        canceled = True
                                if canceled:
                                    proactive_stop_flag.set()
                                    server_runner.reset_proactive_kv()
                                    pending_proactive = None
                                else:
                                    proactive_result = proactive_future.result()
                                    pending_proactive = proactive_result.get("tree")
                                    if proactive_result.get("elapsed_sec") is not None:
                                        proactive_elapsed_sec = float(proactive_result["elapsed_sec"])
                                if pending_proactive and accept_length == tree_depth and next_token == pending_proactive.get("head_token"):
                                    use_proactive_tree = True
                                    proactive_tree = pending_proactive
                                    server_runner.draft_stable_kv = server_runner.proactive_kv
                                else:
                                    use_proactive_tree = False
                                    proactive_tree = None
                                    server_runner.reset_proactive_kv()
                            tree_build_time += proactive_elapsed_sec * 1000.0
                            tree_timing_breakdown = getattr(server_runner, "last_tree_timing_breakdown", {}) or {}
                            verify_msg = {
                                "type": "verify_result",
                                "accepted_tokens": best_candidate,
                                "accepted_plus_next_tokens": list(best_candidate) + [int(next_token)],
                                "accept_length": accept_length,
                                "next_token": next_token,
                                "eos_reached": eos_reached,
                                "best_ids": best_ids,
                                "base_input_len": base_input_len,
                                "kv_start": kv_start,
                                "target_verification_time_ms": timing_stats.get("target_verification_time_ms", None),
                                "target_energy_rate_per_sec": target_energy_rate_per_sec,
                                "tree_build_time_ms": tree_build_time,
                                "draft_to_target_time_ms": 0.0,
                                "target_to_draft_time_ms": 0.0,
                                "final_nnodes": int(final_nnodes),
                                "tree_depth": int(tree_depth),
                                "depth_widths": [int(w) for w in depth_widths],
                                "tree_model_forward_ms": float(tree_timing_breakdown.get("tree_model_forward_ms", 0.0) or 0.0),
                                "tree_width_algo_ms": float(tree_timing_breakdown.get("tree_width_algo_ms", 0.0) or 0.0),
                                "tree_nnodes_algo_ms": float(tree_timing_breakdown.get("tree_nnodes_algo_ms", 0.0) or 0.0),
                                "tree_mask_build_ms": float(tree_timing_breakdown.get("tree_mask_build_ms", 0.0) or 0.0),
                                "tree_finalize_ms": float(tree_timing_breakdown.get("tree_finalize_ms", 0.0) or 0.0),
                                "server_draft_energy_rate_per_sec": float(server_runner.get_draft_objective_rate_per_sec()),
                            }
                            try:
                                send_json(conn, verify_msg)
                            except (BrokenPipeError, ConnectionError, OSError) as e:
                                _target_log(
                                    "server_only verify_result send failed "
                                    f"(client likely disconnected): {e}"
                                )
                                client_disconnected_during_server_only = True
                                break
                            input_ids_t = torch.cat([
                                input_ids_t,
                                torch.tensor([best_candidate], device=input_ids_t.device, dtype=torch.long),
                            ], dim=-1)
                            current_next_token = next_token
                            last_target_verification_ms = timing_stats.get("target_verification_time_ms", None)
                            last_draft_to_target_ms = 0.0
                            last_target_to_draft_ms = 0.0
                            new_token_count += len(best_candidate)
                            turn_steps += 1
                            if eos_reached or new_token_count >= max_new_tokens:
                                break
                        if client_disconnected_during_server_only:
                            _target_log("server_only_init aborted: client disconnected during streaming")
                            continue
                        try:
                            send_json(conn, {"type": "server_only_done"})
                            _target_log("server_only_init done")
                        except (BrokenPipeError, ConnectionError, OSError) as e:
                            _target_log(
                                "server_only_done send failed "
                                f"(client likely disconnected): {e}"
                            )
                            continue
                    elif mtype == "shutdown":
                        # (output_file None )
                        if timing_data or gpu_data:
                            save_performance_stats(timing_data, gpu_data, output_file, step_count, args, accept_lengths)
                        _shutdown_server_draft_executor()
                        _dispose_runtime_object(worker)
                        send_json(conn, {"type": "bye"})
                        _target_log("shutdown requested; server stopping")
                        server_running = False
                        break
                    else:
                        err_msg = f"unknown type {mtype}"
                        print(f"[target][ERROR] {err_msg}")
                        send_json(conn, {"type": "error", "message": err_msg})
                        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="192.168.0.2")
    parser.add_argument("--port", type=int, default=26001)
    parser.add_argument("--base-model-path", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--int8-cpu-offload", action="store_true")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--enable-gpu-monitor", action="store_true", default=False,
                       help="Enable GPU monitoring")
    parser.add_argument("--gpu-monitor-interval", type=float, default=0.05,
                       help="GPU monitoring interval (s) (default: 0.05)")
    parser.add_argument("--output-file", type=str, default=None,
                       help="Performance statistics output filename (do not save if unspecified)")
    parser.add_argument("--fix-gpu-clock", action="store_true", default=False,
                       help="Enable fixed GPU clocks")
    parser.add_argument("--graphics-clock", type=int, default=None,
                       help="Graphics clock to fix (MHz)")
    parser.add_argument("--memory-clock", type=int, default=None,
                       help="Memory clock to fix (MHz)")
    parser.add_argument("--debug", action="store_true", default=False,
                       help="Enable debug output (TARGET-DEBUG messages)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Seed for reproducibility (sets torch/random/numpy seeds when specified)")
    parser.add_argument("--deterministic", action="store_true", default=False,
                       help="Ensure identical results including GPU operations (PyTorch deterministic mode; some operations may be slower)")
    parser.add_argument("--draft-model-path", type=str, default=None,
                       help="Draft model path for server-only mode (run draft/target together on target)")
    parser.add_argument("--eager-load", action="store_true", default=False,
                       help="Load the target model immediately at process start (default is lazy-load)")
    parser.add_argument("--server-name", type=str, default="target",
                       help="Server identifier included in profile filenames")
    parser.add_argument("--enable-auto-target-profile", dest="enable_auto_target_profile", action="store_true", default=True,
                       help="Allow automatic target profiling RPC requested by draft (default: on)")
    parser.add_argument("--disable-auto-target-profile", dest="enable_auto_target_profile", action="store_false",
                       help="Disable automatic target profiling RPC")
    parser.add_argument("--enable-auto-server-draft-profile", dest="enable_auto_server_draft_profile", action="store_true", default=True,
                       help="Enable automatic local draft profile generation/load on target server for server-only mode")
    parser.add_argument("--disable-auto-server-draft-profile", dest="enable_auto_server_draft_profile", action="store_false",
                       help="Disable automatic local draft profile generation/load on target server for server-only mode")

    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)
        print(f"[target] seed fixed: {args.seed}")
    if args.deterministic:
        set_deterministic()
        print("[target] deterministic mode enabled")

    # GPU
    if args.fix_gpu_clock:
        print("=== Available GPU clock combinations ===")
        available_clocks = get_available_gpu_clocks()
        if available_clocks:
            print("Graphics Clock (MHz) | Memory Clock (MHz)")
            print("-" * 40)
            for graphics, memory in available_clocks[:20]:  # 20
                print(f"{graphics:>18} | {memory:>15}")
            if len(available_clocks) > 20:
                print(f"... and {len(available_clocks) - 20}more")
            print("=" * 40)
        else:
            print("Could not get GPU clock information.")
            print("=" * 40)
    
    quantization = "none"
    if args.load_in_8bit:
        quantization = "8bit"
    elif args.load_in_4bit or not args.load_in_8bit:
        # 4-bit quantization 
        quantization = "4bit"

    serve(
        args.host,
        args.port,
        args.base_model_path,
        args.temperature,
        quantization,
        args.int8_cpu_offload,
        args.device_map,
        args.enable_gpu_monitor,
        args.gpu_monitor_interval,
        args.output_file,
        args.fix_gpu_clock,
        args.graphics_clock,
        args.memory_clock,
        args=args,  # args experiment_info
        debug=args.debug,
        draft_model_path=args.draft_model_path,
        preload_model_on_start=bool(args.eager_load),
    )


