import subprocess
import time
import threading
from collections import defaultdict
from datetime import datetime
import psutil
import signal
import os


class GPUServiceManager:
    def __init__(self, gpu_threshold=90):
        """
        Initialize GPU service manager
        
        Args:
            gpu_threshold: GPU utilization threshold (percentage), cleanup will be triggered when exceeded
        """
        self.gpu_threshold = gpu_threshold
        self.services = {}  # {service_name: {'process': process, 'call_count': count, 'start_time': time}}
        self.call_counts = defaultdict(int)
        self.lock = threading.Lock()
        
    def get_gpu_usage(self):
        """Get GPU utilization"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info = []
                for line in lines:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        util = float(parts[0].strip())
                        mem_used = float(parts[1].strip())
                        mem_total = float(parts[2].strip())
                        mem_percent = (mem_used / mem_total) * 100 if mem_total > 0 else 0
                        gpu_info.append({
                            'utilization': util,
                            'memory_percent': mem_percent,
                            'memory_used': mem_used,
                            'memory_total': mem_total
                        })
                return gpu_info
            else:
                print(f"nvidia-smi execution failed: {result.stderr}")
                return []
        except FileNotFoundError:
            print("nvidia-smi command not found, please ensure NVIDIA driver is installed")
            return []
        except Exception as e:
            print(f"Failed to get GPU info: {e}")
            return []
    
    def is_gpu_overloaded(self):
        """Check if GPU is overloaded"""
        gpu_info = self.get_gpu_usage()
        if not gpu_info:
            return False
        
        # Check if any GPU's utilization or memory usage exceeds the threshold
        for gpu in gpu_info:
            if gpu['utilization'] >= self.gpu_threshold or gpu['memory_percent'] >= self.gpu_threshold:
                return True
        return False
    
    def start_service(self, service_name, command, gpu_id=None):
        """
        Start a new service
        
        Args:
            service_name: Service name
            command: Start command (list format)
            gpu_id: Specify GPU ID to use
        """
        with self.lock:
            # If GPU is overloaded, cleanup first
            if self.is_gpu_overloaded():
                print(f"GPU utilization exceeds {self.gpu_threshold}%, preparing to cleanup services...")
                self.kill_least_used_service()
            
            # Start new service
            env = os.environ.copy()
            if gpu_id is not None:
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            
            try:
                process = subprocess.Popen(
                    command,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid  # Create new process group
                )
                
                self.services[service_name] = {
                    'process': process,
                    'call_count': 0,
                    'start_time': datetime.now(),
                    'command': command,
                    'gpu_id': gpu_id
                }
                
                print(f"Service '{service_name}' started (PID: {process.pid})")
                return True
            except Exception as e:
                print(f"Failed to start service: {e}")
                return False
    
    def kill_least_used_service(self):
        """Kill the service with the least number of calls"""
        if not self.services:
            print("No running services")
            return None
        
        # Find the service with the least number of calls
        least_used = min(
            self.services.items(),
            key=lambda x: (x[1]['call_count'], -x[1]['start_time'].timestamp())
        )
        
        service_name = least_used[0]
        service_info = least_used[1]
        
        print(f"Preparing to kill service '{service_name}' (call count: {service_info['call_count']})")
        
        try:
            process = service_info['process']
            # Get process group ID and terminate the entire process group
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGTERM)
            
            # Wait for process to terminate
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # If not terminated after 5 seconds, force kill
                os.killpg(pgid, signal.SIGKILL)
                process.wait()
            
            del self.services[service_name]
            print(f"Service '{service_name}' has been terminated")
            return service_name
            
        except Exception as e:
            print(f"Failed to terminate service: {e}")
            # If failed, remove from list
            if service_name in self.services:
                del self.services[service_name]
            return None
    
    def record_call(self, service_name):
        """Record service call"""
        with self.lock:
            if service_name in self.services:
                self.services[service_name]['call_count'] += 1
                print(f"Service '{service_name}' called (total: {self.services[service_name]['call_count']} times)")
            else:
                print(f"Warning: Service '{service_name}' does not exist")
    
    def get_service_status(self):
        """Get status of all services"""
        gpu_info = self.get_gpu_usage()
        
        status = {
            'gpu_info': gpu_info,
            'services': {}
        }
        
        with self.lock:
            for name, info in self.services.items():
                status['services'][name] = {
                    'pid': info['process'].pid,
                    'call_count': info['call_count'],
                    'start_time': info['start_time'].isoformat(),
                    'is_running': info['process'].poll() is None
                }
        
        return status
    
    def cleanup(self):
        """Cleanup all services"""
        with self.lock:
            for service_name, service_info in list(self.services.items()):
                try:
                    process = service_info['process']
                    pgid = os.getpgid(process.pid)
                    os.killpg(pgid, signal.SIGTERM)
                    process.wait(timeout=5)
                except:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except:
                        pass
            self.services.clear()
            print("All services cleaned up")





