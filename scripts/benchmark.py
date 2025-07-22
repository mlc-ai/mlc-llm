
import argparse
import json
import time
import psutil
import GPUtil
from pathlib import Path
from typing import Dict, List, Any
from mlc_llm import MLCEngine

class PerformanceBenchmark:
    def __init__(self, model_path: str, output_file: str = "benchmark-results.json"):
        self.model_path = model_path
        self.output_file = output_file
        self.results = []
        
    def run_inference_benchmark(self, messages: List[str], max_tokens: int = 100) -> Dict[str, Any]:
        """Run inference benchmark"""
        print(f"Running inference benchmark with {len(messages)} messages...")
        
        engine = MLCEngine(self.model_path)
        
        start_time = time.time()
        total_tokens = 0
        
        for i, message in enumerate(messages):
            msg_start = time.time()
            
            response = engine.chat.completions.create(
                messages=[{"role": "user", "content": message}],
                model=self.model_path,
                max_tokens=max_tokens
            )
            
            msg_end = time.time()
            if response.usage:
                total_tokens += response.usage.total_tokens
            
            print(f"  Message {i+1}/{len(messages)}: {msg_end - msg_start:.2f}s")
        
        end_time = time.time()
        engine.terminate()
        
        total_time = end_time - start_time
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        
        return {
            "test_name": "inference_benchmark",
            "total_time": total_time,
            "total_tokens": total_tokens,
            "tokens_per_second": tokens_per_second,
            "messages_count": len(messages),
            "max_tokens": max_tokens
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        system_info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "memory_percent": psutil.virtual_memory().percent
        }
        
        # GPU information
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_info = []
                for gpu in gpus:
                    gpu_info.append({
                        "id": gpu.id,
                        "name": gpu.name,
                        "memory_total": gpu.memoryTotal,
                        "memory_used": gpu.memoryUsed,
                        "memory_free": gpu.memoryFree,
                        "temperature": gpu.temperature,
                        "load": gpu.load
                    })
                system_info["gpus"] = gpu_info
        except Exception as e:
            print(f"Could not get GPU info: {e}")
            system_info["gpus"] = []
        
        return system_info
    
    def run_all_benchmarks(self):
        """Run all performance benchmarks"""
        print("Starting MLC-LLM Performance Benchmarks")
        print("=" * 50)
        
        # System information
        system_info = self.get_system_info()
        print(f"CPU Count: {system_info['cpu_count']}")
        print(f"Memory Total: {system_info['memory_total'] / (1024**3):.2f} GB")
        if system_info['gpus']:
            for gpu in system_info['gpus']:
                print(f"GPU: {gpu['name']} ({gpu['memory_total']} MB)")
        
        # Test messages
        test_messages = [
            "Hello, how are you?",
            "Explain the concept of machine learning in simple terms.",
            "Write a short story about a robot learning to paint.",
            "What are the advantages and disadvantages of renewable energy?",
            "Describe the process of photosynthesis."
        ]
        
        # Run benchmarks
        try:
            inference_results = self.run_inference_benchmark(test_messages)
            self.results.append({
                "timestamp": time.time(),
                "system_info": system_info,
                "benchmark_results": [inference_results]
            })
            
            print("\nBenchmark Results:")
            print(f"Total Time: {inference_results['total_time']:.2f}s")
            print(f"Tokens/Second: {inference_results['tokens_per_second']:.2f}")
            
        except Exception as e:
            print(f"Benchmark failed: {e}")
            self.results.append({
                "timestamp": time.time(),
                "system_info": system_info,
                "error": str(e)
            })
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save benchmark results to file"""
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="MLC-LLM Performance Benchmark")
    parser.add_argument(
        "--model", 
        default="HF://mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC",
        help="Model path for benchmarking"
    )
    parser.add_argument(
        "--output", 
        default="benchmark-results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark(args.model, args.output)
    benchmark.run_all_benchmarks()

if __name__ == "__main__":
    main()