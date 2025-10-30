# ✅ YENİ: parallel_executor.py - CPU-bound işlemler için optimize
# analysis/parallel_executor.py


import os
import asyncio
import concurrent.futures
from typing import List, Callable, Any

class ParallelExecutor:
    def __init__(self, max_workers: int = None, strategy: str = "auto"):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.strategy = strategy
        
    async def execute_cpu_bound(self, tasks: List[Callable]) -> List[Any]:
        """CPU-bound metrik hesaplamaları için thread pool"""
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = await asyncio.gather(*[
                loop.run_in_executor(executor, task)
                for task in tasks
            ])
        return results
    
    async def execute_io_bound(self, tasks: List[Callable]) -> List[Any]:
        """IO-bound işlemler için direkt async"""
        return await asyncio.gather(*tasks, return_exceptions=True)


    async def execute_optimized(self, tasks: List[Callable], task_types: List[str] = None) -> List[Any]:
        """Task tipine göre optimize execution"""
        if task_types is None:
            task_types = ["io"] * len(tasks)  # Default IO-bound
            
        cpu_tasks = []
        io_tasks = []
        
        for task, task_type in zip(tasks, task_types):
            if task_type == "cpu":
                cpu_tasks.append(task)
            else:
                io_tasks.append(task)
        
        # Paralel execution
        cpu_results = await self.execute_cpu_bound(cpu_tasks)
        io_results = await self.execute_io_bound(io_tasks)
        
        return self._reconstruct_results(tasks, cpu_results, io_results, task_types)
        