from typing import Dict, Type
from backend import Backend
from backends.cpu.cpu_backend import CPUBackend
from backends.npu.npu_backend import NPUBackend
from backends.npu_new.npu_new_backend import NPUNewBackend

class BackendFactory:
    """Factory class for creating backend instances."""
    
    _backends: Dict[str, Type[Backend]] = {
        "cpu": CPUBackend,
        "npu": NPUBackend,  # AMD NPU with AI Engine
        "npu_new": NPUNewBackend,  # AMD NPU with Allo for the testbench
    }
    
    @classmethod
    def create_backend(cls, backend_name: str) -> Backend:
        """
        Create a backend instance.
        
        Args:
            backend_name: Name of the backend ("cpu", "ryzen_npu", "npu")
            
        Returns:
            Backend instance
            
        Raises:
            ValueError: If backend name is not supported
        """
        backend_name = backend_name.lower()
        
        if backend_name not in cls._backends:
            available = ', '.join(cls._backends.keys())
            raise ValueError(f"Unsupported backend: {backend_name}. Available backends: {available}")
        
        backend_class = cls._backends[backend_name]
        return backend_class()
    
    @classmethod
    def get_available_backends(cls) -> list[str]:
        """
        Get list of available backend names.
        
        Returns:
            List of available backend names
        """
        return list(cls._backends.keys())
    
    @classmethod
    def register_backend(cls, name: str, backend_class: Type[Backend]):
        """
        Register a new backend class.
        
        Args:
            name: Name of the backend
            backend_class: Backend class to register
        """
        cls._backends[name] = backend_class 