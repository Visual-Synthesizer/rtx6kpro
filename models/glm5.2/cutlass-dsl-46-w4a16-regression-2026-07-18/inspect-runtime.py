import hashlib
import importlib.metadata as md
from pathlib import Path

import cutlass.cute as cute


mma = Path(cute.__file__).parent / "nvgpu" / "warp" / "mma.py"
print("nvidia-cutlass-dsl", md.version("nvidia-cutlass-dsl"))
print("nvidia-cutlass-dsl-libs-base", md.version("nvidia-cutlass-dsl-libs-base"))
print("nvidia-cutlass-dsl-libs-cu13", md.version("nvidia-cutlass-dsl-libs-cu13"))
print("mma.py", mma)
print("mma.py bytes", mma.stat().st_size)
print("mma.py sha256", hashlib.sha256(mma.read_bytes()).hexdigest())
print("MmaMXF8Op", hasattr(cute.nvgpu.warp, "MmaMXF8Op"))
