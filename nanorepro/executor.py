import sys
import tempfile
import subprocess

def execute_code(code, timeout=5, max_memory=512*1024*1024):
    """Run code in a subprocess with temp cwd and resource limits. NOT a secure sandbox! Use with caution."""
    # Explainer - basically prevent obvious breakouts, trivial to bypass by adversarial code
    # - resource.RLIMIT_AS - total process memory limit (heap, stack, mappings, libs, etc.)
    # - builtins.exit/quit - prevent clever model to use quit() to exit with code 0 (success) before test asserts
    # - OMP_NUM_THREADS=1 - prevent excessive threads by native libraries
    # - os.kill - don't kill other processes
    # - os.system - don't run shell commands
    # - os.fork/os.forkpty - don't fork new processes
    # - os.killpg - don't kill/signal process groups
    # - subprocess.Popen - don't spawn new processes
    guard = f"""
import builtins, os, subprocess, resource
limit = {max_memory}
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
builtins.exit = None
builtins.quit = None
os.environ["OMP_NUM_THREADS"] = "1"
for name in ('kill', 'system', 'fork', 'forkpty', 'killpg'):
    setattr(os, name, None)
subprocess.Popen = None
"""
    # {code!r} - properly escapes code strings, preventing injection attacks (like SQL injection name = "Robert'); DROP TABLE Students;--")
    # compile() - compiles as new program, allowing 'from __future__ ...' to work (needs to be at the top of the file, which guard prevents)
    # exec() - executes compiled code above
    program = guard + f"\nexec(compile({code!r}, '<llm>', 'exec'), {{'__name__': '__main__'}})\n"
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            process = subprocess.run(
                [sys.executable, "-I", "-c", program],  # -I for isolated mode
                cwd=tmpdir,
                env={},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if process.returncode == 0:
        return True, None
    if "MemoryError" in process.stderr:
        return False, "memory limit exceeded"
    else:
        error_lines = process.stderr.strip().splitlines()
        error = error_lines[-1] if error_lines else "unknown error"
        return False, error
