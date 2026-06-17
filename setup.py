from setuptools import setup
import importlib.util
import os

if torch := importlib.util.find_spec("torch") is not None:
    from torch.utils import cpp_extension
    from torch import version as torch_version

extension_name = "exllamav3_ext"
precompile = "EXLLAMA_NOCOMPILE" not in os.environ
verbose = "EXLLAMA_VERBOSE" in os.environ
ext_debug = "EXLLAMA_EXT_DEBUG" in os.environ

if precompile and not torch:
    print("Cannot precompile unless torch is installed.")
    print("To explicitly JIT install run EXLLAMA_NOCOMPILE= pip install <xyz>")

windows = os.name == "nt"

is_rocm = bool(torch and torch_version.hip)

extra_cflags = []
if is_rocm:
    # hipcc / amdclang++ does not accept nvcc-only flags
    extra_cuda_cflags = [
        "-O3", "-ffast-math",
        "-Wno-register",            # kernels use the (C++17-removed) 'register' keyword
        "-Wno-unused-result",
        # Code uses 32-bit warp masks (0xffffffff); HIP's masked warp-sync
        # builtins static_assert on 64-bit masks. Disable them and remap the
        # *_sync names onto HIP's legacy warp builtins via rocm_compat.cuh.
        "-DHIP_DISABLE_WARP_SYNC_BUILTINS",
        "-include", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "exllamav3", extension_name, "rocm_compat.h"),
    ]
else:
    extra_cuda_cflags = [
        "-lineinfo", "-O3", "--use_fast_math",
        "-Xcudafe", "--diag_suppress=177",
        "-Xcudafe", "--diag_suppress=20012",
    ]

if windows:
    extra_cflags += ["/Ox", "/Zc:preprocessor", "/DWIN32_LEAN_AND_MEAN"]
    extra_cuda_cflags += ["-DWIN32_LEAN_AND_MEAN", "-Xcompiler=/Zc:preprocessor"]
    if ext_debug:
        extra_cflags += ["/Zi"]
        extra_cuda_cflags += []
else:
    extra_cflags += ["-Ofast"]
    extra_cuda_cflags += []
    if ext_debug:
        extra_cflags += ["-ftime-report", "-DTORCH_USE_CUDA_DSA"]
        extra_cuda_cflags += []

if cuda_host_cxx := os.environ.get("CUDAHOSTCXX"):
    extra_cuda_cflags += ["-ccbin", cuda_host_cxx]

if torch and torch_version.hip:
    extra_cuda_cflags += ["-DHIPBLAS_USE_HIP_HALF"]
    # The self-contained ROCm wheel ships core HIP headers but not the math-lib
    # headers (hipsparse/hipblas/rocblas) that ATen's HIP headers include. Pull
    # them from a system ROCm install if present (appended, so wheel headers win).
    _sys_rocm_inc = os.environ.get("SYSTEM_ROCM_INCLUDE", "/opt/rocm/include")
    if os.path.isdir(_sys_rocm_inc):
        extra_cflags += [f"-I{_sys_rocm_inc}"]
        extra_cuda_cflags += [f"-I{_sys_rocm_inc}"]

extra_compile_args = {
    "cxx": extra_cflags,
    "nvcc": extra_cuda_cflags,
}

# On ROCm the self-contained wheel splits the HIP runtime (rocm_sdk_core) and the
# math libraries (rocm_sdk_libraries_<arch>) across separate site-packages dirs.
# Point the linker at both (with rpath) so libamdhip64/libhipblas/etc. resolve at
# link and run time.
rocm_library_dirs = []
if is_rocm:
    for _pkg in ("_rocm_sdk_core", "_rocm_sdk_libraries_gfx1151"):
        try:
            _spec = importlib.util.find_spec(_pkg)
            if _spec and _spec.submodule_search_locations:
                _libdir = os.path.join(list(_spec.submodule_search_locations)[0], "lib")
                if os.path.isdir(_libdir):
                    rocm_library_dirs.append(_libdir)
        except Exception:
            pass

library_dir = "exllamav3"
sources_dir = os.path.join(library_dir, extension_name)
sources = [
    os.path.relpath(os.path.join(root, file), start=os.path.dirname(__file__))
    for root, _, files in os.walk(sources_dir)
    for file in files
    if file.endswith(('.c', '.cpp', '.cu'))
]

print (sources)

setup_kwargs = (
    {
        "ext_modules": [
            cpp_extension.CUDAExtension(
                extension_name,
                sources,
                extra_compile_args=extra_compile_args,
                libraries=["cublas"] if windows else [],
                library_dirs=rocm_library_dirs,
                extra_link_args=[f"-Wl,-rpath,{d}" for d in rocm_library_dirs],
            )
        ],
        "cmdclass": {"build_ext": cpp_extension.BuildExtension},
    }
    if precompile and torch
    else {}
)

version_py = {}
with open("exllamav3/version.py", encoding="utf8") as fp:
    exec(fp.read(), version_py)
version = version_py["__version__"]
print("Version:", version)

setup(
    name="exllamav3",
    version=version,
    packages=[
        "exllamav3",
        "exllamav3.generator",
        "exllamav3.generator.sampler",
        "exllamav3.generator.filter",
        "exllamav3.conversion",
        "exllamav3.conversion.standard_cal_data",
        "exllamav3.integration",
        "exllamav3.architecture",
        "exllamav3.architecture.mm_processing",
        "exllamav3.model",
        "exllamav3.modules",
        "exllamav3.modules.attention_fn",
        "exllamav3.modules.arch_specific",
        "exllamav3.modules.gated_delta_net_fn",
        "exllamav3.modules.quant",
        "exllamav3.modules.quant.exl3_lib",
        "exllamav3.tokenizer",
        "exllamav3.cache",
        "exllamav3.loader",
        "exllamav3.util",
    ],
    url="https://github.com/turboderp-org/exllamav3",
    license="MIT",
    author="turboderp",
    install_requires=[
        "torch>=2.6.0",
        "flash_attn>=2.7.4.post1",
        "tokenizers>=0.21.1",
        "numpy>=2.1.0",
        "rich",
        "typing_extensions",
        "ninja",
        "safetensors>=0.3.2",
        "pyyaml",
        "marisa_trie",
        "kbnf>=0.4.2",
        "formatron>=0.5.0",
        "pydantic",
        "xformers",
        "flash-linear-attention>=0.5.0",
    ],
    include_package_data=True,
    package_data = {
        "": ["py.typed"],
    },
    verbose=verbose,
    **setup_kwargs,
)
