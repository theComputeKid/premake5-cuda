<div align="center">
   <p>a premake5 extension for cuda</p>
   <h1><small>premake5-</small><strong>CUDA</strong></h1>
</div>

Compiles CUDA code using the Visual Studio CUDA Toolkit extension on Windows and `nvcc` on Linux. Enabled macros (listed in `src/cuda-exported-variables.lua`):
- `cudaFiles` (Table) -> list of files to be compiled by NVCC to binary (relative path from solution root).
- `cudaPTXFiles` (Table) -> list of files to be compiled by NVCC to PTX (relative path from solution root) - Windows only.
- `cudaRelocatableCode` (Bool) -> triggers -rdc=true.
- `cudaExtensibleWholeProgram` (Bool) -> triggers extensible whole program compilation.
- `cudaCompilerOptions` (Table) -> passed to nvcc.
- `cudaLinkerOptions` (Table) -> passed to nvlink.
- `cudaFastMath` (Bool) -> triggers fast math optimizations.
- `cudaVerbosePTXAS` (Bool) -> triggers code gen verbosity.
- `cudaMaxRegCount` (String) -> number to determine the max used registers.
- `cudaKeep` (Bool) -> keeps preprocessed output.
- `cudaPath` (String) -> custom CUDA install path.
- `cudaGenLineInfo` (Bool) -> generates line info.
- `cudaIntDir` (String) -> Intermediary directory for CUDA files - Windows only.
- `cudaKeepDir` (String) -> Directory to place cudaKeep files (on Linux, requires an existing directory).

The following functions are provided:
- `detectNvccVersion()` -> try to detect the default version of nvcc on the system.
- `detectNvccVersion(cudaPath)` -> try to detect the version of nvcc from a provided path.

----------------
Notes for Windows:
----------------

Files specified by `files` are compiled by `cl` and not `nvcc`.

An example is provided in the test folder where a CUDA executable project containing C++, PTX and CUDA files is linked against a CUDA shared library project. If you clone this repo recursively (i.e. with `-recursive`), it will also pull the premake5 repo, which can be used via the makefile to build premake and then the tests (e.g. via the `nmake` command). You do not need the premake5 repo unless you want to build the tests.

To use:
- Install the CUDA toolkit, along with its Visual Studio integration.
- Copy the premake5-cuda folder to your project.
- Include it in your premake5.lua file as shown in the example.

Tested with Visual Studio 2022 (toolkit v143) with CUDA toolkit 12.1 VS integration.

Note: If PTX is requested, it will be found in the output object folder, with the .obj extension, though, it can be opened with a text editor for inspection.

----------------
Notes for Linux:
----------------

This extension was primarily made for VS on Windows, with the CUDA toolkit extension. Linux is a work in progress and does not have feature parity with Windows yet. Differences are:
- toolset must be set as `"nvcc"`.
- rules must be set to `'cu'`.
- `cudaPTXFiles` and `cudaIntDir` not supported.
- the list of cuda files must be provided in `files` instead of `cudaFiles`.
- unlike Windows, the whole project is compiled by `nvcc` and not just the `.cu` files.

See test premake5 file for how linux config differs. The differences are summarised here:

```
  if os.target() == "windows" then
    cudaFiles { "exe/cu/**.cu" } -- files to be compiled into binaries by VS CUDA.
    cudaPTXFiles { "exe/ptx/**.cu" } -- files to be compiled into ptx, Windows only.
  else
    toolset "nvcc"
    files { "exe/cu/**.cu" }
    rules {"cu"}
  end
```

Admittedly, Linux support is a bit clunky but it should get the job done.
