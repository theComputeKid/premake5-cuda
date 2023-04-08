<div align="center">
   <p>a premake5 extension for cuda</p>
   <h1><small>premake5</small><strong>CUDA</strong></h1>
</div>

Compiles CUDA code using the Visual Studio CUDA Toolkit extension. Enabled macros (listed in `src/cuda-exported-variables.lua`):
- `cudaFiles` (Table) -> list of files to be compiled by NVCC to binary (relative path from solution root)
- `cudaPTXFiles` (Table) -> list of files to be compiled by NVCC to PTX (relative path from solution root)
- `cudaRelocatableCode` (Bool) -> triggers -rdc=true
- `cudaExtensibleWholeProgram` (Bool) -> triggers extensible whole program compilation
- `cudaCompilerOptions` (Table) -> passed to nvcc
- `cudaLinkerOptions` (Table) -> passed to nvlink
- `cudaFastMath` (Bool) -> triggers fast math optimizations
- `cudaVerbosePTXAS` (Bool) -> triggers code gen verbosity
- `cudaMaxRegCount` (String) -> number to determine the max used registers
- `cudaKeep` (Bool) -> keeps preprocessed output
- `cudaPath` (String) -> custom CUDA install path to override the VS integration plugin
- `cudaGenLineInfo` (Bool) -> generates line info

Files specified by the premake5 options `files`  are compiled by the MSVC and not nvcc.

An example is provided in the test folder where a CUDA executable project containing C++, PTX and CUDA files is linked against a CUDA shared library project. If you clone this repo recursively (i.e. with `-recursive`), it will also pull the premake5 repo, which can be used via the makefile to build premake and then the tests (e.g. via the `nmake` command). You do not need the premake5 repo unless you want to build the tests.

To use:
- Install the CUDA toolkit, along with its Visual Studio integration.
- Copy the premake5-cuda folder to your project
- Include it in your premake5.lua file as shown in the example.

Tested with Visual Studio 2022 (toolkit v143) with CUDA toolkit 12.1 VS integration.

Note: If PTX is requested, it will be found in the output object folder, with the .obj extension, though, it can be opened with a text editor for inspection.
