<div align="center">
   <p>a premake5 extension for cuda</p>
   <h1><small>premake5</small><strong>CUDA</strong></h1>
</div>

Compiles CUDA code using the Visual Studio CUDA Toolkit extension. Enabled macros (listed in `src/cuda-exported-variables.lua`):
- `cudaFiles` (Table) -> list of files to be compiled by NVCC to binary (absolute path from solution root)
- `cudaPTXFiles` (Table) -> list of files to be compiled by NVCC to PTX (absolute path from solution root)
- `cudaRelocatableCode` (Bool) -> triggers -rdc=true
- `cudaExtensibleWholeProgram` (Bool) -> triggers extensible whole program compilation
- `cudaCompilerOptions` (Table) -> passed to nvcc
- `cudaLinkerOptions` (Table) -> passed to nvlink
- `cudaFastMath` (Bool) -> triggers fast math optimizations
- `cudaVerbosePTXAS` (Bool) -> triggers code gen verbosity
- `cudaMaxRegCount` (String) -> number to determine the max used registers
- `cudaKeep` (Bool) -> keeps preprocessed output

Files specified by the premake5 options `files`  are compiled by the MSVC and not nvcc.

An example is provided in the test folder where a CUDA executable project containing C++, PTX and CUDA files is linked against a CUDA shared library project.

To use:
- Copy the premake5-cuda folder to your project
- Include it in your premake5.lua file as shown in the example.

Tested with Visual Studio 2022 (toolkit v143) with CUDA toolkit 11.7 VS integration.

Limitations:
- Available options are currently unable to deal with filters.
