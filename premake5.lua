-- Example configuration file for Premake5 using the CUDA module
-- To build this sample, run '.\premake5.exe vs2019' from the root of this project.

-- Include the premake5 CUDA module
require('premake5-cuda')

workspace "my-cuda-sln"
location "example/my-cuda-sln"

-- Premake5 writes these standard options for the host compiler. 
-- By default, optimization and debug settings are inherited by NVCC.
configurations {"debug", "release"}
architecture "x86_64"

filter "configurations:release"
symbols "Off"
optimize "Full"
filter ""

filter "configurations:debug"
defines {"DEBUG"}
symbols "On"
optimize "Off"
filter ""

-- Start of our C++ CUDA project
project "my-cuda-prj"
location "example/my-cuda-prj"
language "C++"
kind "ConsoleApp"
files "example/*.cpp" -- files compiled by host compiler (e.g. CL.exe)

-- Add necessary build customization using standard Premake5
-- This assumes you have installed Visual Studio integration for CUDA
-- Here we have it set to 11.2 (tested on Update 2)
buildcustomizations "BuildCustomizations/CUDA 11.2"

-- CUDA specific properties
cudaFiles {"example/*.cu"} -- files NVCC compiles
cudaMaxRegCount "32"

-- Let's compile for all supported architectures (and also in parallel with -t0)
cudaCompilerOptions {"-arch=sm_52", "-gencode=arch=compute_52,code=sm_52", "-gencode=arch=compute_60,code=sm_60",
                     "-gencode=arch=compute_61,code=sm_61", "-gencode=arch=compute_70,code=sm_70",
                     "-gencode=arch=compute_75,code=sm_75", "-gencode=arch=compute_80,code=sm_80",
                     "-gencode=arch=compute_86,code=sm_86", "-gencode=arch=compute_86,code=compute_86", "-t0"} 

filter "configurations:release"
cudaFastMath "On" -- enable fast math for release
filter ""
