require("../premake5-cuda")

workspace "ExampleProject"

  language "C++"

  cppdialect "C++17"

  location "out"

  targetdir("out/bin/%{cfg.buildcfg}")

  configurations {
    "debug",
    "release"
  }

  architecture "x64"

  warnings "Extra"

  flags {
    "FatalWarnings",
    "RelativeLinks",
    "MultiProcessorCompile"
  }

  filter "configurations:release"
    symbols "Off"
    optimize "Full"
    runtime "Release"

  filter "configurations:debug"
    defines {"DEBUG"}
    symbols "On"
    optimize "Off"
    runtime "Debug"
  filter {}

  includedirs { "include" }

--* test executable project
project "ExampleProjectExe"

  kind "ConsoleApp"

  location "out/%{prj.name}"

  files { "exe/cpp/**.cpp" }

  buildcustomizations "BuildCustomizations/CUDA 12.1"

  externalwarnings "Off" -- thrust gives a lot of warnings

  if os.target() == "windows" then
    cudaFiles { "exe/cu/**.cu" } -- files to be compiled into binaries by VS CUDA.
    cudaPTXFiles { "exe/ptx/**.cu" } -- files to be compiled into ptx, Windows only.
  else
    toolset "nvcc"
    cudaPath "/usr/local/cuda"
    files { "exe/cu/**.cu" }
    rules {"cu"}
  end

  cudaKeep "On" -- keep temporary output files
  cudaFastMath "On"
  cudaRelocatableCode "On"
  cudaVerbosePTXAS "On"
  cudaMaxRegCount "32"
  cudaIntDir "out/bin/cudaobj/%{cfg.buildcfg}"

  vpaths {
    ["Import/*"] = "include/**.hpp",
    ["Sources/*"] = "exe/**.cpp"
  }
  links{ "ExampleProjectDLL" }

--* test excetuable project
project "ExampleProjectDLL"

  kind "SharedLib"

  location "out/%{prj.name}"

  buildcustomizations "BuildCustomizations/CUDA 12.1"

  if os.target() == "windows" then
    -- Just in case we want the VS CUDA extension to use a custom version of CUDA
    cudaPath "$(CUDA_PATH)"
    cudaFiles { "lib/**.cu" }
  else
    toolset "nvcc"
    cudaPath "/usr/local/cuda"
    files { "lib/**.cu" }
    rules {"cu"}
  end

  cudaRelocatableCode "On"

  defines { "PREMAKE_CUDA_EXPORT_API" }

  -- Let's compile for all supported architectures (and also in parallel with -t0)
  cudaCompilerOptions {"-arch=all", "-t0"} 

  filter "configurations:debug"
    cudaLinkerOptions { "-g" }
  filter {}

  filter "configurations:release"
    cudaFastMath "On"
    cudaGenLineInfo "On"
  filter {}

--* test executable project (non-CUDA)
project "ExampleProjectExeNonCUDA"

kind "ConsoleApp"

location "out/%{prj.name}"

files { "nonCuda/**.cpp" }
