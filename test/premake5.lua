require("../premake5-cuda")

workspace "ExampleProject"

  language "C++"

  cppdialect "C++17"

  location "out"

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

  buildcustomizations "BuildCustomizations/CUDA 11.7"

  externalwarnings "Off" -- thrust gives a lot of warnings
  cudaFiles { "exe/cu/**.cu" } -- files to be compiled into binaries
  cudaPTXFiles { "exe/ptx/**.cu" } -- files to be compiled into ptx
  cudaKeep "On" -- keep temporary output files
  cudaFastMath "On"
  cudaRelocatableCode "On"
  cudaVerbosePTXAS "On"
  cudaMaxRegCount "32"

  vpaths {
    ["Import/*"] = "include/**.hpp",
    ["Sources/*"] = "exe/**.cpp"
  }
  links{ "ExampleProjectDLL" }

--* test excetuable project
project "ExampleProjectDLL"

  kind "SharedLib"

  location "out/%{prj.name}"

  buildcustomizations "BuildCustomizations/CUDA 11.7"

  cudaFiles { "lib/**.cu" }
  cudaFastMath "On"
  cudaRelocatableCode "On"

  defines { "PREMAKE_CUDA_EXPORT_API" }
  cudaCompilerOptions { "-std=c++17" }
  cudaLinkerOptions { "-g" }
