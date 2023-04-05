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

  buildcustomizations "BuildCustomizations/CUDA 12.1"

  cudaFiles { "lib/**.cu" }
  cudaRelocatableCode "On"

  defines { "PREMAKE_CUDA_EXPORT_API" }

  -- Let's compile for all supported architectures (and also in parallel with -t0)
  cudaCompilerOptions {"-arch=all", "-t0"} 

  filter "configurations:debug"
    cudaLinkerOptions { "-g" }
  filter {}

  filter "configurations:release"
    cudaFastMath "On"
  filter {}
