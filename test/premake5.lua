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

--* test excetuable project
project "ExampleProjectExe"

  kind "ConsoleApp"

  location "out/%{prj.name}"

  files {
      "exe/**.cpp",
      "exe/**.hpp",
  }

  vpaths {
    ["Import/*"] = "include/**.hpp",
    ["Sources/*"] = "exe/**.cpp",
    ["Sources/*"] = "exe/**.cpp",
  }
