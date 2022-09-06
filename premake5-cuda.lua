require("src/cuda-exported-variables")

if os.target() == "windows" then
    dofile("src/premake5-cuda-vs.lua")
end
