-- * Returns the version of nvcc detected on the system, or 0 if undetected.
function detectNvccVersion(cudaPath)
  local errorCode = 1
  local output

  if cudaPath ~= nil then
        output, errorCode = os.outputof(cudaPath .. " --version")
  else
    if os.host() == "linux" then
      output, errorCode = os.outputof("/usr/local/cuda/bin/nvcc --version")
    elseif os.host() == "windows" then
      output, errorCode = os.outputof("nvcc --version")
    else
      return 0
    end
  end

  if errorCode ~= 0 then
    return 0
  end

  local versionString = output:match("cuda_(%S+)")
  return versionString:match("[^.]*.[^.]*")
end
