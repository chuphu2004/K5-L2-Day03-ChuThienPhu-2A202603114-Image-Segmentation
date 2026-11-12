# Windows (Docker Desktop, WSL2 backend) equivalent of `deploy.sh`.
# nuctl 1.16.3 runs inside a throwaway docker:cli container with the Docker socket mounted; nothing is installed on the host
# (the nuctl binary is cached in the Docker volume sam3-nuctl so it is downloaded once).
# Usage (CVAT + serverless compose must already be up):  powershell -File autolabel\deploy.ps1 [gpu|cpu]
# gpu: fill autolabel\wheels first (README), otherwise the build pulls ~4 GB of torch CUDA wheels from PyPI in one go.
param([string]$Mode = "gpu")
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $Here
New-Item -ItemType Directory -Force "$Here\weights", "$Here\raw", "$Here\wheels" | Out-Null

if ($Mode -eq "gpu") {
    docker build -t cvat.sam3.base.gpu --build-context "wheels=$Here\wheels" -f "$Here\nuclio\Dockerfile.gpu" "$Here\nuclio"; if ($LASTEXITCODE) { throw "build failed" }
    $File = "function-gpu.yaml"
} else {
    docker build -t cvat.sam3.base "$Here\nuclio"; if ($LASTEXITCODE) { throw "build failed" }
    $File = "function.yaml"
}

# The Docker daemon lives in the WSL2 VM, so volumes for the function container use /run/desktop/mnt/host/<drive>/...
function ToDaemonPath($p) { "/run/desktop/mnt/host/" + $p.Substring(0, 1).ToLower() + $p.Substring(2).Replace("\", "/") }
# nuctl --volume splits on spaces ("Personal Project"), so the mounts go into a copy of the function YAML instead.
$Yaml = Join-Path $env:TEMP "sam3-$File"
(Get-Content -Raw "$Here\nuclio\$File").TrimEnd() + @"

  volumes:
    - volume: {name: data, hostPath: {path: '$(ToDaemonPath "$Repo\data")'}}
      volumeMount: {name: data, mountPath: /data}
    - volume: {name: autolabel, hostPath: {path: '$(ToDaemonPath $Here)'}}
      volumeMount: {name: autolabel, mountPath: /autolabel}
"@ | Set-Content -Encoding ascii $Yaml

# No double quotes in this script: Windows PowerShell 5.1 mangles them in native arguments.
$script = @'
set -e
[ -x /cache/nuctl ] || { wget -q -O /cache/nuctl.tmp https://github.com/nuclio/nuclio/releases/download/1.16.3/nuctl-1.16.3-linux-amd64 && chmod +x /cache/nuctl.tmp && mv /cache/nuctl.tmp /cache/nuctl; }
/cache/nuctl create project cvat --platform local 2>/dev/null || true
/cache/nuctl deploy --project-name cvat --path /work/nuclio --file /function.yaml --platform local
/cache/nuctl get function --platform local
'@
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v sam3-nuctl:/cache -v "${Here}:/work" -v "${Yaml}:/function.yaml" `
    docker:cli sh -c $script
if ($LASTEXITCODE) { throw "deploy failed" }
