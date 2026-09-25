# ---------------------------------------------------------------------------
# OPTIONAL shortcut for the exact commands in README.md / docs/docker.md.
#
# You do not need this file - it only saves typing the mount flags:
#
#   .\run.ps1 download --all  ==  docker run --rm -it `
#                                   -v "$($PWD.Path)/data:/app/data" `
#                                   -v "$($PWD.Path)/results:/app/results" `
#                                   ghcr.io/iaa-so-training/day4-clustering `
#                                   uv run cluster download --all
#
# Prerequisites: Docker Desktop (section B of the School Software Installation
# Guide) and git. uv and every dependency live inside the image.
#
#   .\run.ps1 download --all         # catalogue + embeddings (~2.2 GB, once)
#   .\run.ps1 run --fast             # the ~2 min warm-up run
#   .\run.ps1 lab                    # JupyterLab, http://localhost:8889
#   .\run.ps1 python scripts/red_clump.py --clusters "NGC 6819"
#   .\run.ps1 shell                  # a shell inside the image
# ---------------------------------------------------------------------------
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Image = if ($env:DAY4_IMAGE) { $env:DAY4_IMAGE } else { "ghcr.io/iaa-so-training/day4-clustering:latest" }
$Tag = if ($env:DAY4_TAG) { $env:DAY4_TAG } else { "day4-clustering" }

$Data = Join-Path $Here "data"
$Results = Join-Path $Here "results"
$Notebooks = Join-Path $Here "notebooks"
New-Item -ItemType Directory -Force -Path $Data, $Results, $Notebooks | Out-Null

function Ensure-Image {
    docker image inspect $Tag *> $null
    if ($LASTEXITCODE -eq 0) { return }
    Write-Host "No local '$Tag' image yet - trying the prebuilt one:"
    Write-Host "  $Image"
    docker pull $Image *> $null
    if ($LASTEXITCODE -eq 0) {
        docker tag $Image $Tag
        Write-Host "pulled."
        return
    }
    Write-Host "  Not available (published from the workshop repository once merged)."
    Write-Host "Building it locally from this folder - a few minutes, once:"
    docker build -t $Tag $Here
    if ($LASTEXITCODE -ne 0) { throw "docker build failed" }
}

# Docker Desktop wants forward slashes in mount paths.
# Forward the tuning knobs documented in the README (CLUSTER_*, NUMBA_*, OMP_*).
$ExtraEnv = @()
Get-ChildItem env: |
    Where-Object { $_.Name -match '^(CLUSTER|NUMBA|OMP)_' } |
    ForEach-Object { $ExtraEnv += @("-e", $_.Name) }

# The image's entrypoint drops to the owner of .\data, so files you create are
# yours and not root's; it also sets HOME and the cache dirs these mounts need.
$Common = @(
    "--rm", "-i",
    "-v", "$($Data.Replace('\', '/')):/app/data",
    "-v", "$($Results.Replace('\', '/')):/app/results",
    "-v", "$($Notebooks.Replace('\', '/')):/app/notebooks",
    "-w", "/app"
) + $ExtraEnv

$Command = if ($Rest.Count -gt 0) { $Rest[0] } else { "help" }

switch ($Command) {
    "help" {
        Write-Host (Get-Content $MyInvocation.MyCommand.Path | Select-Object -First 21 |
            ForEach-Object { $_ -replace '^#\s?', '' })
        exit 0
    }
    { $_ -in "shell", "bash" } {
        Ensure-Image
        docker run @Common -t $Tag bash @($Rest | Select-Object -Skip 1)
        exit $LASTEXITCODE
    }
    { $_ -in "lab", "jupyter", "notebook" } {
        Ensure-Image
        Write-Host "JupyterLab starting - open http://localhost:8889 (no password). Ctrl-C to stop."
        docker run @Common -t -p 8889:8889 $Tag uv run jupyter lab --ip=0.0.0.0 --port=8889 --no-browser "--IdentityProvider.token=" notebooks @($Rest | Select-Object -Skip 1)
        exit $LASTEXITCODE
    }
    { $_ -in "python", "python3", "pytest" } {
        Ensure-Image
        docker run @Common -t $Tag uv run @Rest
        exit $LASTEXITCODE
    }
    { $_ -eq "uv" } {
        Ensure-Image
        docker run @Common -t $Tag @Rest
        exit $LASTEXITCODE
    }
    default {
        Ensure-Image
        docker run @Common -t $Tag uv run cluster @Rest
        exit $LASTEXITCODE
    }
}
