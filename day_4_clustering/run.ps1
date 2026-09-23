# ---------------------------------------------------------------------------
# The Day 4 workshop runner for Windows (Docker Desktop + PowerShell 5+).
#
# Prerequisites: Docker (section B of the School Software Installation Guide)
# and git. Everything else lives inside the image.
#
#   .\run.ps1 download --all         # fetch the catalogue + embeddings (~2.5 GB)
#   .\run.ps1 run --fast             # the 2-minute warm-up run
#   .\run.ps1 run --spectral         # the same clustering on spectral embeddings
#   .\run.ps1 marimo                 # the interactive notebook, http://localhost:2718
#   .\run.ps1 shell                  # a shell inside the container
#
# Anything you pass is handed to the `cluster` command inside the container.
# Data and results stay on your machine in .\data and .\results.
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
    Write-Host "Building it locally from this folder - this takes a few minutes once:"
    docker build -t $Tag $Here
    if ($LASTEXITCODE -ne 0) { throw "docker build failed" }
}

# Docker Desktop wants forward slashes in mount paths.
# Forward the tuning knobs documented in the README (CLUSTER_*, NUMBA_*, OMP_*).
$ExtraEnv = @()
Get-ChildItem env: |
    Where-Object { $_.Name -match '^(CLUSTER|NUMBA|OMP)_' } |
    ForEach-Object { $ExtraEnv += @("-e", $_.Name) }

$Common = @(
    "--rm", "-i",
    "-v", "$($Data.Replace('\', '/')):/workspace/data",
    "-v", "$($Results.Replace('\', '/')):/workspace/results",
    "-v", "$($Notebooks.Replace('\', '/')):/workspace/notebooks",
    "-w", "/workspace",
    "-e", "MLFLOW_TRACKING_URI=file:///workspace/results/mlruns"
) + $ExtraEnv

$Command = if ($Rest.Count -gt 0) { $Rest[0] } else { "help" }

switch ($Command) {
    "help" {
        Write-Host (Get-Content $MyInvocation.MyCommand.Path | Select-Object -First 18 |
            ForEach-Object { $_ -replace '^#\s?', '' })
        exit 0
    }
    { $_ -in "shell", "bash" } {
        Ensure-Image
        docker run @Common -t $Tag bash @($Rest | Select-Object -Skip 1)
        exit $LASTEXITCODE
    }
    { $_ -in "marimo", "notebook" } {
        Ensure-Image
        $Notebook = "chemical_tagging.py"
        $RestArgs = @($Rest | Select-Object -Skip 1)
        if ($RestArgs.Count -gt 0 -and -not $RestArgs[0].StartsWith("-")) {
            $Notebook = Split-Path -Leaf $RestArgs[0]
            $RestArgs = @($RestArgs | Select-Object -Skip 1)
        }
        Write-Host "Notebook starting - open http://localhost:2718 (no password). Ctrl-C to stop."
        # Copy any notebook that is missing from the mounted folder and run it
        # from the workspace root, where `data/` lives. Edits are yours: the
        # notebooks/ folder is your checkout, mounted into the container.
        $Shell = "for f in /app/notebooks/*.py; do [ -e `"notebooks/`$(basename `"`$f`")`" ] || cp `"`$f`" notebooks/; done; echo 'running notebooks/$Notebook'; exec marimo edit notebooks/$Notebook --host 0.0.0.0 --no-token `"`$@`""
        docker run @Common -t -p 2718:2718 $Tag sh -c $Shell -- @RestArgs
        exit $LASTEXITCODE
    }
    { $_ -in "python", "python3", "pytest", "cluster", "uv" } {
        # any command that exists inside the image, e.g.
        #   .\run.ps1 python scripts/red_clump.py --clusters "NGC 6819"
        Ensure-Image
        docker run @Common -t $Tag @Rest
        exit $LASTEXITCODE
    }
    default {
        Ensure-Image
        docker run @Common -t $Tag cluster @Rest
        exit $LASTEXITCODE
    }
}
