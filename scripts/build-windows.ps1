param(
    [string]$Backend = "",
    [string]$CudaArch = "",
    [string]$RocmArch = "",
    [string]$BuildProfile = "",
    [switch]$DynamicHost,
    [switch]$HostOnly
)

# Stable workspace entrypoint; implementation belongs to Mesh.
$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "../mesh/scripts/build-windows.ps1") @PSBoundParameters
exit $LASTEXITCODE
