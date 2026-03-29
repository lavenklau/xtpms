#Requires -Version 5.0
<#
.SYNOPSIS
  Configure CMake (Visual Studio 2019 + vcpkg, Release-only) or Ninja + Release.

.DESCRIPTION
  Sets VCPKG_ROOT when unset (default C:\vcpkg), then runs cmake --preset from the repo root.

.PARAMETER VcpkgRoot
  vcpkg installation directory. Overrides VCPKG_ROOT for this run.

.PARAMETER Preset
  CMake configure preset name (see CMakePresets.json).

.PARAMETER Ninja
  Shorthand for -Preset vcpkg-ninja-release (output under build-ninja).

.EXAMPLE
  .\configure.ps1

.EXAMPLE
  .\configure.ps1 -VcpkgRoot D:\tools\vcpkg

.EXAMPLE
  .\configure.ps1 -Ninja
#>
param(
	[string] $VcpkgRoot = $env:VCPKG_ROOT,
	[ValidateSet("vs2019-vcpkg", "vcpkg-ninja-release")]
	[string] $Preset = "vs2019-vcpkg",
	[switch] $Ninja
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
if (-not (Test-Path (Join-Path $RepoRoot "CMakeLists.txt"))) {
	Write-Error "Run this script from the repository root (CMakeLists.txt missing in $RepoRoot)."
}

if ($Ninja) {
	$Preset = "vcpkg-ninja-release"
}

if (-not $VcpkgRoot) {
	$VcpkgRoot = "C:\vcpkg"
}
$env:VCPKG_ROOT = $VcpkgRoot

$Toolchain = Join-Path $VcpkgRoot "scripts\buildsystems\vcpkg.cmake"
if (-not (Test-Path -LiteralPath $Toolchain)) {
	Write-Error "Toolchain file not found: $Toolchain`nSet -VcpkgRoot or environment variable VCPKG_ROOT."
}

Push-Location $RepoRoot
try {
	Write-Host "VCPKG_ROOT=$($env:VCPKG_ROOT)"
	Write-Host "cmake --preset $Preset"
	& cmake --preset $Preset
	if ($LASTEXITCODE -ne 0) {
		exit $LASTEXITCODE
	}
}
finally {
	Pop-Location
}

Write-Host ""
Write-Host "Next: cmake --build --preset release"
Write-Host "      ctest -C Release --output-on-failure   (from build directory, or use --test-dir build)"
