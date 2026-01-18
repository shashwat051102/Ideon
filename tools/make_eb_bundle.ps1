<#
.SYNOPSIS
  Create an Elastic Beanstalk application bundle ZIP for Docker platform.

.DESCRIPTION
  Copies the essential project files into a temporary staging folder and produces
  a timestamped ZIP in output\eb-bundles. Excludes runtime data (storage, logs, output),
  caches, and VCS metadata.

.PARAMETER OutputDir
  Destination directory for the bundle. Default: output\eb-bundles

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File tools\make_eb_bundle.ps1

.EXAMPLE
  powershell -NoProfile -File tools\make_eb_bundle.ps1 -OutputDir E:\bundles
#>

param(
  [string]$OutputDir = "output/eb-bundles"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function New-Timestamp {
  Get-Date -Format 'yyyyMMdd-HHmmss'
}

$root = Split-Path -Parent $PSCommandPath
$root = Split-Path -Parent $root  # project root

# Ensure output folder
$outAbs = Join-Path -Path $root -ChildPath $OutputDir
New-Item -ItemType Directory -Path $outAbs -Force | Out-Null

$stamp = New-Timestamp
$bundleName = "app-$stamp.zip"
$bundlePath = Join-Path $outAbs $bundleName

# Create staging dir
$staging = Join-Path $env:TEMP ("ideon-eb-staging-" + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $staging | Out-Null

Write-Host "[bundle] Staging to: $staging"

# Helper to copy a path if it exists
function Copy-IfExists($src, $dst) {
  if (Test-Path $src) {
    Copy-Item $src -Destination $dst -Recurse -Force
  }
}

# Include top-level files
$includeFiles = @(
  'Dockerfile',
  'Dockerrun.aws.json',
  'requirements.txt',
  'requirements.docker.txt',
  'run.py',
  'run_collective.py',
  'README.md',
  '.ebignore'
)

foreach ($f in $includeFiles) {
  Copy-IfExists (Join-Path $root $f) (Join-Path $staging $f)
}

# Include folders
$includeDirs = @(
  'app',
  'core',
  'configs',
  'research_crew'
)

foreach ($d in $includeDirs) {
  $src = Join-Path $root $d
  if (Test-Path $src) {
    Copy-Item $src -Destination (Join-Path $staging $d) -Recurse -Force
  }
}

# Clean unwanted files from staging
$removeGlobs = @(
  '**/__pycache__',
  '**/*.pyc',
  '.git',
  '.vscode',
  '.pytest_cache',
  'logs',
  'output',
  'storage',
  'tests',
  'data'
)

foreach ($glob in $removeGlobs) {
  Get-ChildItem -Path (Join-Path $staging $glob) -Recurse -Force -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

# Create zip
if (Test-Path $bundlePath) { Remove-Item $bundlePath -Force }
Compress-Archive -Path (Join-Path $staging '*') -DestinationPath $bundlePath -CompressionLevel Optimal

# Cleanup staging
Remove-Item $staging -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "[bundle] Created: $bundlePath"
Write-Output $bundlePath
param(
    [string]$OutFile
)

if (-not $OutFile) {
    $ts = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutFile = "output/eb-bundle-$ts.zip"
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoPath = (Resolve-Path "$root\..\").Path
Set-Location $repoPath

# Build a curated include list to keep bundle lean
$includes = @(
    'Dockerfile',
    'requirements.txt',
    'requirements.docker.txt',
    'run.py',
    'run_collective.py',
    'README.md',
    'LICENSE',
    'app',
    'core',
    'configs',
    'research_crew'
)

$outDir = Split-Path -Parent $OutFile
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir -Force | Out-Null }

$zipPath = Join-Path $repoPath $OutFile
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

Compress-Archive -Path $includes -DestinationPath $zipPath -CompressionLevel Optimal
Write-Host "Created $zipPath"
