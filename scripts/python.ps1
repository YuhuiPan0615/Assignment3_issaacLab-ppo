# Run the project's Conda prefix environment without changing the global PATH.
# Example: .\scripts\python.ps1 -m scripts.build_scene
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskEnvironment = Join-Path $taskRoot '.venv'
$taskCondaPython = Join-Path $taskEnvironment 'python.exe'
$taskVenvPython = Join-Path $taskEnvironment 'Scripts\python.exe'
if (Test-Path -LiteralPath $taskCondaPython) {
    $taskPython = $taskCondaPython
}
elseif (Test-Path -LiteralPath $taskVenvPython) {
    $taskPython = $taskVenvPython
}
else {
    $taskPython = $null
}
if (-not $taskPython -or -not (Test-Path -LiteralPath $taskPython)) {
    throw 'Project .venv is missing. Create the pinned Python 3.12 environment; see README.md.'
}
$taskRuntimeTemp = Join-Path $taskRoot '.cache\tmp-runtime'
$taskWarpCache = Join-Path $taskRoot '.cache\warp'
New-Item -ItemType Directory -Force -Path $taskRuntimeTemp, $taskWarpCache | Out-Null
$taskPreviousPath = $env:PATH
$taskPreviousPythonPath = $env:PYTHONPATH
$taskPreviousPythonHome = $env:PYTHONHOME
$taskPreviousNoUserSite = $env:PYTHONNOUSERSITE
$taskPreviousOmniEula = $env:OMNI_KIT_ACCEPT_EULA
$taskPreviousPythonUtf8 = $env:PYTHONUTF8
$taskPreviousPythonIoEncoding = $env:PYTHONIOENCODING
$taskPreviousTemp = $env:TEMP
$taskPreviousTmp = $env:TMP
$taskPreviousWarpCache = $env:WARP_CACHE_PATH
$taskExitCode = 1
try {
    $env:PATH = "$taskEnvironment;$taskEnvironment\Scripts;$taskEnvironment\Library\bin;$taskPreviousPath"
    $env:PYTHONPATH = $null
    $env:PYTHONHOME = $null
    $env:PYTHONNOUSERSITE = '1'
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    # The project owner explicitly accepted the NVIDIA Omniverse EULA before
    # this setting was added. It keeps non-interactive Isaac checks runnable.
    $env:OMNI_KIT_ACCEPT_EULA = 'YES'
    # Keep large simulator downloads and compiled Warp kernels on the project drive.
    $env:TEMP = $taskRuntimeTemp
    $env:TMP = $taskRuntimeTemp
    $env:WARP_CACHE_PATH = $taskWarpCache
    & $taskPython @args
    $taskExitCode = $LASTEXITCODE
}
finally {
    $env:PATH = $taskPreviousPath
    $env:PYTHONPATH = $taskPreviousPythonPath
    $env:PYTHONHOME = $taskPreviousPythonHome
    $env:PYTHONNOUSERSITE = $taskPreviousNoUserSite
    $env:OMNI_KIT_ACCEPT_EULA = $taskPreviousOmniEula
    $env:PYTHONUTF8 = $taskPreviousPythonUtf8
    $env:PYTHONIOENCODING = $taskPreviousPythonIoEncoding
    $env:TEMP = $taskPreviousTemp
    $env:TMP = $taskPreviousTmp
    $env:WARP_CACHE_PATH = $taskPreviousWarpCache
}
exit $taskExitCode
