cd mlc-llm
rd /s /q build
mkdir build

echo set(USE_VULKAN ON) >> config.cmake

REM Conda ships a GNU coreutils link.exe in %CONDA_PREFIX%\Library\usr\bin that
REM shadows the MSVC linker on PATH and breaks the Rust (cargo) build of the
REM tokenizers crate. Remove it so cargo/rustc pick up the MSVC link.exe.
if exist "%CONDA_PREFIX%\Library\usr\bin\link.exe" del /f /q "%CONDA_PREFIX%\Library\usr\bin\link.exe"

pip install . -v

if %errorlevel% neq 0 exit %errorlevel%
