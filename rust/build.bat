@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set CARGO_TARGET_DIR=C:\Users\chrom\AppData\Local\rust_target
C:\Users\chrom\.cargo\bin\cargo.exe build
