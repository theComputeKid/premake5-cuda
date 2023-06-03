#* Makefile to build premake-core for tests on windows
PREMAKE_OUT = vendor\premake-core\bin\release\premake5.exe

#* Runs the tests.
all: build
	@echo Testing debug config
	@cd test\out\bin\debug && .\ExampleProjectExe.exe && .\ExampleProjectExeNonCUDA.exe
	@echo Testing release config
	@cd test\out\bin\release && .\ExampleProjectExe.exe && .\ExampleProjectExeNonCUDA.exe

build: premake
	@cd test\out && msbuild -p:Configuration=debug -m -v:Normal
	@cd test\out && msbuild -p:Configuration=release -m -v:Normal

premake: $(PREMAKE_OUT)
	@$(PREMAKE_OUT) --file=test\premake5.lua vs2022

$(PREMAKE_OUT):
	@cd vendor\premake-core && $(MAKE) -nologo -f Bootstrap.mak windows PLATFORM=x64

clean:
	@if exist $(PREMAKE_OUT) del /s /q $(PREMAKE_OUT)
	@if exist test\out rmdir /s /q test\out
