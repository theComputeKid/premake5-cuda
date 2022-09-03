#* Makefile to build premake-core for tests on windows
PREMAKE_OUT = vendor\premake-core\bin\release\premake5.exe

#* Runs the tests.
all: build
	@echo Testing debug config
	@cd test\out\ExampleProjectExe\bin\debug && .\ExampleProjectExe.exe
	@echo Testing release config
	@cd test\out\ExampleProjectExe\bin\release && .\ExampleProjectExe.exe

build: premake
	@cd test\out && msbuild -p:Configuration=debug -m -v:Normal
	@cd test\out && msbuild -p:Configuration=release -m -v:Normal

premake: $(PREMAKE_OUT)
	@cd test && ..\$(PREMAKE_OUT) vs2022

$(PREMAKE_OUT):
	@cd vendor\premake-core && $(MAKE) -nologo -f Bootstrap.mak windows PLATFORM=x64
