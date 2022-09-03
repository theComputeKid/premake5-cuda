#* Makefile to build premake-core for tests on windows
PREMAKE_OUT = vendor\premake-core\bin\release\premake5.exe

$(PREMAKE_OUT):
	@cd vendor\premake-core && $(MAKE) -nologo -f Bootstrap.mak windows PLATFORM=x64
