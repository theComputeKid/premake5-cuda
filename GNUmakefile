#* Makefile to build premake-core for tests on windows
PREMAKE_OUT := vendor/premake-core/bin/release/premake5

#* Runs the tests.
all: build
	@echo Testing debug config
	@cd test/out/bin/debug && ./ExampleProjectExe && ./ExampleProjectExeNonCUDA
	@echo Testing release config
	@cd test/out/bin/release && ./ExampleProjectExe && ./ExampleProjectExeNonCUDA

build: premake
	@cd test/out && $(MAKE) config=debug
	@cd test/out && $(MAKE) config=release

premake: $(PREMAKE_OUT)
	@$(PREMAKE_OUT) --file=test/premake5.lua gmake2

$(PREMAKE_OUT):
	@cd vendor/premake-core && $(MAKE) -f Bootstrap.mak linux PLATFORM=x64

clean:
	@rm -f $(PREMAKE_OUT)
	@rm -rf test/out
