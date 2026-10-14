#!/usr/bin/env bash
# EmuDump as a NativeAOT executable: tools/emudump/bin/aot/EmuDump.exe. No JIT, so a call costs
# about 0.4 s of CPU less to start and runs optimized code from its first fight. It accepts only
# `fight <out.json> [--compact]` and `reanchor <out.json> [--seed S]...` (checked byte for byte
# against `dotnet EmuDump.dll` on the corpus); everything else needs the JIT build. Rebuild it
# after any engine change, like the JIT build. Offline: the ILCompiler packages come from the
# local NuGet cache, the linker from the Visual Studio Build Tools.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
VSWHERE_DIR="/c/Program Files (x86)/Microsoft Visual Studio/Installer"
if [ -x "$VSWHERE_DIR/vswhere.exe" ]; then
  export PATH="$PATH:$VSWHERE_DIR"  # the ILCompiler targets find link.exe through it
fi
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
rm -f bin/aot/EmuDump.exe  # never leave an old one behind a failed build
log="$(mktemp)"
if ! dotnet publish -c Release -r win-x64 -p:PublishAot=true --source "$HOME/.nuget/packages" -o bin/aot -v q -nologo > "$log" 2>&1; then
  grep -vE "warning (IL|CS)[0-9]+" "$log" | tail -20
  exit 1
fi
echo "built tools/emudump/bin/aot/EmuDump.exe"
