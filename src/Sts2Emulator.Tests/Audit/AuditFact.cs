using Xunit;

namespace Sts2Emulator.Tests.Audit;

/// <summary>
/// A test that encodes the game's behaviour where the emulator is KNOWN or SUSPECTED to differ
/// (static audit pilot). It is skipped in the normal suite so the suite stays green; run the
/// audit with SPIRELINE_AUDIT=1 to see which ones fail:
///
///   SPIRELINE_AUDIT=1 dotnet test src/Sts2Emulator.Tests --no-build --filter "FullyQualifiedName~Sts2Emulator.Tests.Audit"
///
/// A test whose bug gets fixed should switch to a plain FactAttribute.
/// </summary>
public sealed class AuditFactAttribute : FactAttribute
{
    public AuditFactAttribute(string reason)
    {
        if (Environment.GetEnvironmentVariable("SPIRELINE_AUDIT") != "1")
        {
            Skip = "audit: " + reason;
        }
    }
}
