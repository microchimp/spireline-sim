// The server inbox's claim protocol (Server.cs), kept free of game types so the test project can
// exercise it directly (src/Sts2Emulator.Tests/RefereeInboxClaimTests.cs).

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Spireline.Referee;

public static class InboxClaim
{
    public static IEnumerable<FileInfo> Pending(string inbox) =>
        new DirectoryInfo(inbox).GetFiles("*.json").Where(f => f.Name.EndsWith(".json", StringComparison.Ordinal));

    /// <summary>
    /// The oldest pending spec, claimed by renaming it to "&lt;id&gt;.json.taken.&lt;pid&gt;"; null when there is none.
    /// A rename alone is not exclusive on Windows: MoveFileEx opens the source by name and renames the
    /// open handle, so two processes that both open "&lt;id&gt;.json" before either renames both succeed, and
    /// the file ends up under the second name only (night04, one inbox shared by every server: 4 specs
    /// played twice, one copy logged as "bad spec file: Could not find file ...taken.&lt;pid&gt;"). So no two
    /// processes ever rename in one inbox at once: serve.py gives each server an inbox of its own, writes
    /// new specs there only under their final name (a tmp file renamed in), and withdraws specs from it
    /// only once the server's status says it holds or the server is dead (serve.Pool.close). A claim
    /// whose file is gone after the rename (the guard that remains) is skipped, not answered.
    /// </summary>
    public static (string taken, string id)? Claim(string inbox, int pid) =>
        ClaimFrom(Pending(inbox).OrderBy(f => f.LastWriteTimeUtc).ThenBy(f => f.Name, StringComparer.Ordinal), pid);

    /// <summary>The first of <paramref name="candidates" /> this process can claim (Claim passes them oldest first).</summary>
    public static (string taken, string id)? ClaimFrom(IEnumerable<FileInfo> candidates, int pid)
    {
        foreach (FileInfo f in candidates)
        {
            string taken = f.FullName + ".taken." + pid;
            try
            {
                File.Move(f.FullName, taken);
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                continue;  // gone since it was listed
            }

            if (File.Exists(taken))
            {
                return (taken, f.Name[..^".json".Length]);
            }
        }

        return null;
    }
}
