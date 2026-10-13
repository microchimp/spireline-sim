// The server inbox's claim protocol (Server.cs), kept free of game types so the test project can
// race it directly (src/Sts2Emulator.Tests/RefereeInboxClaimTests.cs).

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
    /// open handle, so two servers that both open "&lt;id&gt;.json" before either renames both succeed, and
    /// the file ends up under the second name only (night04: 4 specs played twice, one copy logged as
    /// "bad spec file: Could not find file ...taken.&lt;pid&gt;"). So the rename runs under a per-spec lock,
    /// "&lt;id&gt;.json.lock", created with FileMode.CreateNew (one creator wins) and removed right after:
    /// a server that gets the lock later finds the source gone. The lock holds the pid, so the
    /// dispatcher can remove one left by a server that died holding it.
    /// </summary>
    public static (string taken, string id)? Claim(string inbox, int pid)
    {
        foreach (FileInfo f in Pending(inbox).OrderBy(f => f.LastWriteTimeUtc).ThenBy(f => f.Name, StringComparer.Ordinal))
        {
            string taken = f.FullName + ".taken." + pid;
            string lockPath = f.FullName + ".lock";
            try
            {
                using var lk = new FileStream(lockPath, FileMode.CreateNew, FileAccess.Write, FileShare.None);
                lk.Write(System.Text.Encoding.ASCII.GetBytes(pid.ToString()));
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                continue;  // another server is claiming it
            }

            bool moved;
            try
            {
                File.Move(f.FullName, taken);
                moved = true;
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                moved = false;  // another server took it before we locked
            }
            finally
            {
                File.Delete(lockPath);
            }

            if (moved)
            {
                return (taken, f.Name[..^".json".Length]);
            }
        }

        return null;
    }
}
