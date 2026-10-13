using System.Collections.Concurrent;
using Spireline.Referee;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The referee servers' inbox claim (referee/driver/InboxClaim.cs). Each server has an inbox of its own
/// (serve.Pool), so a claim is a plain rename: every spec must be played exactly once, from a file its
/// claimer can read, oldest first, and a spec that vanished after it was listed is skipped.
/// </summary>
public class RefereeInboxClaimTests
{
    private static string NewDir()
    {
        string d = Path.Combine(Path.GetTempPath(), "spireline-claim-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(d);
        return d;
    }

    [Fact]
    public void ServersClaimingFromTheirOwnInboxesAtOnceEachPlayEverySpecOnce()
    {
        string root = NewDir();
        try
        {
            const int servers = 12, specs = 50;
            for (int s = 0; s < servers; s++)
            {
                Directory.CreateDirectory(Path.Combine(root, $"s{s:D2}"));
                for (int i = 0; i < specs; i++)
                {
                    File.WriteAllText(Path.Combine(root, $"s{s:D2}", $"{s:D2}{i:D4}.json"), "{\"i\": " + i + "}");
                }
            }

            var claimedBy = new ConcurrentDictionary<string, ConcurrentBag<int>>();
            var unreadable = new ConcurrentBag<string>();
            using var start = new Barrier(servers);
            var threads = Enumerable.Range(0, servers).Select(s => new Thread(() =>
            {
                string inbox = Path.Combine(root, $"s{s:D2}");
                start.SignalAndWait();
                while (InboxClaim.Claim(inbox, 1000 + s) is { } claim)
                {
                    claimedBy.GetOrAdd(claim.id, _ => new ConcurrentBag<int>()).Add(s);
                    try
                    {
                        File.ReadAllText(claim.taken);
                        File.Delete(claim.taken);
                    }
                    catch (IOException)
                    {
                        unreadable.Add(claim.id);
                    }
                }
            })).ToList();
            threads.ForEach(t => t.Start());
            threads.ForEach(t => t.Join());

            Assert.Empty(unreadable);
            Assert.Equal(servers * specs, claimedBy.Count);
            Assert.All(claimedBy, kv => Assert.Single(kv.Value));
            Assert.Empty(Directory.GetFiles(root, "*", SearchOption.AllDirectories));  // nothing left behind
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void TheOldestSpecIsClaimedFirstAndOtherFilesAreLeftAlone()
    {
        string inbox = NewDir();
        try
        {
            var t0 = DateTime.UtcNow.AddMinutes(-10);
            foreach (var (name, age) in new[] { ("000002.json", 1), ("000001.json", 3), ("000003.json", 3) })
            {
                string f = Path.Combine(inbox, name);
                File.WriteAllText(f, "{}");
                File.SetLastWriteTimeUtc(f, t0.AddMinutes(-age));
            }

            foreach (string other in new[] { "STOP", "hold.7", "000009.tmp", "000008.json.taken.5", "000007.json.withdrawn" })
            {
                File.WriteAllText(Path.Combine(inbox, other), "");
            }

            var order = new List<string>();
            while (InboxClaim.Claim(inbox, 7) is { } claim)
            {
                order.Add(claim.id);
                Assert.True(File.Exists(claim.taken));
                Assert.EndsWith(".json.taken.7", claim.taken);
            }

            // oldest write first, then by name among equals
            Assert.Equal(new[] { "000001", "000003", "000002" }, order);
            Assert.Equal(5 + 3, Directory.GetFiles(inbox).Length);  // the other files untouched, the claims kept
        }
        finally
        {
            Directory.Delete(inbox, recursive: true);
        }
    }

    [Fact]
    public void ASpecGoneSinceItWasListedIsSkipped()
    {
        string inbox = NewDir();
        try
        {
            string gone = Path.Combine(inbox, "000001.json"), next = Path.Combine(inbox, "000002.json");
            File.WriteAllText(gone, "{}");
            File.WriteAllText(next, "{}");
            var listed = new[] { new FileInfo(gone), new FileInfo(next) };
            File.Move(gone, gone + ".withdrawn");  // withdrawn after the listing

            var claim = InboxClaim.ClaimFrom(listed, 7);

            Assert.Equal("000002", claim?.id);
            Assert.True(File.Exists(gone + ".withdrawn"));
            Assert.Null(InboxClaim.Claim(inbox, 7));
        }
        finally
        {
            Directory.Delete(inbox, recursive: true);
        }
    }
}
