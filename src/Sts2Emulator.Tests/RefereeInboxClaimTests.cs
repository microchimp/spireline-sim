using System.Collections.Concurrent;
using Spireline.Referee;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The referee servers' inbox claim (referee/driver/InboxClaim.cs): many servers race for the same spec
/// files, and each spec must be played exactly once, from a file its claimer can read.
/// </summary>
public class RefereeInboxClaimTests
{
    [Fact]
    public void EverySpecIsClaimedOnceAndItsClaimerCanReadIt()
    {
        string inbox = Path.Combine(Path.GetTempPath(), "spireline-claim-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(inbox);
        try
        {
            const int specs = 600, claimers = 12;
            for (int i = 0; i < specs; i++)
            {
                File.WriteAllText(Path.Combine(inbox, $"{i:D6}.json"), "{\"i\": " + i + "}");
            }

            var claimedBy = new ConcurrentDictionary<string, ConcurrentBag<int>>();
            var unreadable = new ConcurrentBag<string>();
            using var start = new Barrier(claimers);
            var threads = Enumerable.Range(1, claimers).Select(pid => new Thread(() =>
            {
                start.SignalAndWait();
                while (InboxClaim.Claim(inbox, pid) is { } claim)
                {
                    claimedBy.GetOrAdd(claim.id, _ => new ConcurrentBag<int>()).Add(pid);
                    try
                    {
                        File.ReadAllText(claim.taken);
                        File.Delete(claim.taken);
                    }
                    catch (IOException)  // gone or being renamed: another claimer has the same spec
                    {
                        unreadable.Add(claim.id);
                    }
                }
            })).ToList();
            threads.ForEach(t => t.Start());
            threads.ForEach(t => t.Join());

            Assert.Empty(unreadable);
            Assert.Equal(specs, claimedBy.Count);
            Assert.All(claimedBy, kv => Assert.Single(kv.Value));
            Assert.Empty(Directory.GetFiles(inbox));  // no spec, claim or lock left behind
        }
        finally
        {
            Directory.Delete(inbox, recursive: true);
        }
    }
}
