// Check for referee/driver/ResultsFile.cs (the driver has no unit-test project): dotnet run -c Release --project tools/resultsfile_check
// The file after every Append is a complete JSON document, byte for byte what serialising the whole list gives.
using System.Diagnostics;
using System.Text.Json.Nodes;
using Spireline.Referee;

// Synthetic result shaped like a fight result: some scalars plus a list of step rows, non-ASCII text included.
static JsonObject Make(int i, int rows)
{
    var steps = new JsonArray();
    for (int r = 0; r < rows; r++) steps.Add(new JsonArray("A", "Play STRIKE_IRONCLAD \u00e9\u4e2d", r, i, "x" + (r * 7919 % 1000)));
    return new JsonObject { ["seed"] = "S" + i, ["end"] = "win", ["hp"] = 80 - i % 30, ["note"] = "caf\u00e9 \"q\" " + (char)92 + " \u4e2d", ["steps"] = steps };
}

static string Old(IReadOnlyList<JsonObject> all)
{
    var arr = new JsonArray();
    foreach (var o in all) arr.Add(o.DeepClone());
    return new JsonObject { ["results"] = arr.DeepClone() }.ToJsonString();
}

string dir = args.Length > 0 ? args[0] : Path.Combine(Path.GetTempPath(), "resultsfile-check-" + Environment.ProcessId);
Directory.CreateDirectory(dir);
int fails = 0;

// 1. Byte equality with the whole-file save and validity after every append, 0 / 1 / 40 results.
{
    string p = Path.Combine(dir, "t1.json");
    if (File.Exists(p)) File.Delete(p);
    var rf = new ResultsFile(p);
    var all = new List<JsonObject>();
    rf.Finish();
    if (File.ReadAllText(p) != "{\"results\":[]}") { Console.WriteLine("FAIL empty doc: " + File.ReadAllText(p)); fails++; }
    File.Delete(p);
    rf = new ResultsFile(p);
    for (int i = 0; i < 40; i++)
    {
        var o = Make(i, 3 + i % 5);
        all.Add(JsonNode.Parse(o.ToJsonString())!.AsObject());
        rf.Append(o);
        string got = File.ReadAllText(p);
        if (got != Old(all)) { Console.WriteLine($"FAIL bytes differ after {i + 1}"); fails++; break; }
        var doc = JsonNode.Parse(got)!;
        if (doc["results"]!.AsArray().Count != i + 1) { Console.WriteLine("FAIL count"); fails++; break; }
    }
    rf.Finish();  // after results: must not change the file
    if (File.ReadAllText(p) != Old(all)) { Console.WriteLine("FAIL Finish changed a non-empty file"); fails++; }
    Console.WriteLine($"t1 bytes+validity over 40 appends (utf-8, escapes): {(fails == 0 ? "OK" : "FAIL")}");
}

// 2. An old larger file at the path is replaced by the first result.
{
    string p = Path.Combine(dir, "t2.json");
    File.WriteAllText(p, new string('x', 100000));
    var rf = new ResultsFile(p);
    rf.Append(Make(0, 2));
    if (File.ReadAllText(p) != Old(new List<JsonObject> { Make(0, 2) })) { Console.WriteLine("FAIL stale file not replaced"); fails++; } else Console.WriteLine("t2 stale file replaced: OK");
}

// 3. A reader may hold the file open (Python-style share) while appends continue.
{
    string p = Path.Combine(dir, "t3.json");
    var rf = new ResultsFile(p);
    rf.Append(Make(0, 2));
    using var reader = new FileStream(p, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
    rf.Append(Make(1, 2));
    Console.WriteLine("t3 append while a reader holds the file open: OK");
}

try { Directory.Delete(dir, true); } catch (IOException) { }
Console.WriteLine(fails == 0 ? "ALL OK" : $"{fails} FAILED");
return fails;
