// The "runs" job's out file: one JSON document {"results":[r1,r2,...]} that grows by one result per
// fight. Each Append writes only the new result (cost independent of how many came before) and leaves
// the file a complete, valid document, byte for byte what serialising the whole list would give, so
// every reader (batch.py, probe.py, refset.py, runs.py, scenarios.py, envproto.py) is unchanged.
using System.IO;
using System.Text;
using System.Text.Json.Nodes;
using Microsoft.Win32.SafeHandles;

namespace Spireline.Referee;

internal sealed class ResultsFile
{
    private const string Head = "{\"results\":[";
    private const string Tail = "]}";
    private readonly string _path;
    private readonly object _lock = new();
    private long _tailAt;  // offset of the closing "]}" in the file; -1 until the first result is written
    private int _count;

    internal ResultsFile(string path)
    {
        _path = path;
        _tailAt = -1;
    }

    internal int Count => _count;

    /// <summary>Milliseconds the last Append took (serialising the result and writing it), and all of them together.</summary>
    internal double LastMs { get; private set; }

    internal double TotalMs { get; private set; }

    /// <summary>
    /// Adds one result. The first one replaces the old file atomically (temp file, then move), as the
    /// whole-file save always did. Later ones overwrite the closing "]}" with ",result]}" in one write
    /// call at a known offset: the file only ever grows, and a kill before the call leaves the previous
    /// complete document. The result is serialised here, so the caller keeps no reference to it.
    /// </summary>
    internal void Append(JsonNode result)
    {
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        string json = result.ToJsonString();
        lock (_lock)
        {
            if (_tailAt < 0)
            {
                WriteWhole(Head + json + Tail);
                _tailAt = Encoding.UTF8.GetByteCount(Head + json);
            }
            else
            {
                byte[] bytes = Encoding.UTF8.GetBytes("," + json + Tail);
                using SafeFileHandle h = File.OpenHandle(_path, FileMode.Open, FileAccess.Write, FileShare.ReadWrite | FileShare.Delete);
                RandomAccess.Write(h, bytes, _tailAt);
                _tailAt += bytes.Length - Tail.Length;
            }

            _count++;
            LastMs = System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
            TotalMs += LastMs;
        }
    }

    /// <summary>Makes sure the file exists: a job that produced no result still leaves {"results":[]}.</summary>
    internal void Finish()
    {
        lock (_lock)
        {
            if (_tailAt < 0)
            {
                WriteWhole(Head + Tail);
                _tailAt = Encoding.UTF8.GetByteCount(Head);
            }
        }
    }

    private void WriteWhole(string text)
    {
        string tmp = _path + ".tmp";
        File.WriteAllText(tmp, text);
        File.Move(tmp, _path, overwrite: true);
    }
}
