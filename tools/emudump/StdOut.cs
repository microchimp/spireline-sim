// Writes a whole document to stdout in one go. Console.WriteLine goes through an auto-flushing
// writer with a small buffer: a fight dump of a few megabytes became thousands of write calls,
// as much CPU as building the dump. The bytes are the same: the text in Console's own output
// encoding, then Console's newline.
static class StdOut
{
    internal static void WriteLine(string text)
    {
        Console.Out.Flush();
        var bytes = Console.OutputEncoding.GetBytes(text + Console.Out.NewLine);
        using var stdout = Console.OpenStandardOutput();
        stdout.Write(bytes, 0, bytes.Length);
        stdout.Flush();
    }
}
