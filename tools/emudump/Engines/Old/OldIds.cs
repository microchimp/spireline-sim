using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace EmuDump.Engines.Old;

/// <summary>The old engine's card def ids and the game's model ids (UPPER_SNAKE), through data/id_map.json.</summary>
static class OldIds
{
    static Dictionary<string, int>? _bySnake;
    static Dictionary<int, string>? _byId;

    static Dictionary<string, int> BySnake()
    {
        if (_bySnake == null)
        {
            var map = JsonNode.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "id_map.json")))!["cards"]!.AsObject();
            _bySnake = map.ToDictionary(kv => Regex.Replace(kv.Key,
                "(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_").ToUpperInvariant(), kv => (int)kv.Value!);
        }

        return _bySnake;
    }

    /// <summary>Game card id (UPPER_SNAKE) to the emulator's def id, through data/id_map.json.</summary>
    internal static int CardIdBySnake(string snake) => BySnake()[snake];

    internal static bool TryCardIdBySnake(string snake, out int id) => BySnake().TryGetValue(snake, out id);

    /// <summary>
    /// A def id's game model id. The map is one to one, so equal ids give equal names; an id outside it (a pile
    /// index past the end, a source that is not a card) gets a name of its own, which no recording uses.
    /// </summary>
    internal static string SnakeByCardId(int id)
    {
        if (_byId == null)
        {
            var byId = new Dictionary<int, string>();
            foreach (var (snake, def) in BySnake()) byId[def] = snake;
            _byId = byId;
        }

        return _byId.TryGetValue(id, out var s) ? s : $"UNMAPPED_{id}";
    }
}
