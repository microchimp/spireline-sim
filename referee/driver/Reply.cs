using System.Text.Json.Nodes;

namespace Spireline.Referee;

/// <summary>
/// What a line the agent sends to an open decision asks for (Env.Decide). {"a": K} with 0 &lt;= K &lt; n answers it; {"cmd":"quit"}
/// ends the session; {"cmd":"anchor_state"} asks for the open decision's hidden state (Env.AnchorStateOf), answered with
/// {"t":"anchor_state","state":{...}} and the decision left open without being sent again (lane 22, fix d: the lockstep re-anchors
/// its copy inside a fight at the decision where a difference is found). The command is known only to a run whose reset asked for
/// the hidden state (spec "anchor_state"); in any other run it is a bad answer, as it always was (bad_action and the decision
/// again). Anything else is a bad answer. Plain values in and out, so the rule is testable without the game (referee/driver.Tests).
/// </summary>
internal static class Reply
{
    internal enum What { Answer, Quit, AnchorState, Bad }

    internal static (What what, int index) Of(JsonObject reply, int n, bool anchorState)
    {
        string? cmd = reply["cmd"] is JsonValue c && c.TryGetValue(out string? s) ? s : null;
        if (cmd == "quit") return (What.Quit, -1);
        if (cmd == "anchor_state" && anchorState) return (What.AnchorState, -1);
        int a = reply["a"] is JsonValue v && v.TryGetValue(out int k) ? k : -1;
        return a >= 0 && a < n ? (What.Answer, a) : (What.Bad, -1);
    }
}
