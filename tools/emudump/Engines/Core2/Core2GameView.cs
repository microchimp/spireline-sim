// The kernel's own state in the core2 engine's game view (IEngine.GameView), written only when a switch is on
// (docs/rebuild/TIER1_KERNEL.md section 6, docs/rebuild/COMPARER.md). With no switch on nothing is added: the view is the old
// engine's, byte for byte.
//
// Where the game's NetFullCombatState has a field for it, the kernel writes it in the game's terms and name:
//   net.last_action  lastExecutedActionId: the id the last action to finish ran under;
//   net.last_hook    lastExecutedHookId: the hook action's id, when that action was a hook action.
// (nextChoiceIds is PlayerChoiceSynchronizer's counter, which the kernel does not keep; the schema has no field for the queue
// itself.) Everything else is the kernel's own and goes under private.kernel, which no comparer reads, and which the
// comparer lists as a surplus (struct_compare: the census's `surplus`), never against the record:
//   queue            every action (kind, type, state, id, owner) in enqueue order, and each queue's front-first ids and flags;
//   hook_actions     the pending hook actions the passes recorded: where each paused (dispatcher, pass, index, listener,
//                    context), its screen request and its resume record;
//   screen           the combat's open screen when it is the kernel's (the hook action it belongs to, its answer key);
//   keyed_answers    the answers stored by key in the open screen's replay record, in flight;
//   answer_sequence  how many screens each (dispatcher, listener, resume site) has reached;
//   turn_start_flow  whether the turn start's flow pause (its hook action, CombatPlayPhaseOnly) is pending.
using System.Reflection;
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;

namespace EmuDump.Engines.Core2;

public static class Core2GameView
{
    /// <summary>Adds the kernel's fields to the old engine's view of the run's current combat (nothing with no switch on).</summary>
    internal static void Add(JsonObject view, RunEngine run)
    {
        var combat = run.State.ActiveCombat;
        var k = combat?.Kernel ?? run.State.Kernel;
        if (k is null || !k.AnyOn)
            return;
        view["private"] ??= new JsonObject();
        view["private"]!["kernel"] = Block(k, combat);
        if (view["net"] is JsonObject net)
            Sequencing(k, net);
    }

    /// <summary>The game's lastExecutedActionId and lastExecutedHookId (net.last_action, net.last_hook), from the queue's own
    /// counters; absent until an action finished.</summary>
    internal static void Sequencing(KernelState k, JsonObject net)
    {
        if (k.Queue is not { } q)
            return;
        if (q.LastExecutedActionId is { } a)
            net["last_action"] = a;
        if (q.LastExecutedHookId is { } h)
            net["last_hook"] = h;
    }

    /// <summary>The kernel part of the view: everything under <c>private.kernel</c>. Plain data from the state alone, so a
    /// copy of the state writes the same block.</summary>
    public static JsonObject Block(KernelState k, CombatState? combat)
    {
        var q = k.Queue!;
        var o = new JsonObject
        {
            ["switches"] = new JsonArray(k.Options.Switches.Select(s => s.ToString()).Order(StringComparer.Ordinal).Select(s => (JsonNode)s).ToArray()),
            ["executor_paused"] = k.ExecutorPaused,
            ["win_checks"] = k.WinChecks,
            ["queue"] = Queue(q),
            ["hook_actions"] = new JsonArray(k.HookActions.Select(h => (JsonNode)HookAction(h)).ToArray()),
            ["stops"] = new JsonArray(k.Stops.Select(s => (JsonNode)Site(s)).ToArray()),
            ["waits"] = new JsonArray(k.Waits.Select(w => (JsonNode)new JsonObject
                { ["dispatcher"] = w.Dispatcher.ToString(), ["mark"] = w.Mark, ["pending"] = w.Pending }).ToArray()),
            ["screen"] = Screen(combat?.PendingSelection, q),
            ["keyed_answers"] = new JsonArray((combat?.PendingSelection?.Replay?.Keyed ?? [])
                .Select(a => (JsonNode)With(Key(a.Key), "answer", a.Answer)).ToArray()),
            ["answer_sequence"] = new JsonArray(k.AnswerSequence
                .OrderBy(kv => kv.Key.Item1).ThenBy(kv => kv.Key.Item2.Kind).ThenBy(kv => kv.Key.Item2.ModelId)
                .ThenBy(kv => kv.Key.Item2.Instance).ThenBy(kv => kv.Key.Item3, StringComparer.Ordinal)
                .Select(kv => (JsonNode)With(Key(new HookAnswerKey(kv.Key.Item1, kv.Key.Item2, kv.Key.Item3, 0)), "count", kv.Value))
                .ToArray()),
        };
        // The flow pause: the hook action the auto-pre-play phase enqueues (CombatPlayPhaseOnly, not a recorded hook action).
        var flow = q.All.FirstOrDefault(a => a is { Kind: GameActionKind.Hook, Type: GameActionType.CombatPlayPhaseOnly, HookAction: 0 }
                                             && a.State is not (GameActionState.Finished or GameActionState.Canceled));
        o["turn_start_flow"] = flow is null
            ? new JsonObject { ["paused"] = false }
            : new JsonObject { ["paused"] = true, ["action"] = flow.Id, ["state"] = flow.State.ToString() };
        return o;
    }

    private static JsonObject Queue(ActionQueueSet q)
    {
        var queues = new JsonArray();
        for (int i = 0; i < q.QueueCount; i++)
        {
            var (play, driven, combat) = q.Cancelling(i);
            queues.Add(new JsonObject
            {
                ["owner"] = i, ["paused"] = q.IsPaused(i), ["cancelling_play_card"] = play, ["cancelling_player_driven"] = driven,
                ["cancelling_combat"] = combat,
                ["front_first"] = new JsonArray(q.QueueOf(i).Select(a => (JsonNode?)a.Id).ToArray()),
            });
        }

        return new JsonObject
        {
            ["next_id"] = q.NextId, ["in_combat"] = q.InCombat, ["queues"] = queues,
            ["actions"] = new JsonArray(q.All.Select(a => (JsonNode)Action(a)).ToArray()),
        };
    }

    private static JsonObject Action(GameActionRec a)
    {
        var o = new JsonObject
        {
            ["id"] = a.Id, ["kind"] = a.Kind.ToString(), ["type"] = a.Type.ToString(), ["state"] = a.State.ToString(), ["owner"] = a.Owner,
        };
        if (a.FirstId is { } first)
            o["first_id"] = first;
        if (a.HookAction != 0)
            o["hook_action"] = a.HookAction;
        if (a.Kind is GameActionKind.PlayCard or GameActionKind.UsePotion or GameActionKind.EndTurn)
        {
            o["action"] = a.Action;
            o["target"] = a.Target;
        }

        if (a.Answer is { } answer)
            o["answer"] = answer;
        return o;
    }

    private static JsonObject Listener(ListenerRef who) =>
        new() { ["kind"] = ((ListenerKind)who.Kind).ToString(), ["model"] = who.ModelId, ["instance"] = who.Instance };

    private static JsonObject Site(PauseSite s) => new()
    {
        ["dispatcher"] = s.Dispatcher.ToString(), ["hook"] = s.Hook.ToString(), ["pass"] = s.Pass, ["index"] = s.Index,
        ["listener"] = Listener(s.Who), ["context"] = s.Context.ToString(),
    };

    private static JsonObject HookAction(RecordedHookAction h)
    {
        var o = Site(h.Site);
        o["id"] = h.Id;
        o["screen"] = Plain(h.Step.Request);
        o["resume"] = h.Step.Resume is { } r ? new JsonObject { ["type"] = r.GetType().Name, ["site"] = r.ResumeSite } : null;
        return o;
    }

    private static JsonObject Key(HookAnswerKey key) => new()
    {
        ["dispatcher"] = key.Dispatcher.ToString(), ["listener"] = Listener(key.Who), ["site"] = key.Site, ["seq"] = key.Seq,
    };

    private static JsonObject With(JsonObject o, string name, int value)
    {
        o[name] = value;
        return o;
    }

    /// <summary>The combat's open screen, when it is one the kernel opened (its hook action, or its key inside a step).</summary>
    private static JsonNode? Screen(PendingCardSelection? s, ActionQueueSet q)
    {
        // Action ids start at 0, so a screen naming action 0 is the kernel's only when that hook action is gathering its choice.
        bool firstAction = s is { KernelAction: 0, HookKey: null }
                           && q.All.Any(a => a is { Kind: GameActionKind.Hook, Id: 0, State: GameActionState.GatheringPlayerChoice, HookAction: not 0 });
        if (s is null || (s.KernelAction == 0 && s.HookKey is null && !firstAction))
            return null;
        var o = new JsonObject
        {
            ["kind"] = s.Kind.ToString(), ["candidates"] = s.Candidates.Count, ["skippable"] = s.Skippable,
        };
        if (s.KernelAction != 0 || firstAction)
            o["hook_action_id"] = s.KernelAction;
        if (s.HookKey is { } key)
            o["key"] = Key(key);
        return o;
    }

    /// <summary>A screen request's plain data (HookPlain: primitives, strings, enums, structs and records of the same):
    /// its type name and public properties, enums by name.</summary>
    private static JsonNode? Plain(object? x)
    {
        switch (x)
        {
            case null:
                return null;
            case string or bool or int or long or uint or short or byte or decimal or double or float:
                return JsonValue.Create(x);
            case Enum e:
                return e.ToString();
            case System.Collections.IEnumerable list:
                return new JsonArray(list.Cast<object?>().Select(Plain).ToArray());
        }

        var o = new JsonObject { ["type"] = x.GetType().Name };
        foreach (var p in x.GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance).Where(p => p.GetIndexParameters().Length == 0
                     && p.Name is not "EqualityContract"))
            o[p.Name] = Plain(p.GetValue(x));
        return o;
    }
}
