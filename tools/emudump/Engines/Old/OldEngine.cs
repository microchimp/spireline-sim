// The old engine (Sts2Emulator's RunEngine) behind IEngine. Only this folder and the consumers listed in
// Engine/not_moved.txt name Sts2Emulator; those consumers have not moved onto IEngine yet and reach the
// run engine through OldEngine.Of(engine).Run.
using System.Text.Json.Nodes;
using EmuDump.Engine;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;

namespace EmuDump.Engines.Old;

public sealed class OldEngine : IEngine
{
    internal OldEngine(RunEngine run) => Run = run;

    public OldEngine() : this(new RunEngine())
    {
    }

    /// <summary>The run engine underneath, for the consumers not yet moved onto <see cref="IEngine"/>.</summary>
    public RunEngine Run { get; }

    /// <summary>The old engine behind <paramref name="e"/>; throws for any other engine.</summary>
    public static OldEngine Of(IEngine e) =>
        e as OldEngine ?? throw new InvalidOperationException($"engine '{e.Name}' is not the old engine");

    public string Name => "old";

    CombatState? Combat => Run.State.ActiveCombat;

    public bool CombatOpen => Combat != null;

    public OpenScreen? Screen => OldChoices.ScreenOf(Combat);

    public void Pick(int candidate) => Step(candidate, -1);

    public void Decline() => Step(Combat!.PendingSelection!.Candidates.Count, -1);

    public void RequireModel(string modelId) => OldIds.CardIdBySnake(modelId);

    public void PlayCard(int handIndex, int targetSlot) => Step(handIndex, targetSlot);

    public void UsePotion(int slot, int targetSlot) => Step(Combat!.Hand.Count + 1 + slot, targetSlot);

    public void EndTurn() => Step(Combat!.Hand.Count, -1);

    public JsonObject DumpStep() => OldDump.Dump(Run);

    public JsonObject PowerKeys() => OldAnchor.PowerKeys(OldReanchor.Tables);

    public JsonObject? GameState() => OldNetState.From(Run);

    public JsonObject RngCounters() => OldSetup.Counters(Run.State);

    public SetupResult ObtainRelic(string relicId, IReadOnlyList<JsonObject> cardChoices) => OldSetup.Relic(Run, relicId, cardChoices.ToList());

    public SetupResult ProcurePotion(string potionId) => OldSetup.Potion(Run, potionId);

    public SetupResult AddCard(string cardId, bool upgraded, string pile, bool onTop) => OldSetup.Card(Run, cardId, upgraded, pile, onTop);

    public SetupResult ApplyPower(string powerId, int targetSlot, int amount) => OldSetup.Power(Run, powerId, targetSlot, amount);

    public SetupResult GainEnergy(int amount) => OldSetup.Energy(Run, amount);

    public SetupResult SetHp(int? maxHp, int? currentHp) => OldSetup.Hp(Run, maxHp, currentHp);

    public SetupResult? RefsetItem(JsonObject item, IReadOnlyList<JsonObject> cardChoices) => OldSetup.Refset(Run, item, cardChoices.ToList());

    public string EnterEncounter(int mapOption, string act, string kind, int index, string gameId) =>
        OldSetup.EnterEncounter(Run, mapOption, act, kind, index, gameId);

    public JsonObject SetupInfo() => OldSetup.Info(PowerKeys());

    public void StartRngLog() => OldRngLog.Start();

    public IReadOnlyList<RngDraw> TakeRngLog() => OldRngLog.Take(Run);

    public string StateHash() => OldStateWalk.RunHash(Run);

    public IReadOnlyList<KeyValuePair<string, string>> StateFieldHashes()
    {
        var all = OldStateWalk.FieldHashes(Run.State).ToList();
        if (Run.State.ActiveCombat is { } c)
        {
            all.AddRange(OldStateWalk.FieldHashes(c).Select(kv => KeyValuePair.Create("ActiveCombat." + kv.Key, kv.Value)));
        }

        return all;
    }

    public JsonObject HookProbe(JsonObject scenarios) => OldHookProbe.Run(scenarios);

    public ReanchorOpening ReanchorOpen() => OldReanchor.Open(this);

    public ReanchorOutcome ReanchorStep(EngineBlock carry, ReanchorMode mode, ReanchorInput input, JsonObject record) =>
        OldReanchor.Step(this, carry as OldCarry ?? throw new ArgumentException("not a block of the old engine", nameof(carry)), mode, input, record);

    void Step(int action, int target) => Run.Step(action, target, out _, out _, out _);

    /// <summary>
    /// The run up to the first combat, as the referee drove it: deck additions, the Neow pick,
    /// card choices, the walk into the first room. Fills <c>seed</c> and <c>neow_options</c> of
    /// <paramref name="report"/> and leaves the fight open.
    /// </summary>
    public void StartFight(FightSpec spec, JsonObject report)
    {
        var e = Run;
        var game = spec.Record;
        OldDump.Extended = Setup.IsExtended(game);
        var setupLog = new JsonArray();
        if (OldDump.Extended) report["setup_log"] = setupLog;
        using (Prof.Scope("reset")) e.Reset(spec.Seed, spec.Ascension);
        var s = e.State;
        report["seed"] = s.StringSeed;
        report["neow_options"] = new JsonArray(s.NeowOptions.Select(i => (JsonNode)i).ToArray());
        // The referee's deck additions, appended the same way (before Neow).
        foreach (var name in spec.DeckAdd)
        {
            RunNonCombatEffects.AddCardToDeck(s, new CardInstance(OldIds.CardIdBySnake(name), Upgraded: false));
        }

        Step(spec.NeowPick, -1);
        int guard = 0;
        while (s.Phase != RunPhase.Map && s.Phase != RunPhase.Combat && guard++ < 10)
        {
            Step(0, -1);  // first offered card, as the referee's selector
            if (s.Phase == RunPhase.BundleSelect)
            {
                Step(RunConstants.BundleConfirmAction, -1);  // bundle 0 picked; confirm
            }
        }

        Setup.MapItems(this, game, setupLog);  // the setup items before the fight, on the map after Neow
        var want = spec.Room;
        int idx = Array.FindIndex(s.MapOptionCoords, c => c == want);
        if (idx < 0) throw new InvalidOperationException($"room {want} not offered; phase {s.Phase}");
        var fight = new JsonObject();
        if (Setup.EnterFight(this, game, idx, fight))
        {
            report["fight"] = fight;
        }
        else
        {
            Step(idx, -1);
        }

        if (game["room_choices_from"] is JsonNode from)
        {
            // Choices the fight's start opened (Gambling Chip, Toolbox), answered as the game did.
            var start = new JsonArray();
            foreach (var choice in (game["choices"]?.AsArray() ?? new JsonArray()).Skip((int)from)
                         .Where(c => (int?)c!["step"] == 0 && (string?)c["kind"] == "cards"))
            {
                start.Add(OldChoices.Apply(() => s.ActiveCombat, i => Step(i, -1), choice!.AsObject()));
            }

            if (start.Count > 0) report["start_choices"] = start;
        }

        Setup.CombatItems(this, game, setupLog);  // the setup items after the fight item, before step 0
    }

    /// <summary>The run up to the first combat, on a fresh engine; for the callers that hold the run engine itself.</summary>
    internal static RunEngine EnterRoom(JsonObject game, JsonObject outp)
    {
        var engine = new OldEngine();
        engine.StartFight(new FightSpec(game), outp);
        return engine.Run;
    }

    /// <summary>Card pools for the batch maker: <c>EmuDump pools</c>.</summary>
    public static JsonObject Pools() => new()
    {
        ["ironclad"] = new JsonArray(Sts2Emulator.GeneratedData.CardPools.Ironclad.ToArray().Select(i => (JsonNode)i).ToArray()),
        ["colorless"] = new JsonArray(Sts2Emulator.GeneratedData.CardPools.Colorless.ToArray().Select(i => (JsonNode)i).ToArray()),
    };
}
