// EmuDump kernel-handlers (Engines/Core2/KernelHandlers.cs): the list tools/kernel/port_list.py reads is the handler
// table's own registrations. A registered handler is listed, a handler marked [StubHandler] is listed as a stub (the port
// list does not count it), a family that is off in the default switches is listed all the same (all switches are on), and a
// model that registers nothing is absent however much its source says.
using System.Text.Json.Nodes;
using EmuDump.Engines.Core2;
using Sts2Emulator.Core2;
using Xunit;

public class KernelHandlersTests
{
    sealed class FakeIds : IModelIds
    {
        readonly Dictionary<(ListenerKind, string), int> _ids = new();

        public int Resolve(ListenerKind kind, string gameId) =>
            _ids.TryGetValue((kind, gameId), out int id) ? id : _ids[(kind, gameId)] = _ids.Count + 1;
    }

    [StubHandler]
    static void StubAfterDeath() => throw new NotImplementedException();

    static void RealAfterDeath()
    {
    }

    static (string Model, string Hook, bool Stub)[] Pairs(JsonObject doc) =>
        doc["handlers"]!.AsArray()
            .Select(h => ((string)h!["model"]!, (string)h["hook"]!, (bool)h["stub"]!)).ToArray();

    [Fact]
    public void A_registered_handler_is_listed_and_a_stub_is_listed_as_one()
    {
        Action<HandlerTable> real = t => t.Model(ListenerKind.Power, "RavenousPower").On(HookId.AfterDeath, new Action(RealAfterDeath));
        Action<HandlerTable> stub = t => t.Model(ListenerKind.Power, "HopperPower").On(HookId.AfterDeath, new Action(StubAfterDeath));
        Action<HandlerTable> lambdaStub = t => t.Model(ListenerKind.Monster, "Hopper")
            .On(HookId.AfterDeath, new Action([StubHandler] () => throw new NotImplementedException()));
        var got = Pairs(KernelHandlersMode.Dump(new FakeIds(), [real, stub, lambdaStub]));
        var expected = new (string, string, bool)[]
        {
            ("HopperPower", "AfterDeath", true), ("Hopper", "AfterDeath", true), ("RavenousPower", "AfterDeath", false),
        };
        Assert.Equal(expected, got.OrderBy(g => g.Model == "HopperPower" ? 0 : g.Model == "Hopper" ? 1 : 2).ToArray());
    }

    [Fact]
    public void A_model_that_registers_nothing_is_absent_and_every_family_is_on()
    {
        Action<HandlerTable> silent = t => t.Model(ListenerKind.Relic, "BurningBlood");   // names a model, registers no hook
        Action<HandlerTable> damage = t => t.Model(ListenerKind.Power, "VulnerablePower")
            .On(HookId.ModifyDamageMultiplicative, new Action(RealAfterDeath));
        Action<HandlerTable> kill = t => t.Model(ListenerKind.Power, "RavenousPower").On(HookId.AfterDeath, new Action(RealAfterDeath));
        var got = Pairs(KernelHandlersMode.Dump(new FakeIds(), [silent, damage, kill]));
        Assert.Equal(2, got.Length);
        Assert.Contains(("VulnerablePower", "ModifyDamageMultiplicative", false), got);
        Assert.Contains(("RavenousPower", "AfterDeath", false), got);
    }

    [Fact]
    public void The_mode_writes_the_registry_s_list_to_a_file()
    {
        string path = Path.Combine(Path.GetTempPath(), $"kernel-handlers-{Guid.NewGuid():N}.json");
        try
        {
            Assert.Equal(0, KernelHandlersMode.Run([path]));
            Assert.Equal(Sts2Emulator.Core2.Models.Registry.All.Count > 0, JsonNode.Parse(File.ReadAllText(path))!["handlers"]!.AsArray().Count > 0);
        }
        finally
        {
            File.Delete(path);
        }

        Assert.Equal(2, KernelHandlersMode.Run(["a", "b"]));
    }
}
