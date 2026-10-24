// Batch R7a (rebuild/f08-r7a), the R7 dealer check of docs/rebuild/MIGRATION_F08.md: the game's ConstrictPower and
// DisintegrationPower call CreatureCmd.Damage with their owner, the player, as dealer; the old entry
// (CardEffects.DealDamageToPlayer) takes a creature reference that can name it. Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.ModelTestKit;

namespace Sts2Emulator.Tests.Core2.Models;

public class R7aDealerTests
{
    [Theory]
    [InlineData(BuffId.Constrict, "ConstrictPower")]
    [InlineData(BuffId.Disintegration, "DisintegrationPower")]
    public void ConstrictAndDisintegrationDealTheirDamageWithThePlayerAsDealer(BuffId buff, string model)
    {
        // game: ConstrictPower and DisintegrationPower call CreatureCmd.Damage with their owner as target and dealer.
        var s = Combat(40);
        s.PlayerBuffs.Add(new BuffState(buff, 3));
        var seen = new List<(CreatureRef Target, decimal Amount, CreatureRef? Dealer)>();
        void Recorder(HandlerTable t) =>
            t.Model(ListenerKind.Power, model).On(HookId.BeforeDamageReceived,
                (HookEvent<BeforeDamageReceivedArgs>)((HookCtx c, ListenerRef self, in BeforeDamageReceivedArgs a,
                    ChoiceCtx? ctx) =>
                {
                    if (!a.Props.IsPoweredAttack())
                        seen.Add((a.Target, a.Amount, a.Dealer));
                    return HookStep.Done;
                }));
        _ = Kernel(s, Recorder);
        CombatEngine.Step(s, s.Hand.Count, new Random(1));
        Assert.Contains((new CreatureRef(0), 3m, (CreatureRef?)new CreatureRef(0)), seen);
    }
}
