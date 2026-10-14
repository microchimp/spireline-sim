using System.Diagnostics;
using System.Reflection;
using System.Runtime.CompilerServices;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Relics;
using Spireline.Referee;
using Xunit;

namespace Sts2Emulator.Tests
{
    /// <summary>
    /// Which relic flashes the referee records (referee/driver/FlashSource.cs): those the rules raise, on both hosts;
    /// not those display code raises (a Nodes screen or VFX, the card preview's tween), which exist only where the
    /// node does and land on whatever step the driver has reached. The stand-in types below carry the game's
    /// namespaces and names, so the frames are shaped as in the game; no game code is involved.
    /// </summary>
    public class RefereeFlashSourceTests
    {
        private static MethodBase? RaiserOf(Action raise)
        {
            FlashProbe.Last = null;
            raise();
            return FlashProbe.Last;
        }

        [Fact]
        public void AFlashARelicsOwnHookRaisesIsRecorded()
        {
            var relic = new AmethystAubergine();
            var raiser = RaiserOf(relic.AfterModifyingRewards);
            Assert.Equal(nameof(AmethystAubergine.AfterModifyingRewards), raiser?.Name);
            Assert.False(FlashSource.IsDisplay(raiser));
        }

        [Fact]
        public void ARuleHookReachedFromANodeIsStillTheRaiser()
        {
            // The full host's Amethyst Aubergine flash: NCombatUi.OnCombatWon -> ... -> the relic's AfterModifyingRewards.
            var relic = new AmethystAubergine();
            var raiser = RaiserOf(() => MegaCrit.Sts2.Core.Nodes.Combat.NCombatUi.OnCombatWon(relic));
            Assert.Equal(typeof(AmethystAubergine), raiser?.DeclaringType);
            Assert.False(FlashSource.IsDisplay(raiser));
        }

        [Fact]
        public void AFlashAScreenRaisesIsDisplay()
        {
            var relic = new AmethystAubergine();
            var raiser = RaiserOf(() => MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardRewardSelectionScreen.ShowModifiedOption(relic));
            Assert.True(FlashSource.IsDisplay(raiser));
            Assert.True(FlashSource.IsDisplay(RaiserOf(() => MegaCrit.sts2.Core.Nodes.TopBar.NTopBarRelic.Ping(relic))));
        }

        [Fact]
        public void TheCardPreviewsFlashIsDisplayEvenWhenInlinedIntoItsCallback()
        {
            var relic = new AmethystAubergine();
            var viaHelper = RaiserOf(() => CardCmd.PreviewInternal(relic, inlined: false)());
            Assert.Equal("CardCmd.FlashRelics", FlashSource.Name(viaHelper));
            Assert.True(FlashSource.IsDisplay(viaHelper));
            var inlined = RaiserOf(() => CardCmd.PreviewInternal(relic, inlined: true)());
            Assert.Contains("<PreviewInternal>", inlined?.Name ?? inlined?.DeclaringType?.Name ?? "");
            Assert.True(FlashSource.IsDisplay(inlined));
        }

        [Fact]
        public void OtherCardCmdFlashesAreRuleFlashes()
        {
            var relic = new AmethystAubergine();
            var raiser = RaiserOf(() => CardCmd.Upgrade(relic));
            Assert.Equal("CardCmd.Upgrade", FlashSource.Name(raiser));
            Assert.False(FlashSource.IsDisplay(raiser));
        }

        [Fact]
        public async Task AnAsyncRaiserIsJudgedByTheTypeItsStateMachineBelongsTo()
        {
            var relic = new AmethystAubergine();
            FlashProbe.Last = null;
            await relic.AfterSomethingAsync();
            Assert.Equal(typeof(AmethystAubergine), Root(FlashProbe.Last!.DeclaringType!));
            Assert.False(FlashSource.IsDisplay(FlashProbe.Last));
            FlashProbe.Last = null;
            await MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardRewardSelectionScreen.ShowAsync(relic);
            Assert.True(FlashSource.IsDisplay(FlashProbe.Last));
        }

        [Fact]
        public void NoRaiserIsNotDisplay()
        {
            Assert.Null(FlashSource.Raiser(new MethodBase?[] { null, typeof(FlashProbe).GetMethod(nameof(FlashProbe.OnFlashed)) }, typeof(RelicModel)));
            Assert.False(FlashSource.IsDisplay(null));
            Assert.Equal("?", FlashSource.Name(null));
        }

        private static Type Root(Type t)
        {
            while (t.DeclaringType != null) t = t.DeclaringType;
            return t;
        }
    }
}

namespace Spireline.Referee
{
    /// <summary>The recorder's handler as Chooser.cs has it: the raiser, from the frames below it.</summary>
    internal static class FlashProbe
    {
        internal static MethodBase? Last;

        [MethodImpl(MethodImplOptions.NoInlining)]
        internal static void OnFlashed(RelicModel relic)
        {
            Last = FlashSource.Raiser(new StackTrace(0, false).GetFrames().Select(f => f.GetMethod()), typeof(RelicModel));
        }
    }
}

namespace MegaCrit.Sts2.Core.Models
{
    /// <summary>Stand-in: Flash() calls Flash(targets), which raises Flashed, as RelicModel's do.</summary>
    public class RelicModel
    {
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void Flash() => Flash(new object[] { this });

        [MethodImpl(MethodImplOptions.NoInlining)]
        public void Flash(IEnumerable<object> targets) => Spireline.Referee.FlashProbe.OnFlashed(this);
    }
}

namespace MegaCrit.Sts2.Core.Models.Relics
{
    public class AmethystAubergine : MegaCrit.Sts2.Core.Models.RelicModel
    {
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void AfterModifyingRewards() => Flash();

        [MethodImpl(MethodImplOptions.NoInlining)]
        public async Task AfterSomethingAsync()
        {
            await Task.Yield();
            Flash();
        }
    }
}

namespace MegaCrit.Sts2.Core.Commands
{
    public static class CardCmd
    {
        /// <summary>The preview's tween callback: FlashRelics, or (inlined) the flash itself.</summary>
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static Action PreviewInternal(MegaCrit.Sts2.Core.Models.RelicModel relic, bool inlined) =>
            inlined ? () => relic.Flash() : () => FlashRelics(relic);

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void FlashRelics(MegaCrit.Sts2.Core.Models.RelicModel relic) => relic.Flash();

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void Upgrade(MegaCrit.Sts2.Core.Models.RelicModel relic) => relic.Flash();
    }
}

namespace MegaCrit.Sts2.Core.Nodes.Combat
{
    public static class NCombatUi
    {
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void OnCombatWon(MegaCrit.Sts2.Core.Models.Relics.AmethystAubergine relic) => relic.AfterModifyingRewards();
    }
}

namespace MegaCrit.Sts2.Core.Nodes.Screens.CardSelection
{
    public static class NCardRewardSelectionScreen
    {
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void ShowModifiedOption(MegaCrit.Sts2.Core.Models.RelicModel relic) => relic.Flash();

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static async Task ShowAsync(MegaCrit.Sts2.Core.Models.RelicModel relic)
        {
            await Task.Yield();
            relic.Flash();
        }
    }
}

namespace MegaCrit.sts2.Core.Nodes.TopBar
{
    public static class NTopBarRelic
    {
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void Ping(MegaCrit.Sts2.Core.Models.RelicModel relic) => relic.Flash();
    }
}
