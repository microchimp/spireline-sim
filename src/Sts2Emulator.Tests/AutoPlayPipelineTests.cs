using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The one play pipeline (referee/AUTOPLAY.md). The game has two entry points --
/// `PlayCardAction` for a chosen card, `CardCmd.AutoPlay` for every card the game plays for
/// the player -- and ONE body, `CardModel.OnPlayWrapper`. Every auto-play runs where its
/// caller awaits `CardCmd.AutoPlay`: inside the draw, inside the causing card, inside the
/// phase hook. These play real cards rather than setting counters.
/// </summary>
public class AutoPlayPipelineTests
{
    private const int DistilledChaos = 15;
    private const int IAmInvincible = 257;
    private const int Uproar = 530;

    private static CardInstance Card(int id, bool upgraded = false) => new(id, upgraded);

    private static int Total(Fight fight) => fight.State.Enemies.Sum(enemy => enemy.Hp);

    // ── The shared body: an auto-play gets everything a chosen play gets ─────────────────

    /// <summary>`RagePower.AfterCardPlayed` fires for the Strike Havoc plays.</summary>
    [Fact]
    public void AnAutoPlayedAttackPaysRage()
    {
        var fight = Fight
            .Hand(Card(IC.Havoc))
            .Enemy(hp: 100)
            .Draw(Card(IC.StrikeIronclad))
            .PlayerBuff(BuffId.Rage, 3)
            .Energy(3);

        fight.Play(0);

        Assert.Equal(94, fight.State.Enemies[0].Hp);
        Assert.Equal(3, fight.State.PlayerBlock);
    }

    /// <summary>
    /// `DuplicationPower.ModifyCardPlayCount` doubles the first card played whatever plays
    /// it: the Strike Distilled Chaos plays hits twice and spends the power.
    /// </summary>
    [Fact]
    public void DuplicationRepeatsAnAutoPlay()
    {
        var fight = Fight
            .Hand(Card(IC.DefendIronclad))
            .Enemy(hp: 100)
            .Draw(Card(IC.StrikeIronclad), Card(IC.DefendIronclad), Card(IC.DefendIronclad))
            .PlayerBuff(BuffId.Duplication, 1)
            .Energy(3);
        fight.State.PotionSlots[0] = DistilledChaos;

        fight.Potion(0);

        Assert.Equal(88, fight.State.Enemies[0].Hp);
        Assert.Equal(10, fight.State.PlayerBlock);
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.Duplication));
    }

    /// <summary>`CorruptionPower.ModifyCardPlayResultPileTypeAndPosition` exhausts an auto-played Skill.</summary>
    [Fact]
    public void CorruptionExhaustsAnAutoPlayedSkill()
    {
        var fight = Fight
            .Hand(Card(IC.StrikeIronclad))
            .Enemy(hp: 100)
            .Draw(Card(IC.DefendIronclad), Card(IC.DefendIronclad), Card(IC.DefendIronclad))
            .PlayerBuff(BuffId.Corruption, 1)
            .Energy(3);
        fight.State.PotionSlots[0] = DistilledChaos;

        fight.Potion(0);

        Assert.Equal(3, fight.State.ExhaustPile.Count(c => c.DefId == IC.DefendIronclad));
        Assert.DoesNotContain(fight.State.DiscardPile, c => c.DefId == IC.DefendIronclad);
    }

    /// <summary>
    /// `VoidFormPower.AfterCardPlayed` counts only `!IsAutoPlay`: Havoc counts, the card it
    /// plays does not.
    /// </summary>
    [Fact]
    public void VoidFormCountsTheChosenCardOnly()
    {
        var fight = Fight
            .Hand(Card(IC.Havoc))
            .Enemy(hp: 100)
            .Draw(Card(IC.StrikeIronclad))
            .PlayerBuff(BuffId.VoidForm, 5)
            .Energy(3);

        fight.Play(0);

        Assert.Equal(1, fight.State.VoidFormCardsPlayedThisTurn);
        Assert.Equal(2, fight.State.CardPlaysStartedThisTurn);
    }

    /// <summary>
    /// A play nested inside another starts clean and hands the outer card its own per-play
    /// state back: Momentum's growth on a Pommel Strike survives the Strike Hellraiser plays
    /// out of the Pommel Strike's draw.
    /// </summary>
    [Fact]
    public void ANestedPlayLeavesTheOuterCardsGrowthOnIt()
    {
        var pommel = Card(IC.PommelStrike) with
        {
            Enchantment = Enchantment.Momentum,
            EnchantAmount = 2,
        };
        var fight = Fight
            .Hand(pommel)
            .Enemy(hp: 100)
            .Draw(Card(IC.StrikeIronclad))
            .PlayerBuff(BuffId.Hellraiser, 1)
            .Energy(3);

        fight.Play(0);

        Assert.Equal(100 - 9 - 6, fight.State.Enemies[0].Hp);
        Assert.Equal(2, fight.State.DiscardPile.Single(c => c.DefId == IC.PommelStrike).BonusDamage);
    }

    // ── Timing: where the game awaits CardCmd.AutoPlay ──────────────────────────────────

    /// <summary>
    /// `HellraiserPower.AfterCardDrawnEarly` plays the Strike inside the draw, and
    /// `CardPileCmd.Draw` stops once the combat is over: the Strike kills, and the second
    /// card an upgraded Pommel Strike would draw stays in the draw pile.
    /// </summary>
    [Fact]
    public void HellraiserPlaysInsideTheDrawAndAKillStopsIt()
    {
        var fight = Fight
            .Hand(Card(IC.PommelStrike, upgraded: true))
            .Enemy(hp: 16)
            .Draw(Card(IC.StrikeIronclad), Card(IC.DefendIronclad), Card(IC.DefendIronclad))
            .PlayerBuff(BuffId.Hellraiser, 1)
            .Energy(3);

        var result = fight.Play(0);

        Assert.True(result.Terminal);
        Assert.Empty(fight.State.Hand);
        Assert.Equal(2, fight.State.DrawPile.Count);
    }

    /// <summary>
    /// Knife Trap auto-plays its Shivs inside its own play, so under Velvet Choker a trap
    /// played sixth still throws one: `VelvetChoker.AfterCardPlayed` has not counted the trap
    /// yet when the first Shiv asks, and has counted that Shiv when the second asks.
    /// </summary>
    [Fact]
    public void KnifeTrapPlayedSixthThrowsOneShivUnderVelvetChoker()
    {
        var fight = Fight.WithRelics(RelicEffects.VelvetChoker).Enemy(hp: 100).Energy(20);
        fight.State.Hand =
        [
            Card(IC.DefendIronclad),
            Card(IC.DefendIronclad),
            Card(IC.DefendIronclad),
            Card(IC.DefendIronclad),
            Card(IC.DefendIronclad),
            Card(SI.KnifeTrap),
        ];
        fight.State.ExhaustPile = [Card(SI.Shiv), Card(SI.Shiv)];

        for (int i = 0; i < 6; i++)
        {
            fight.Play(0);
        }

        Assert.Equal(96, fight.State.Enemies[0].Hp);
        Assert.Equal(2, fight.State.ExhaustPile.Count(c => c.DefId == SI.Shiv));
    }

    /// <summary>
    /// Catastrophe picks, plays, then picks again, so the second `StableShuffle` is over the
    /// pile the first play left: a Pommel Strike's draw takes a card out of it. Four Pommel
    /// Strikes: 3 Shuffle draws for the first pick, 1 for the second (two left), not 2.
    /// </summary>
    [Fact]
    public void CatastrophePicksAgainAfterEachPlay()
    {
        var fight = Fight
            .Hand(Card(CL.Catastrophe))
            .Enemy(hp: 200)
            .Draw(
                Card(IC.PommelStrike),
                Card(IC.PommelStrike),
                Card(IC.PommelStrike),
                Card(IC.PommelStrike)
            )
            .Energy(3);
        var shuffle = new CountingRandom(7);
        fight.State.ShuffleRng = shuffle;

        fight.Play(0);

        Assert.Equal(4, shuffle.CallCount);
        Assert.Equal(200 - 2 * 9, fight.State.Enemies[0].Hp);
        Assert.Equal(2, fight.State.Hand.Count);
    }

    /// <summary>
    /// The case the game was run on (fix/autoplay-target-2 review): Catastrophe over five
    /// Pommel Strikes makes 6 Shuffle draws in the game -- 4 for the first pick, the Pommel
    /// Strike's draw, then 2 over the three left. Picking both first made 7.
    /// </summary>
    [Fact]
    public void CatastropheOverFivePommelStrikesDrawsSixAsTheGameDoes()
    {
        var fight = Fight
            .Hand(Card(CL.Catastrophe))
            .Enemy(hp: 200)
            .Draw(
                Card(IC.PommelStrike),
                Card(IC.PommelStrike),
                Card(IC.PommelStrike),
                Card(IC.PommelStrike),
                Card(IC.PommelStrike)
            )
            .Energy(3);
        var shuffle = new CountingRandom(7);
        fight.State.ShuffleRng = shuffle;

        fight.Play(0);

        Assert.Equal(6, shuffle.CallCount);
    }

    /// <summary>
    /// The combat ending stops the plays, not the picks: Catastrophe's second
    /// `StableShuffle` still rolls after the first Strike kills, and the card it picked stays
    /// in the draw pile because `CardCmd.AutoPlay` returns at once.
    /// </summary>
    [Fact]
    public void AfterAKillCatastrophesNextPickRollsAndStays()
    {
        var fight = Fight
            .Hand(Card(CL.Catastrophe))
            .Enemy(hp: 6)
            .Draw(Card(IC.StrikeIronclad), Card(IC.StrikeIronclad), Card(IC.StrikeIronclad))
            .Energy(3);
        var shuffle = new CountingRandom(3);
        fight.State.ShuffleRng = shuffle;

        var result = fight.Play(0);

        Assert.True(result.Terminal);
        Assert.Equal(3, shuffle.CallCount);
        Assert.Equal(2, fight.State.DrawPile.Count);
    }

    /// <summary>
    /// Beat Down rolls `Rng.CombatTargets` itself for an AnyEnemy Attack and hands the
    /// target to `CardCmd.AutoPlay`, which then rolls nothing; an all-enemies Attack takes
    /// no roll at all.
    /// </summary>
    [Fact]
    public void BeatDownRollsOnceForEachAimedAttack()
    {
        var fight = Fight
            .Hand(Card(CL.BeatDown))
            .Enemy(hp: 100)
            .Enemy(hp: 100)
            .Discard(Card(IC.StrikeIronclad), Card(IC.Thunderclap))
            .Energy(3);
        var targets = new CountingRandom(5);
        fight.State.TargetRng = targets;

        fight.Play(0);

        Assert.Equal(1, targets.CallCount);
        Assert.Equal(200 - 6 - 2 * 4, Total(fight));
    }

    /// <summary>
    /// `HistoryCourse` auto-plays a DUPE (`CreateDupe`), whose result pile is none: the dupe
    /// hits and leaves combat, and the deck keeps the Strikes it had.
    /// </summary>
    [Fact]
    public void HistoryCoursesDupeLeavesCombat()
    {
        var fight = Fight.WithRelics(RelicEffects.HistoryCourse).Enemy(hp: 200).Energy(3);
        fight.State.Hand = [Card(IC.StrikeIronclad)];
        int strikes = fight.State.AllCards().Count(c => c.DefId == IC.StrikeIronclad);

        fight.Play(0);
        fight.EndTurn();

        Assert.Equal(200 - 6 - 6, fight.State.Enemies[0].Hp);
        Assert.Equal(strikes, fight.State.AllCards().Count(c => c.DefId == IC.StrikeIronclad));
    }

    /// <summary>
    /// History Course replays the last Attack or Skill whose play FINISHED last turn, auto-
    /// plays included (only dupes are left out): Stampede's Strike, played as the turn ends,
    /// is the one replayed, not the Defend chosen before it.
    /// </summary>
    [Fact]
    public void HistoryCourseReplaysAnAutoPlayedCard()
    {
        var fight = Fight.WithRelics(RelicEffects.HistoryCourse).Enemy(hp: 200).Energy(3);
        fight.State.Hand = [Card(IC.DefendIronclad), Card(IC.StrikeIronclad)];
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Stampede, 1);

        fight.Play(0);
        fight.EndTurn();

        Assert.Equal(200 - 6 - 6, fight.State.Enemies[0].Hp);
    }

    /// <summary>
    /// `IAmInvincible.AfterAutoPostPlayPhaseEntered` is the AutoPostPlay pass, before every
    /// turn-end hook: its block is up when Orichalcum (`BeforeSideTurnEndVeryEarly`) looks, so
    /// Orichalcum pays nothing.
    /// </summary>
    [Fact]
    public void IAmInvincibleBlocksBeforeOrichalcumLooks()
    {
        var fight = Fight.WithRelics(RelicEffects.Orichalcum).Enemy(hp: 200).Energy(3);
        fight.State.Hand = [];
        fight.State.DrawPile = [Card(IAmInvincible), Card(IC.DefendIronclad)];
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Barricade, 1);

        fight.EndTurn();

        Assert.Equal(10, fight.State.PlayerBlock);
    }

    /// <summary>
    /// Stampede's pick goes through the one body too: Duplication doubles it, which the old
    /// hand-only core never asked.
    /// </summary>
    [Fact]
    public void StampedesAttackIsDoubledByDuplication()
    {
        var fight = Fight
            .Hand(Card(IC.StrikeIronclad))
            .Enemy(hp: 100)
            .PlayerBuff(BuffId.Stampede, 1)
            .PlayerBuff(BuffId.Duplication, 1)
            .Energy(3);

        fight.EndTurn();

        Assert.Equal(88, fight.State.Enemies[0].Hp);
    }

    // ── Inside the discard: `CardCmd.DiscardAndDraw` ─────────────────────────────────────

    /// <summary>
    /// `CardCmd.DiscardAndDraw` discards, draws, and only then auto-plays the Sly cards:
    /// Calculated Gamble's one draw takes the Defend before Reflex draws the Strike, and the
    /// pile is then empty with nothing to shuffle (Reflex and the Gamble are still in play).
    /// Playing Reflex first would draw both and shuffle Reflex back in for the Gamble.
    /// </summary>
    [Fact]
    public void ASlyCardPlaysAfterTheDrawOfDiscardAndDraw()
    {
        var fight = Fight
            .Hand(Card(SI.CalculatedGamble), Card(SI.Reflex))
            .Enemy(hp: 100)
            .Draw(Card(IC.DefendIronclad), Card(IC.StrikeIronclad))
            .Discard()
            .Energy(3);

        fight.Play(0);

        Assert.Equal([IC.DefendIronclad, IC.StrikeIronclad], Fight.Ids(fight.State.Hand));
        Assert.Contains(fight.State.DiscardPile, c => c.DefId == SI.Reflex);
        Assert.Empty(fight.State.DrawPile);
    }

    /// <summary>
    /// The Sly card plays inside the card that discarded it: Storm of Steel played sixth under
    /// Velvet Choker still plays the Flick-Flack it discards, because the Choker counts a play
    /// when it finishes and Storm of Steel has not.
    /// </summary>
    [Fact]
    public void ASlyCardPlaysInsideTheCardThatDiscardedIt()
    {
        var fight = Fight.WithRelics(RelicEffects.VelvetChoker).Enemy(hp: 100).Energy(20);
        fight.State.Hand =
        [
            Card(IC.DefendIronclad),
            Card(IC.DefendIronclad),
            Card(IC.DefendIronclad),
            Card(IC.DefendIronclad),
            Card(IC.DefendIronclad),
            Card(SI.StormOfSteel),
            Card(SI.FlickFlack),
        ];

        for (int i = 0; i < 6; i++)
        {
            fight.Play(0);
        }

        Assert.Equal(94, fight.State.Enemies[0].Hp);
        Assert.Contains(fight.State.DiscardPile, c => c.DefId == SI.FlickFlack);
    }

    // ── Play count: settled as the play is set up (`GeneratePlayCount`) ─────────────────

    /// <summary>
    /// `DuplicationPower` is read and spent as Havoc is set up, before its `OnPlay`: Havoc
    /// plays twice and each Strike it plays once, so both leave the draw pile.
    /// </summary>
    [Fact]
    public void DuplicationDoublesHavocNotTheCardItPlays()
    {
        var fight = Fight
            .Hand(Card(IC.Havoc))
            .Enemy(hp: 100)
            .Draw(Card(IC.StrikeIronclad), Card(IC.StrikeIronclad))
            .PlayerBuff(BuffId.Duplication, 1)
            .Energy(3);

        fight.Play(0);

        Assert.Empty(fight.State.DrawPile);
        Assert.Equal(88, fight.State.Enemies[0].Hp);
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.Duplication));
    }

    /// <summary>
    /// `EchoFormPower` counts the turn's `CardPlaysStarted` before the play being set up, and
    /// the Strike Havoc plays starts after Havoc: Havoc, the turn's first card, plays twice,
    /// and each Strike once.
    /// </summary>
    [Fact]
    public void EchoFormDoublesHavocNotTheCardItPlays()
    {
        var fight = Fight
            .Hand(Card(IC.Havoc))
            .Enemy(hp: 100)
            .Draw(Card(IC.StrikeIronclad), Card(IC.StrikeIronclad))
            .PlayerBuff(BuffId.EchoForm, 1)
            .Energy(3);

        fight.Play(0);

        Assert.Equal(88, fight.State.Enemies[0].Hp);
        Assert.Empty(fight.State.DrawPile);
    }

    /// <summary>
    /// `BurstPower` is spent as Calculated Gamble is set up, so the Sly Defend its discard
    /// plays finds none: the Gamble is doubled and the Defend plays once.
    /// </summary>
    [Fact]
    public void BurstDoublesCalculatedGambleNotTheSlyCard()
    {
        var fight = Fight
            .Hand(Card(SI.CalculatedGamble), Card(SI.DefendSilent) with { SlyForCombat = true })
            .Enemy(hp: 100)
            .Draw(Card(SI.StrikeSilent), Card(SI.StrikeSilent), Card(SI.StrikeSilent))
            .PlayerBuff(BuffId.Burst, 1)
            .Energy(3);

        fight.Play(0);

        Assert.Equal(5, fight.State.PlayerBlock);
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.Burst));
    }

    /// <summary>
    /// `OneTwoPunchPower` is spent as Pommel Strike is set up: Pommel Strike plays twice, and
    /// each Strike Hellraiser plays out of its draws plays once.
    /// </summary>
    [Fact]
    public void OneTwoPunchDoublesPommelStrikeNotTheStrikeHellraiserPlays()
    {
        var fight = Fight
            .Hand(Card(IC.PommelStrike))
            .Enemy(hp: 100)
            .Draw(Card(IC.StrikeIronclad), Card(IC.StrikeIronclad), Card(IC.DefendIronclad))
            .PlayerBuff(BuffId.Hellraiser, 1)
            .PlayerBuff(BuffId.OneTwoPunch, 1)
            .Energy(3);

        fight.Play(0);

        Assert.Equal(100 - 9 - 9 - 6 - 6, fight.State.Enemies[0].Hp);
        Assert.Equal([IC.DefendIronclad], Fight.Ids(fight.State.DrawPile));
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.OneTwoPunch));
    }

    // ── Inside the card: the auto-play is part of the causing card's play ───────────────

    /// <summary>
    /// Uproar's auto-played Attack is inside Uproar's play too: played sixth under Velvet
    /// Choker, the Strike it takes off the draw pile still hits.
    /// </summary>
    [Fact]
    public void UproarsAttackPlaysInsideUproarsPlay()
    {
        var fight = Fight
            .WithRelics(RelicEffects.VelvetChoker)
            .Enemy(hp: 100)
            .Draw(Card(IC.StrikeIronclad))
            .Energy(20);
        fight.State.Hand =
        [
            Card(IC.DefendIronclad),
            Card(IC.DefendIronclad),
            Card(IC.DefendIronclad),
            Card(IC.DefendIronclad),
            Card(IC.DefendIronclad),
            Card(Uproar),
        ];

        for (int i = 0; i < 6; i++)
        {
            fight.Play(0);
        }

        Assert.Equal(100 - 2 * 6 - 6, fight.State.Enemies[0].Hp);
    }

    // ── Inside the draw: the AfterCardDrawn hooks come after the Early one ───────────────

    /// <summary>
    /// Chains of Binding binds a drawn card in `AfterCardDrawn`, after Hellraiser's
    /// `AfterCardDrawnEarly` has played it: both Strikes an upgraded Pommel Strike draws hit,
    /// and are Bound where they come to rest. Binding them first refused the second (a Bound
    /// card has already been played this turn).
    /// </summary>
    [Fact]
    public void HellraiserPlaysAStrikeBeforeChainsOfBindingBindsIt()
    {
        var fight = Fight
            .Hand(Card(IC.PommelStrike, upgraded: true))
            .Enemy(hp: 100)
            .Draw(Card(IC.StrikeIronclad), Card(IC.StrikeIronclad), Card(IC.DefendIronclad))
            .PlayerBuff(BuffId.Hellraiser, 1)
            .PlayerBuff(BuffId.ChainsOfBinding, 2)
            .Energy(3);

        fight.Play(0);

        Assert.Equal(100 - 10 - 6 - 6, fight.State.Enemies[0].Hp);
        Assert.Equal(2, fight.State.DiscardPile.Count(c => c.DefId == IC.StrikeIronclad && c.Bound));
    }

    // ── Phase hooks: Stampede rolls for every stack ──────────────────────────────────────

    /// <summary>
    /// `StampedePower` rolls `Rng.Shuffle.NextItem` for every stack, the fight over or not
    /// (the stream is the run's): the first Strike kills, the second stack still rolls, and
    /// `CardCmd.AutoPlay` plays nothing.
    /// </summary>
    [Fact]
    public void StampedeRollsForEveryStackOnceTheFightIsOver()
    {
        var fight = Fight
            .Hand(Card(IC.StrikeIronclad), Card(IC.StrikeIronclad))
            .Enemy(hp: 6)
            .PlayerBuff(BuffId.Stampede, 2)
            .Energy(3);
        var shuffle = new CountingRandom(1);
        fight.State.ShuffleRng = shuffle;

        var result = fight.EndTurn();

        Assert.True(result.Terminal);
        Assert.Equal(2, shuffle.CallCount);
        Assert.Single(fight.State.DiscardPile, c => c.DefId == IC.StrikeIronclad);
    }

    // ── The result pile: the card's own pile first, then the modifiers ──────────────────

    /// <summary>
    /// `CorruptionPower.ModifyCardPlayResultPileTypeAndPosition` sends every Skill to the
    /// exhaust pile, whatever `GetResultPileTypeForCardPlay` said: History Course's dupe of a
    /// Defend (pile none) is exhausted, next to the Defend Corruption exhausted the turn before.
    /// </summary>
    [Fact]
    public void CorruptionExhaustsAHistoryCourseDupeOfASkill()
    {
        var fight = Fight.WithRelics(RelicEffects.HistoryCourse).Enemy(hp: 200).Energy(3);
        fight.State.Hand = [Card(IC.DefendIronclad)];
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Corruption, 1);

        fight.Play(0);
        fight.EndTurn();

        Assert.Equal(2, fight.State.ExhaustPile.Count(c => c.DefId == IC.DefendIronclad));
    }

    /// <summary>
    /// The result pile is decided as the play is set up, before the card's effect: the
    /// Corruption that Catastrophe auto-plays does not exhaust Catastrophe (refset
    /// C0164-LHUH2MNFN9: the game's Catastrophe+ lands in the discard).
    /// </summary>
    [Fact]
    public void ACorruptionPlayedInsideACardDoesNotExhaustIt()
    {
        var fight = Fight
            .Hand(Card(CL.Catastrophe))
            .Enemy(hp: 100)
            .Draw(Card(IC.Corruption))
            .Energy(3);

        fight.Play(0);

        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.Corruption));
        Assert.Contains(fight.State.DiscardPile, c => c.DefId == CL.Catastrophe);
        Assert.DoesNotContain(fight.State.ExhaustPile, c => c.DefId == CL.Catastrophe);
    }

    /// <summary>
    /// `ShiningStrike` does not override `GetResultPileTypeForCardPlay`: its pile is the
    /// discard, so `ReboundPower` redirects it and is spent (`AfterModifyingCardPlayResultPile
    /// OrPosition`); the card then tops itself in its own `OnPlay`. A Rebound left over would
    /// send the next card to the draw pile.
    /// </summary>
    [Fact]
    public void ReboundIsSpentOnShiningStrike()
    {
        Assert.Equal("ShiningStrike", GeneratedData.Cards.Get(429).Name);
        var fight = Fight
            .Hand(Card(429), Card(IC.StrikeIronclad))
            .Enemy(hp: 100)
            .Draw(Card(IC.DefendIronclad))
            .PlayerBuff(BuffId.Rebound, 1)
            .Energy(3);

        fight.Play(0, 0);

        Assert.Equal(429, fight.State.DrawPile[0].DefId);
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.Rebound));

        fight.Play(0, 0);

        Assert.Contains(fight.State.DiscardPile, c => c.DefId == IC.StrikeIronclad);
        Assert.Equal(429, fight.State.DrawPile[0].DefId);
    }

    /// <summary>
    /// `FeralPower.ModifyCardPlayResultPileTypeAndPosition` returns a free Attack to hand
    /// without asking where it was going: a Shiv (Exhaust) comes back instead of exhausting.
    /// </summary>
    [Fact]
    public void FeralReturnsAShivToHand()
    {
        var fight = Fight
            .Hand(Card(SI.Shiv))
            .Enemy(hp: 100)
            .PlayerBuff(BuffId.Feral, 1)
            .Energy(3);

        fight.Play(0);

        Assert.Single(fight.State.Hand, c => c.DefId == SI.Shiv);
        Assert.DoesNotContain(fight.State.ExhaustPile, c => c.DefId == SI.Shiv);
        Assert.Equal(96, fight.State.Enemies[0].Hp);
    }
}
