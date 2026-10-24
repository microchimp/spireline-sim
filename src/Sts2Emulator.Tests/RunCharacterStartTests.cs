using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.GeneratedData;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `RunEngine.Reset` with a character: each `CharacterModel`'s StartingDeck, StartingRelics and
/// StartingHp, and the character's own pool in the relic bag -- which UpFront shuffles before
/// the rooms, so a run of anyone else is a different run from the same seed.
/// </summary>
public class RunCharacterStartTests
{
    [Theory]
    [InlineData(PlayerCharacter.Silent, 70, "RING_OF_THE_SNAKE", 12, "NEUTRALIZE")]
    [InlineData(PlayerCharacter.Defect, 75, "CRACKED_CORE", 10, "DUALCAST")]
    [InlineData(PlayerCharacter.Regent, 75, "DIVINE_RIGHT", 10, "VENERATE")]
    [InlineData(PlayerCharacter.Necrobinder, 66, "BOUND_PHYLACTERY", 10, "BODYGUARD")]
    [InlineData(PlayerCharacter.Ironclad, 80, "BURNING_BLOOD", 10, "BASH")]
    public void Each_character_starts_with_its_own_kit(
        PlayerCharacter character,
        int maxHp,
        string relic,
        int deckSize,
        string signatureCard
    )
    {
        var engine = new RunEngine();
        engine.Reset("CHARSTART1", ascension: 0, character);
        var s = engine.State;

        Assert.Equal(character, s.Character);
        Assert.Equal(maxHp, s.PlayerMaxHp);
        Assert.Equal(maxHp, s.PlayerHp); // Neow heals the whole of it below WearyTraveler
        Assert.Equal(99, s.Gold);
        Assert.Equal(relic, Relics.Get(Assert.Single(s.Relics).DefId).Entry);
        Assert.Equal(deckSize, s.Deck.Count);
        Assert.Contains(s.Deck, c => Cards.Get(c.DefId).Entry == signatureCard);
    }

    [Fact]
    public void Ascenders_bane_joins_every_characters_deck_from_its_level()
    {
        var engine = new RunEngine();
        engine.Reset("CHARSTART1", ascension: Ascension.AscendersBane, PlayerCharacter.Silent);
        Assert.Equal(13, engine.State.Deck.Count);
        Assert.Equal(RunConstants.CardAscendersBane, engine.State.Deck[^1].DefId);
    }

    [Fact]
    public void The_default_start_is_the_ironclad_as_before()
    {
        var plain = new RunEngine();
        plain.Reset("CHARSTART1", 10);
        var named = new RunEngine();
        named.Reset("CHARSTART1", 10, PlayerCharacter.Ironclad);
        Assert.Equal(plain.State.Deck.Select(c => c.DefId), named.State.Deck.Select(c => c.DefId));
        Assert.Equal(plain.State.NeowOptions, named.State.NeowOptions);
        Assert.Equal(plain.State.Rng.UpFront.CallCount, named.State.Rng.UpFront.CallCount);
    }

    [Fact]
    public void The_character_pool_moves_the_draws_after_the_relic_bag()
    {
        // Same seed, different character: the relic bag holds a different pool, so UpFront
        // stands somewhere else once it is shuffled -- and with it every room the act rolls.
        var ironclad = new RunEngine();
        ironclad.Reset("CHARSTART1", 10);
        var silent = new RunEngine();
        silent.Reset("CHARSTART1", 10, PlayerCharacter.Silent);
        var snake = RunNonCombatEffects.NamedRelic("RingOfTheSnake");
        Assert.DoesNotContain(snake, ironclad.State.RelicBag.Remaining);
        Assert.Contains(
            silent.State.RelicBag.Remaining,
            id => CharacterPools.Relics(PlayerCharacter.Silent).ToArray().Contains(id)
        );
        Assert.DoesNotContain(
            ironclad.State.RelicBag.Remaining,
            id => CharacterPools.Relics(PlayerCharacter.Silent).ToArray().Contains(id)
                && !CharacterPools.Relics(PlayerCharacter.Ironclad).ToArray().Contains(id)
        );
    }

    /// <summary>
    /// `Sts2Run_ResetAs`: the native export that starts a run as a named character (the lockstep's
    /// `--character X`). Each of the five characters starts with its own kit through the export, and a
    /// character or ascension the engine does not model is refused before it can abort the host.
    /// </summary>
    [Theory]
    [InlineData(PlayerCharacter.Ironclad, 80, "BURNING_BLOOD", 10)]
    [InlineData(PlayerCharacter.Silent, 70, "RING_OF_THE_SNAKE", 12)]
    [InlineData(PlayerCharacter.Regent, 75, "DIVINE_RIGHT", 10)]
    [InlineData(PlayerCharacter.Necrobinder, 66, "BOUND_PHYLACTERY", 10)]
    [InlineData(PlayerCharacter.Defect, 75, "CRACKED_CORE", 10)]
    public unsafe void ResetAs_starts_the_run_as_the_named_character(
        PlayerCharacter character,
        int maxHp,
        string relic,
        int deckSize
    )
    {
        int handle = Sts2Emulator.Interop.RunNativeExports.Sts2Run_Create();
        try
        {
            var obs = new int[RunConstants.RunObsSize];
            byte[] seed = System.Text.Encoding.UTF8.GetBytes("RESETAS1");
            int status;
            fixed (byte* seedPtr = seed)
            fixed (int* obsPtr = obs)
            {
                status = Sts2Emulator.Interop.RunNativeExports.Sts2Run_ResetAs(
                    handle, seedPtr, seed.Length, 0, (int)character, obsPtr);
            }

            Assert.Equal(0, status);
            Assert.True(Sts2Emulator.Interop.RunNativeExports.TryGetRun(handle, out var run));
            var s = run.State;
            Assert.Equal(character, s.Character);
            Assert.Equal(maxHp, s.PlayerMaxHp);
            Assert.Equal(maxHp, s.PlayerHp);
            Assert.Equal(deckSize, s.Deck.Count);
            Assert.Equal(relic, Relics.Get(Assert.Single(s.Relics).DefId).Entry);
            Assert.Equal(
                RunConstants.StartingDeckIds(0, character),
                s.Deck.Select(c => c.DefId).ToArray()
            );
        }
        finally
        {
            Sts2Emulator.Interop.RunNativeExports.Sts2Run_Destroy(handle);
        }
    }

    [Fact]
    public unsafe void ResetAs_refuses_an_unknown_character_handle_or_ascension_without_touching_the_run()
    {
        int handle = Sts2Emulator.Interop.RunNativeExports.Sts2Run_Create();
        try
        {
            var obs = new int[RunConstants.RunObsSize];
            byte[] seed = System.Text.Encoding.UTF8.GetBytes("RESETAS1");
            fixed (byte* seedPtr = seed)
            fixed (int* obsPtr = obs)
            {
                Assert.Equal(-3, Sts2Emulator.Interop.RunNativeExports.Sts2Run_ResetAs(handle, seedPtr, seed.Length, 0, 5, obsPtr));
                Assert.Equal(-3, Sts2Emulator.Interop.RunNativeExports.Sts2Run_ResetAs(handle, seedPtr, seed.Length, 0, -1, obsPtr));
                Assert.Equal(-2, Sts2Emulator.Interop.RunNativeExports.Sts2Run_ResetAs(handle, seedPtr, seed.Length, 99, 1, obsPtr));
                Assert.Equal(-1, Sts2Emulator.Interop.RunNativeExports.Sts2Run_ResetAs(255, seedPtr, seed.Length, 0, 1, obsPtr));
            }
        }
        finally
        {
            Sts2Emulator.Interop.RunNativeExports.Sts2Run_Destroy(handle);
        }
    }
}
