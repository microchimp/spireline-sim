using Sts2Emulator.Core;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// <c>PowerInstanceType.Instanced</c>: <c>PowerCmd.FindExistingInstanceForStacking</c> returns
/// null for it, so every application builds a power of its own with its own Amount and
/// DynamicVars and appends it to <c>Creature._powers</c>; the hooks then run over the powers in
/// that order. <see cref="BuffSystem.IsInstanced" /> is the emulator's list of those ids, and
/// these pin what the list means to <c>Apply</c>, to addressing one instance, and to a copy.
/// </summary>
public class InstancedPowerTests
{
    [Fact]
    public void EveryApplicationIsAnEntryOfItsOwnWithItsOwnData()
    {
        var buffs = new List<BuffState>();

        BuffSystem.Apply(buffs, BuffId.TheBombPower, 3, 40);
        BuffSystem.Apply(buffs, BuffId.TheBombPower, 2, 50);

        Assert.Equal(
            [new BuffState(BuffId.TheBombPower, 3, 40), new BuffState(BuffId.TheBombPower, 2, 50)],
            buffs
        );
    }

    /// <summary>A power the game does not instance still stacks into one entry.</summary>
    [Fact]
    public void APowerThatIsNotInstancedStillStacks()
    {
        var buffs = new List<BuffState>();

        BuffSystem.Apply(buffs, BuffId.Strength, 2);
        BuffSystem.Apply(buffs, BuffId.Strength, 3);

        Assert.Equal([new BuffState(BuffId.Strength, 5)], buffs);
    }

    /// <summary>
    /// The k-th instance is the k-th entry of that id in list order, whatever sits between
    /// them; past the last there is none.
    /// </summary>
    [Fact]
    public void InstanceIndexCountsOnlyEntriesOfThatIdOldestFirst()
    {
        var buffs = new List<BuffState>
        {
            new(BuffId.Strength, 1),
            new(BuffId.TheBombPower, 3, 40),
            new(BuffId.Weak, 1),
            new(BuffId.TheBombPower, 1, 50),
        };

        Assert.Equal(1, BuffSystem.InstanceIndex(buffs, BuffId.TheBombPower, 0));
        Assert.Equal(3, BuffSystem.InstanceIndex(buffs, BuffId.TheBombPower, 1));
        Assert.Equal(-1, BuffSystem.InstanceIndex(buffs, BuffId.TheBombPower, 2));
        Assert.Equal(-1, BuffSystem.InstanceIndex(buffs, BuffId.Knockdown, 0));
    }

    /// <summary>
    /// There is no instance to subtract from, so a negative <c>Apply</c> is a caller that
    /// still reads the power as a stack. Failing loudly beats appending a power of -1.
    /// </summary>
    [Fact]
    public void ANegativeApplyIsRefused()
    {
        var buffs = new List<BuffState> { new(BuffId.TheBombPower, 3, 40) };

        Assert.Throws<InvalidOperationException>(() =>
            BuffSystem.Apply(buffs, BuffId.TheBombPower, -1)
        );
        Assert.Equal([new BuffState(BuffId.TheBombPower, 3, 40)], buffs);
    }

    /// <summary>
    /// A removal that names the id takes every instance: `AfterSideTurnEnd` runs on each
    /// instance of Knockdown and each one removes itself.
    /// </summary>
    [Fact]
    public void RemoveTakesEveryInstance()
    {
        var buffs = new List<BuffState>
        {
            new(BuffId.Knockdown, 2),
            new(BuffId.Weak, 1),
            new(BuffId.Knockdown, 3),
        };

        BuffSystem.Remove(buffs, BuffId.Knockdown);

        Assert.Equal([new BuffState(BuffId.Weak, 1)], buffs);
    }

    /// <summary>
    /// An Artifact is eaten per APPLICATION of a debuff, and each instance is an application:
    /// one charge stops one Knockdown and the next lands.
    /// </summary>
    [Fact]
    public void ArtifactStopsOneApplicationAtATime()
    {
        var buffs = new List<BuffState> { new(BuffId.Artifact, 1) };

        BuffSystem.Apply(buffs, BuffId.Knockdown, 2);
        BuffSystem.Apply(buffs, BuffId.Knockdown, 2);

        Assert.Equal([new BuffState(BuffId.Knockdown, 2)], buffs);
    }

    /// <summary>A copy of the combat keeps each instance, its Data, and its place in the order.</summary>
    [Fact]
    public void ACloneKeepsEveryInstanceItsDataAndTheirOrder()
    {
        var fight = Fight.Hand().Enemy(hp: 50);
        fight.PlayerBuff(BuffId.TheBombPower, 2, data: 40).PlayerBuff(BuffId.TheBombPower, 3, data: 50);

        var clone = fight.State.Clone();
        BuffSystem.Remove(fight.State.PlayerBuffs, BuffId.TheBombPower);

        Assert.Equal(
            [new BuffState(BuffId.TheBombPower, 2, 40), new BuffState(BuffId.TheBombPower, 3, 50)],
            clone.PlayerBuffs
        );
    }
}
