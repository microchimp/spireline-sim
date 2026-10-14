namespace Sts2Emulator.Core;

/// <summary>
/// The playable characters, in <c>ModelDb.AllCharacters</c> order. The order is load-bearing:
/// every place the game walks "all characters" (Splash, Kaleidoscope, Prismatic Gem, Orobas)
/// walks it in this order, and a roll indexes into the result.
/// </summary>
public enum PlayerCharacter
{
    Ironclad,
    Silent,
    Regent,
    Necrobinder,
    Defect,
}

/// <summary>
/// `Player.Character.CardPool / PotionPool / RelicPool`: the ONE place the engine turns "the
/// player's character" into the pools the game reads off it. A run or a combat carries a
/// <see cref="PlayerCharacter" /> (default Ironclad) and everything that draws from "the
/// character's own" cards, potions or relics asks here, instead of naming a pool.
/// </summary>
internal static class CharacterPools
{
    public static ReadOnlySpan<int> Cards(PlayerCharacter character) =>
        character switch
        {
            PlayerCharacter.Silent => GeneratedData.CardPools.Silent,
            PlayerCharacter.Regent => GeneratedData.CardPools.Regent,
            PlayerCharacter.Necrobinder => GeneratedData.CardPools.Necrobinder,
            PlayerCharacter.Defect => GeneratedData.CardPools.Defect,
            _ => GeneratedData.CardPools.Ironclad,
        };

    public static ReadOnlySpan<int> Potions(PlayerCharacter character) =>
        character switch
        {
            PlayerCharacter.Silent => GeneratedData.PotionPools.Silent,
            PlayerCharacter.Regent => GeneratedData.PotionPools.Regent,
            PlayerCharacter.Necrobinder => GeneratedData.PotionPools.Necrobinder,
            PlayerCharacter.Defect => GeneratedData.PotionPools.Defect,
            _ => GeneratedData.PotionPools.Ironclad,
        };

    public static ReadOnlySpan<int> Relics(PlayerCharacter character) =>
        character switch
        {
            PlayerCharacter.Silent => GeneratedData.RelicPools.Silent,
            PlayerCharacter.Regent => GeneratedData.RelicPools.Regent,
            PlayerCharacter.Necrobinder => GeneratedData.RelicPools.Necrobinder,
            PlayerCharacter.Defect => GeneratedData.RelicPools.Defect,
            _ => GeneratedData.RelicPools.Ironclad,
        };

    /// <summary>
    /// `UnlockState.CharacterCardPools` minus the player's own (Splash's
    /// `list.Remove(Owner.Character.CardPool)`): the other four characters, in
    /// <c>ModelDb.AllCharacters</c> order, on a profile with every character unlocked. A run or
    /// combat asks its own <see cref="UnlockState.Others" />.
    /// </summary>
    public static IEnumerable<PlayerCharacter> Others(PlayerCharacter own) =>
        Enum.GetValues<PlayerCharacter>().Where(character => character != own);

    /// <summary>The pool that holds a card, or null for a card no character owns.</summary>
    public static PlayerCharacter? OwnerOfCard(int cardId)
    {
        foreach (var character in Enum.GetValues<PlayerCharacter>())
        {
            if (Cards(character).IndexOf(cardId) >= 0)
            {
                return character;
            }
        }

        return null;
    }
}
