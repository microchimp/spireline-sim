using System.Text;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Interop;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// A driver reads the rewards screen from the state list (<c>Sts2Run_GetStateList</c>, list 6:
/// gold, potion, relic, card items) and answers it from the action mask. The two must name the
/// same card items: Kaleidoscope's rewards moved from a counter the list read to the card-offer
/// queue it did not (154c8dd2), so after the first card was picked the mask still offered the
/// second reward while the list said the screen held no card, and a driver that took the list at
/// its word left the screen with the reward unclaimed (tests/python/test_replay_real_run.py,
/// QS2GYXRKWN-a8). Called through the unmanaged entry points ctypes uses.
/// </summary>
public unsafe class RewardScreenListTests
{
    private const int RewardsList = 6;
    private const int Ascension = 8;

    [Fact]
    public void KaleidoscopesQueuedRewardIsACardItemOnTheListAsWellAsOnTheMask()
    {
        int run = Create();
        try
        {
            Reset(run, "QS2GYXRKWN"); // Neow's first option on this seed is Kaleidoscope
            Step(run, 0);
            Assert.Equal((int)RunPhase.RelicReward, Phase(run));
            AssertListAndMaskAgree(run, expectedCardItems: 1);

            Step(run, 0); // claim the first reward: its card screen
            Assert.Equal((int)RunPhase.CardReward, Phase(run));
            Step(run, 0); // take a card: back on the rewards screen, the second reward waiting
            Assert.Equal((int)RunPhase.RelicReward, Phase(run));
            AssertListAndMaskAgree(run, expectedCardItems: 1);

            Step(run, 0); // claim the second
            Assert.Equal((int)RunPhase.CardReward, Phase(run));
            Step(run, 0);
            AssertNoCardItems(run);
        }
        finally
        {
            Destroy(run);
        }
    }

    private static void AssertListAndMaskAgree(int run, int expectedCardItems)
    {
        int[] list = StateList(run, RewardsList);
        int[] mask = LegalActions(run);
        int cardItems = list[3];
        Assert.Equal(expectedCardItems, cardItems);
        // The items sit in the order gold, potion, relic, card items; the leave action is
        // always legal beside them.
        int itemsBeforeCards = list[..3].Count(count => count != 0);
        int[] cardActions = mask.Where(a => a != RunConstants.RewardSkipAction).ToArray();
        Assert.Equal(
            Enumerable.Range(itemsBeforeCards, cardItems),
            cardActions.Where(a => a >= itemsBeforeCards)
        );
    }

    private static void AssertNoCardItems(int run)
    {
        Assert.Equal(0, StateList(run, RewardsList)[3]);
    }

    private static int Create()
    {
        delegate* unmanaged<int> create = &NativeExports.Sts2Run_Create;
        return create();
    }

    private static void Destroy(int run)
    {
        delegate* unmanaged<int, void> destroy = &NativeExports.Sts2Run_Destroy;
        destroy(run);
    }

    private static void Reset(int run, string seed)
    {
        delegate* unmanaged<int, byte*, int, int, int*, int> reset =
            &NativeExports.Sts2Run_Reset;
        byte[] text = Encoding.UTF8.GetBytes(seed);
        var obs = new int[RunConstants.RunObsSize];
        fixed (byte* s = text)
        fixed (int* o = obs)
        {
            Assert.Equal(0, reset(run, s, text.Length, Ascension, o));
        }
    }

    private static void Step(int run, int action)
    {
        delegate* unmanaged<int, int, int, int*, float*, int*, int*, int> step =
            &NativeExports.Sts2Run_Step;
        var obs = new int[RunConstants.RunObsSize];
        float reward;
        int terminal,
            truncated;
        fixed (int* o = obs)
        {
            Assert.Equal(0, step(run, action, -1, o, &reward, &terminal, &truncated));
        }
    }

    private static int Phase(int run)
    {
        delegate* unmanaged<int, int> phase = &NativeExports.Sts2Run_GetPhase;
        return phase(run);
    }

    private static int[] StateList(int run, int listId)
    {
        delegate* unmanaged<int, int, int*, int, int> list = &NativeExports.Sts2Run_GetStateList;
        var buffer = new int[8];
        fixed (int* b = buffer)
        {
            Assert.True(list(run, listId, b, buffer.Length) >= 0);
        }

        return buffer;
    }

    private static int[] LegalActions(int run)
    {
        delegate* unmanaged<int, int*, int, int> mask = &NativeExports.Sts2Run_ActionMask;
        var buffer = new int[RunConstants.MaxActions];
        fixed (int* m = buffer)
        {
            Assert.Equal(0, mask(run, m, buffer.Length));
        }

        return Enumerable.Range(0, buffer.Length).Where(i => buffer[i] != 0).ToArray();
    }
}
