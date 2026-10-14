// Which code raised a relic flash (Chooser.cs records flashes as evidence that a relic's effect fired).
// RelicModel.Flash only raises the model's Flashed event; the game's own subscribers are display code
// (Player.OnRelicFlashed: a sound and an NRelicFlashVfx; NRelicInventoryHolder: a VFX over the top
// bar). No rule reads a flash: nothing counts it, draws for it, orders hooks by it or writes history.
// Most flashes are raised by the rules themselves (a relic's hook, Hook.AfterModifyingRewards), and
// those happen the same way on the full and the nodeless host. A few are raised by display code only:
// the screens and VFX in the game's Nodes namespaces (the card reward, merchant, card select and
// transform screens) and CardCmd.FlashRelics, which only the card preview's tween callback calls
// (CardCmd.PreviewInternal, a display helper the game skips in test mode and the nodeless host
// patches out). Those land on whatever step the driver has reached when the tween ends, and only
// where the node exists, so they are not rule evidence and are not recorded on either host.
// No game types here: the classification is tested on its own (Sts2Emulator.Tests links this file).

using System;
using System.Collections.Generic;
using System.Reflection;

namespace Spireline.Referee;

internal static class FlashSource
{
    /// <summary>
    /// The method that raised the flash: the first frame, innermost first, that is neither the recorder
    /// (this driver's namespace), a delegate's Invoke, nor <paramref name="flashType" />'s own Flash overloads.
    /// Null when no frame qualifies.
    /// </summary>
    internal static MethodBase? Raiser(IEnumerable<MethodBase?> frames, Type flashType)
    {
        foreach (var m in frames)
        {
            var t = m?.DeclaringType;
            if (m == null || t == null) continue;
            if (Root(t).Namespace == typeof(FlashSource).Namespace) continue;
            if (typeof(Delegate).IsAssignableFrom(t)) continue;
            if (t == flashType && m.Name == "Flash") continue;
            return m;
        }

        return null;
    }

    /// <summary>
    /// The raiser is display code: a type (or a lambda or state machine nested in one) in the game's Nodes
    /// namespaces (one is spelled "MegaCrit.sts2.Core.Nodes", hence no case), or the card preview's flash
    /// (CardCmd.FlashRelics, or the PreviewInternal callback it may be inlined into). An unknown raiser is
    /// not display: the flash is recorded, as before.
    /// </summary>
    internal static bool IsDisplay(MethodBase? raiser)
    {
        var t = raiser?.DeclaringType;
        if (raiser == null || t == null) return false;
        var root = Root(t);
        if ((root.Namespace ?? "").StartsWith("MegaCrit.Sts2.Core.Nodes", StringComparison.OrdinalIgnoreCase)) return true;
        if (root.FullName != "MegaCrit.Sts2.Core.Commands.CardCmd") return false;
        return raiser.Name == "FlashRelics" || raiser.Name.Contains("<PreviewInternal>", StringComparison.Ordinal)
            || t.Name.Contains("<PreviewInternal>", StringComparison.Ordinal) || t.Name.Contains("<FlashRelics>", StringComparison.Ordinal);
    }

    /// <summary>A short name for the raiser ("CardCmd.FlashRelics"), for the diagnostics.</summary>
    internal static string Name(MethodBase? raiser) =>
        raiser?.DeclaringType is { } t ? Root(t).Name + "." + raiser.Name : "?";

    private static Type Root(Type t)
    {
        while (t.DeclaringType != null) t = t.DeclaringType;
        return t;
    }
}
