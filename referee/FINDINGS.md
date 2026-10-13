# Referee findings

One line each: date, seed, step, cause, fix.

- 2026-10-13, YDT1E1R92A step 19: Splash offered other characters' Attacks from pools in the wrong order (Silent, Defect, Necrobinder, Regent; the game uses ModelDb.AllCharacters order Silent, Regent, Necrobinder, Defect). Same RNG draws, different cards. Fixed: CardEffects.OpenSplashSelection uses RunConstants.OtherCharacterPoolFor.
- 2026-10-13, suspect (unverified): RunRewardGenerator's "every pool" card rewards (WidensCardRewardsToEveryPool) also list Silent, Defect, Necrobinder, Regent. Check against the referee before changing.
- 2026-10-13, 0CERYNNVH0 and LT3K8D3K44: Neow's Bones offers two relics from inside its own relic reward (a nested rewards screen); the driver now claims nested screens. Still differs at step 0: game Niche counter 10 vs emulator 3, different curse and deck. Open: check whether the emulator replay takes both relics, then compare NeowsBones.AfterObtained with the emulator.
