using Xunit;

// The emulator keeps static state (handle pools, factory fields, profiling hooks), so its tests
// must not run concurrently in one process: the same rule as src/Sts2Emulator.Tests/AssemblyInfo.cs.
// Seen: SetupFightTests failed once with "Collection was modified" while another class ran.
[assembly: CollectionBehavior(DisableTestParallelization = true)]
