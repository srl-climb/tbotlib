import cProfile, pstats

class Profiler:

    def __init__(self) -> None:

        self._profiler = cProfile.Profile()

    def on(self) -> None:

        self._profiler.enable()

    def off(self) -> None:
        
        self._profiler.disable()

    def print(self,sort: str = "cumulative") -> None:
        
        stats = pstats.Stats(self._profiler).sort_stats(sort)
        print()
        stats.print_stats()




