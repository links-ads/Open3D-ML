import cProfile
import logging
import pstats
from run_pipeline import main

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(asctime)s - %(module)s - %(message)s",
    )

    # Create a profiler object
    profiler = cProfile.Profile()

    # Start profiling
    profiler.enable()

    # Call the main function (or the code you want to profile)
    main()

    # Stop profiling
    profiler.disable()

    # Save the profiling results to a .prof file
    profiler.dump_stats("output_classsample_nounique.prof")

    # Optionally, print the profiling stats sorted by cumulative time
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumtime")
    stats.print_stats()
