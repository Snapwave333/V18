#!/usr/bin/env python3
"""
Simple Console Loading Screen for VJ Engine
"""

import sys
import time
import os


def run_loading_screen(duration=120):
    """Run a simple loading screen"""
    start_time = time.time()

    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        remaining = max(0, duration - elapsed)
        mins = int(remaining // 60)
        secs = int(remaining % 60)
        progress = elapsed / duration

        # Clear line and print
        print(f"\r{'=' * 60}", end="")
        print(f"\r  V18 - BUILDING {duration}s DELAY BUFFER", end="")
        print(f"\r{'=' * 60}", end="")
        print(f"\r  Time Remaining: {mins:02d}:{secs:02d}", end="")
        print(
            f"\r  Progress: [{'█' * int(progress * 30):<30}] {int(progress * 100)}%",
            end="",
        )
        print(f"\r{'=' * 60}", end="")

        time.sleep(0.5)

    print("\r" + " " * 60)
    print("=" * 60)
    print("  ★ SHOWTIME - BUFFER READY ★")
    print("=" * 60)


if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    run_loading_screen(duration)
