"""
Headless entrypoint. Same as:

  python script_for_raspy.py --headless
"""

import sys

from script_for_raspy import main

if __name__ == "__main__":
    # keep old habit: this file always runs without a window
    if "--headless" not in sys.argv:
        sys.argv.append("--headless")
    main()
