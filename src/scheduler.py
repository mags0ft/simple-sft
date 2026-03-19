"""
Schedules the execution of n threads in parallel and correctly writes their
responses atomically into an output JSONL file. Handles resuming after a crash.
"""

import threading
