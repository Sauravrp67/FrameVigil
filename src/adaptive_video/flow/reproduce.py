"""Explicit Python helper for repeating analysis of a newly collected capture.

Reviewers may run shadow, reliability and summary separately. This helper does
not download inputs, run inference, copy historical results or alter module globals.
"""

from pathlib import Path

from . import reliability, shadow, summarize

def run(capture, output, detector_cost=None):
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    
    shadow.run(capture, destination / 'shadow')
    reliability.run(destination / 'shadow', destination / 'reliability')
    summarize.run(destination / 'shadow', destination / 'reliability',
                  destination / 'summary', detector_cost=detector_cost)
