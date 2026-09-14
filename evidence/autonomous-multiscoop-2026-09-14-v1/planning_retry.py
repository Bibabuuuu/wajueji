"""Bounded policy for retrying a stale dispatch observation."""

MAX_PLANNING_ATTEMPTS=3


def replan_allowed(error,cycle_number,attempt):
    return (cycle_number>1 and attempt<MAX_PLANNING_ATTEMPTS and
            str(error).startswith('cut_selection_invalid:'))
