"""
The reference agent: a recurrent MAPPO policy trained by self-play.

`agent` imports torch; `observation` does not, so the auditor can reuse the
encoder for its belief-matrix checks without pulling in a deep-learning stack.
Nothing is imported eagerly here for that reason.
"""
