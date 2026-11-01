"""The playing agent: drives games through the public environment wrapper.

Nothing here imports the oracle or reads game source. The same code plays the
25 public games offline (``LocalEnvironmentWrapper``) and the private games in a
competition rerun (``RemoteEnvironmentWrapper`` over the Kaggle gateway).
"""
