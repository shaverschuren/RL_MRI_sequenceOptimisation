"""Module implementing several environments used for RL optimization

We provide definitions for two environments, namely:
- SimulationEnv (An environment for optimizing EPG sequence parameters)
- ScannerEnv (An environment for optimizing actual scanner sequence parameters)

Additionally, these environments may be adjusted to accept
either continuous or discrete action spaces. Also, optimisations
may be performed over both snr and cnr.
"""


class SimulationEnv(object):
    pass


class ScannerEnv(object):
    pass
