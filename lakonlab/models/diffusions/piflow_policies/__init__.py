from .dx import DXPolicy
from .gmflow import GMFlowPolicy
from .fourier import FourierPolicy


POLICY_CLASSES = dict(
    DX=DXPolicy,
    GMFlow=GMFlowPolicy,
    Fourier=FourierPolicy
)
