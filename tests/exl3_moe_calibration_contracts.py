"""Symbolic searches cover map isolation; tensor arithmetic is checked separately."""

from exl3_moe_calibration import expert_qmap


def expert_isolation(a: int, b: int) -> str:
    """
    pre: a >= 0 and b >= 0 and a != b
    post: _ != expert_qmap('block', b, 'up')
    """
    return expert_qmap("block", a, "up")


def gate_up_alignment(expert: int) -> str:
    """
    pre: expert >= 0
    post: _ == expert_qmap('block', expert, 'up')
    """
    return expert_qmap("block", expert, "gate")


def down_isolation(expert: int) -> str:
    """
    pre: expert >= 0
    post: _ != expert_qmap('block', expert, 'up')
    """
    return expert_qmap("block", expert, "down")
