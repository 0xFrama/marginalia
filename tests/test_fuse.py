from models.evidence import Evidence
from orchestrator.fuse import fuse_evidence


def patient_block(n):
    return Evidence(citation_id=n, text=f"p{n}", kind="patient")


def guideline_block(n):
    return Evidence(citation_id=n, text=f"g{n}", kind="guideline")


def test_renumbers_across_both_lists():
    patient = [patient_block(1), patient_block(2)]
    guideline = [guideline_block(1)]

    fused = fuse_evidence(patient, guideline)

    # 1..N, unique and continuous across the combined list
    assert [b.citation_id for b in fused] == [1, 2, 3]
    # patient first, then guideline
    assert [b.kind for b in fused] == ["patient", "patient", "guideline"]


def test_single_list_still_renumbers():
    fused = fuse_evidence([guideline_block(5), guideline_block(9)])
    assert [b.citation_id for b in fused] == [1, 2]


def test_does_not_mutate_inputs():
    original = [guideline_block(7)]
    fuse_evidence(original)
    # the caller's block keeps its original number; fusing made a copy
    assert original[0].citation_id == 7


def test_empty_input():
    assert fuse_evidence() == []
    assert fuse_evidence([], []) == []
