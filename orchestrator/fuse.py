def fuse_evidence(*evidence_lists):
    fused = []
    counter = 1
    for evidence_list in evidence_lists:
        for block in evidence_list:
            fused.append(block.model_copy(update={"citation_id": counter}))
            counter += 1
    return fused
