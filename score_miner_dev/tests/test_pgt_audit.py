import json

from score_miner_core.validator_sim.pgt_audit import audit_pgt


def test_audit_pgt_blocks_unreviewed_bootstrap(tmp_path) -> None:
    pgt = tmp_path / "pgt.json"
    pgt.write_text(
        json.dumps(
            {
                "review_required": True,
                "annotations": [
                    {
                        "frame_id": 1,
                        "bbox": [1, 2, 3, 4],
                        "label": "player",
                        "review_status": "needs_manual_review",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = audit_pgt(pgt)

    assert report.score_ready is False
    assert report.review_required is True
    assert report.annotations == 1
    assert report.frames == 1

