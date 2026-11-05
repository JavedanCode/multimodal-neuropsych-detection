"""Verify that base-model cross-fitting uses only the allowed training people."""


def check_crossfit(records, expected, forbidden):
    held = []
    for r in records:
        a = set(r["fit_ids"])
        b = set(r["heldout_ids"])
        assert (
            a.isdisjoint(b) and a | b == set(expected) and (a | b).isdisjoint(forbidden)
        )
        held.extend(r["heldout_ids"])
    assert len(held) == len(set(held)) and set(held) == set(expected)
