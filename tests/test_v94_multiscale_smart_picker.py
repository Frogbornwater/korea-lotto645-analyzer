import numpy as np
import pandas as pd
import pytest

from v94_multiscale_smart_picker import (
    ENGINE_VERSION, fit_v94_multiscale_next, generate_v94_tickets,
)


def _history(rows: int = 1001) -> pd.DataFrame:
    rng = np.random.default_rng(9401)
    records = []
    for draw in range(1, rows + 1):
        numbers = sorted(rng.choice(np.arange(1, 46), size=6, replace=False).tolist())
        bonus_pool = sorted(set(range(1, 46)) - set(numbers))
        bonus = int(rng.choice(bonus_pool))
        records.append({
            "draw_no": draw, "date": "", **{f"n{i}": numbers[i - 1] for i in range(1, 7)},
            "bonus": bonus,
        })
    return pd.DataFrame(records)


def test_v94_probability_vector_and_target_draw():
    prediction = fit_v94_multiscale_next(_history())
    assert prediction["engine_version"] == ENGINE_VERSION
    assert prediction["target_draw"] == 1002
    assert prediction["history_last_draw"] == 1001
    assert len(prediction["number_probabilities"]) == 45
    assert prediction["probability_sum"] == pytest.approx(6.0, abs=1e-8)


def test_v94_generates_ten_valid_diversified_tickets():
    prediction = fit_v94_multiscale_next(_history())
    frame, effective_overlap, relaxed = generate_v94_tickets(
        prediction, count=10, seed=9402, max_overlap=4
    )
    assert len(frame) == 10
    assert frame["号码"].is_unique
    tickets = [
        [int(value) for value in text.split(" · ")]
        for text in frame["号码"]
    ]
    assert all(len(ticket) == 6 and len(set(ticket)) == 6 for ticket in tickets)
    assert all(all(1 <= number <= 45 for number in ticket) for ticket in tickets)
    assert all(
        len(set(left).intersection(right)) <= effective_overlap
        for index, left in enumerate(tickets)
        for right in tickets[index + 1:]
    )
    assert relaxed is (effective_overlap > 4)


def test_v94_rejects_short_or_gapped_history():
    with pytest.raises(ValueError, match="至少需要1001期"):
        fit_v94_multiscale_next(_history(1000))
    gapped = _history()
    gapped.loc[500, "draw_no"] = 9999
    with pytest.raises(ValueError, match="期号存在缺失"):
        fit_v94_multiscale_next(gapped)
