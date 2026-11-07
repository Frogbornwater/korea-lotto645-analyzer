import unittest

import pandas as pd

from v40_governance import make_event
from v40_store import (
    export_event_archive, import_event_archive, latest_model_snapshot,
    events_to_forward_locks,
)


class V40StoreTests(unittest.TestCase):
    def test_archive_round_trip_and_snapshot(self):
        fingerprint = "a" * 64
        event = make_event("model_snapshot", fingerprint, {
            "selected_config": {"model_name": "贝叶斯平滑模型", "params": {"bayes_prior": 20}},
            "selection_data_fingerprint": fingerprint,
        })
        archive = export_event_archive([event])
        restored = import_event_archive(archive)
        self.assertEqual(restored[0]["event_id"], event["event_id"])
        self.assertEqual(latest_model_snapshot(restored)["payload"]["selected_config"]["model_name"], "贝叶斯平滑模型")

    def test_forward_lock_restore(self):
        fingerprint = "b" * 64
        metadata = {"target_draw": 351, "history_last_draw": 350, "model_name": "贝叶斯平滑模型"}
        rows = []
        for strategy in ["贝叶斯平滑模型", "均匀随机基准"]:
            rows.append({
                "目标期号": 351, "策略": strategy, "随机种子组": 1,
                "配置编号": 1, "注序": 1, "号码": "01-02-03-04-05-06",
                "n1": 1, "n2": 2, "n3": 3, "n4": 4, "n5": 5, "n6": 6,
            })
        event = make_event("forward_lock", fingerprint, {
            "metadata": metadata, "detail_records": rows,
        })
        restored = events_to_forward_locks([event])
        self.assertIn("351", restored)
        self.assertEqual(len(restored["351"]["detail"]), 2)


if __name__ == "__main__":
    unittest.main()
