import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from simplettsworkflow.candidates import CandidateStore


class CandidateStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = CandidateStore(self.root)
        self.paths = []
        for run in ("original", "retry"):
            directory = self.root / run
            directory.mkdir()
            items = []
            for index in (1, 2):
                path = directory / f"line_{index:03}.wav"
                path.write_bytes(b"test audio")
                self.paths.append(path)
                items.append({"index": index, "filename": path.name})
            (directory / "metadata.json").write_text(json.dumps({
                "items": items, "emotion_analyses": [{"index": 1}, {"index": 2}],
            }), encoding="utf-8")

    def test_retains_choice_without_touching_other_lines_and_updates_metadata(self):
        original, other, selected, unrelated = self.paths
        group = {"selected": self.store.token(str(selected)), "discard": [self.store.token(str(original))]}
        self.assertEqual(self.store.retain([group]), 1)
        self.assertFalse(original.exists())
        self.assertTrue(all(path.exists() for path in (selected, other, unrelated)))
        metadata = json.loads((original.parent / "metadata.json").read_text())
        self.assertEqual([item["index"] for item in metadata["items"]], [2])
        self.assertEqual(metadata["emotion_analyses"], [{"index": 2}])
        self.store.retain([group])  # Retrying a completed cleanup is safe.

    def test_validates_all_groups_before_deleting(self):
        token = self.store.token(str(self.paths[0]))
        with self.assertRaises(ValueError):
            self.store.retain([
                {"selected": self.store.token(str(self.paths[2])), "discard": [token]},
                {"selected": "forged:../../outside.wav", "discard": []},
            ])
        self.assertTrue(all(path.exists() for path in self.paths))

    def test_rejects_missing_selection_or_conflicting_selection(self):
        token = self.store.token(str(self.paths[0]))
        with self.assertRaises(ValueError):
            self.store.retain([{"selected": token, "discard": [token]}])
        missing = self.store.token(str(self.root / "missing.wav"))
        with self.assertRaises(ValueError):
            self.store.retain([{"selected": missing, "discard": [token]}])
        self.assertTrue(self.paths[0].exists())

    def test_token_cannot_be_retargeted_or_used_outside_root(self):
        token = self.store.token(str(self.paths[0]))
        with self.assertRaises(ValueError):
            self.store.resolve(token.replace("line_001", "line_002"))
        with self.assertRaises(ValueError):
            self.store.token(str(self.root.parent / "outside.wav"))

    def test_persisted_key_keeps_candidates_valid_after_restart(self):
        key = self.root / "private" / "candidate-key"
        first = CandidateStore(self.root, key)
        chosen, discarded = self.paths[:2]
        selected_id = first.token(str(chosen))
        discarded_id = first.token(str(discarded))
        restarted = CandidateStore(self.root, key)
        restarted.retain([{"selected": selected_id, "discard": [discarded_id]}])
        self.assertTrue(chosen.exists())
        self.assertFalse(discarded.exists())


if __name__ == "__main__":
    unittest.main()
