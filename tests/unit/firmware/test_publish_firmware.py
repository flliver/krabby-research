"""Unit tests for firmware.scripts.publish_firmware pure helpers (builds.json merge)."""
from firmware.scripts.publish_firmware import _merge_build


def _entry(build_key, commit, ver="v"):
    return {
        "build_key": build_key,
        "commit": commit,
        "commit_date": "2026-01-01",
        "ver_string": ver,
        "hex_url": "h",
        "manifest_url": "m",
    }


class TestMergeBuild:
    def test_appends_new_commit(self):
        doc = {"schema_version": 1, "branch": "release/0.2.0", "builds": []}
        _merge_build(doc, _entry("20260101-000000-aaa0001", "aaa0001"))
        assert [b["commit"] for b in doc["builds"]] == ["aaa0001"]

    def test_seeds_schema_version(self):
        doc = {"branch": "release/0.2.0", "builds": []}
        _merge_build(doc, _entry("20260101-000000-aaa0001", "aaa0001"))
        assert doc["schema_version"] == 1

    def test_daily_rebuild_of_same_commit_does_not_duplicate(self):
        # A scheduled rebuild produces a fresh build_key (timestamp) but the SAME
        # commit — must collapse to one entry (the newer build_key wins), not two.
        doc = {"schema_version": 1, "branch": "release/0.2.0", "builds": []}
        _merge_build(doc, _entry("20260101-070000-aaa0001", "aaa0001", "old"))
        _merge_build(doc, _entry("20260102-070000-aaa0001", "aaa0001", "new"))
        assert len(doc["builds"]) == 1
        assert doc["builds"][0]["build_key"] == "20260102-070000-aaa0001"
        assert doc["builds"][0]["ver_string"] == "new"

    def test_distinct_commits_accumulate(self):
        doc = {"schema_version": 1, "branch": "release/0.2.0", "builds": []}
        _merge_build(doc, _entry("20260101-070000-aaa0001", "aaa0001"))
        _merge_build(doc, _entry("20260102-070000-bbb0002", "bbb0002"))
        assert [b["commit"] for b in doc["builds"]] == ["aaa0001", "bbb0002"]

    def test_cap_keeps_newest(self):
        doc = {"schema_version": 1, "branch": "release/0.2.0", "builds": []}
        for i in range(5):
            _merge_build(doc, _entry(f"2026010{i}-000000-c{i}", f"c{i}"), cap=3)
        assert len(doc["builds"]) == 3
        # newest three appended (c2, c3, c4) survive the cap
        assert [b["commit"] for b in doc["builds"]] == ["c2", "c3", "c4"]
