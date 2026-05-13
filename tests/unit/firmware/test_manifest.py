import pytest

from firmware.manifest import (
    BuildManifest,
    BranchEntry,
    FirmwareIndex,
    parse_manifest,
    parse_latest,
    parse_index,
    latest_release_branch,
)


# --- fixtures ---

MANIFEST_DATA = {
    "schema_version": 1,
    "branch": "release/0.2.0",
    "commit": "abc1234",
    "commit_date": "2026-05-13",
    "build_timestamp": "2026-05-13T12:34:56Z",
    "board_fqbn": "arduino:avr:mega",
    "ver_string": "0.2.0 release/0.2.0 abc1234",
    "hex_filename": "firmware.hex",
}

LATEST_DATA = {
    "branch": "release/0.2.0",
    "build_key": "20260513-123456-abc1234",
    "hex_url": "https://s3.example.com/krabby-firmware-public/release%2F0.2.0/20260513-123456-abc1234/firmware.hex",
    "manifest_url": "https://s3.example.com/krabby-firmware-public/release%2F0.2.0/20260513-123456-abc1234/manifest.json",
}

INDEX_DATA = {
    "schema_version": 1,
    "updated": "2026-05-13T12:34:56Z",
    "branches": {
        "mainline": {
            "build_key": "20260513-123456-abc1234",
            "hex_url": "https://s3.example.com/.../mainline/.../firmware.hex",
            "manifest_url": "https://s3.example.com/.../mainline/.../manifest.json",
        },
        "release/0.2.0": {
            "build_key": "20260512-100000-def5678",
            "hex_url": "https://s3.example.com/.../release%2F0.2.0/.../firmware.hex",
            "manifest_url": "https://s3.example.com/.../release%2F0.2.0/.../manifest.json",
        },
        "release/0.1.0": {
            "build_key": "20260401-090000-ghi9012",
            "hex_url": "https://s3.example.com/.../release%2F0.1.0/.../firmware.hex",
            "manifest_url": "https://s3.example.com/.../release%2F0.1.0/.../manifest.json",
        },
    },
}


# --- parse_manifest ---

class TestParseManifest:
    def test_parses_all_fields(self):
        m = parse_manifest(MANIFEST_DATA)
        assert isinstance(m, BuildManifest)
        assert m.branch == "release/0.2.0"
        assert m.commit == "abc1234"
        assert m.commit_date == "2026-05-13"
        assert m.build_timestamp == "2026-05-13T12:34:56Z"
        assert m.board_fqbn == "arduino:avr:mega"
        assert m.ver_string == "0.2.0 release/0.2.0 abc1234"
        assert m.hex_filename == "firmware.hex"
        assert m.schema_version == 1

    def test_schema_version_coerced_to_int(self):
        data = {**MANIFEST_DATA, "schema_version": "1"}
        assert parse_manifest(data).schema_version == 1

    def test_missing_field_raises(self):
        for field in ("branch", "commit", "commit_date", "build_timestamp",
                      "board_fqbn", "ver_string", "hex_filename"):
            data = {k: v for k, v in MANIFEST_DATA.items() if k != field}
            with pytest.raises(ValueError, match=field):
                parse_manifest(data)


# --- parse_latest ---

class TestParseLatest:
    def test_parses_all_fields(self):
        e = parse_latest(LATEST_DATA)
        assert isinstance(e, BranchEntry)
        assert e.branch == "release/0.2.0"
        assert e.build_key == "20260513-123456-abc1234"
        assert "firmware.hex" in e.hex_url
        assert "manifest.json" in e.manifest_url

    def test_missing_field_raises(self):
        for field in ("branch", "build_key", "hex_url", "manifest_url"):
            data = {k: v for k, v in LATEST_DATA.items() if k != field}
            with pytest.raises(ValueError, match=field):
                parse_latest(data)


# --- parse_index ---

class TestParseIndex:
    def test_parses_all_branches(self):
        idx = parse_index(INDEX_DATA)
        assert isinstance(idx, FirmwareIndex)
        assert idx.schema_version == 1
        assert set(idx.branches.keys()) == {"mainline", "release/0.2.0", "release/0.1.0"}

    def test_branch_entries_are_branch_entry_objects(self):
        idx = parse_index(INDEX_DATA)
        e = idx.branches["release/0.2.0"]
        assert isinstance(e, BranchEntry)
        assert e.branch == "release/0.2.0"
        assert e.build_key == "20260512-100000-def5678"

    def test_empty_branches_is_valid(self):
        data = {**INDEX_DATA, "branches": {}}
        idx = parse_index(data)
        assert idx.branches == {}

    def test_missing_top_level_field_raises(self):
        for field in ("schema_version", "updated", "branches"):
            data = {k: v for k, v in INDEX_DATA.items() if k != field}
            with pytest.raises(ValueError, match=field):
                parse_index(data)

    def test_branch_entry_missing_field_raises(self):
        data = {
            **INDEX_DATA,
            "branches": {
                "mainline": {"build_key": "20260513-123456-abc1234", "hex_url": "x"},
                # missing manifest_url
            },
        }
        with pytest.raises(ValueError, match="manifest_url"):
            parse_index(data)


# --- latest_release_branch ---

class TestLatestReleaseBranch:
    def test_returns_most_recently_built_release_branch(self):
        idx = parse_index(INDEX_DATA)
        best = latest_release_branch(idx)
        # release/0.2.0 has build_key 20260512-... vs release/0.1.0's 20260401-...
        assert best.branch == "release/0.2.0"

    def test_ignores_mainline(self):
        # Even though mainline has a newer build_key, it must not be returned
        idx = parse_index(INDEX_DATA)
        best = latest_release_branch(idx)
        assert best.branch != "mainline"

    def test_returns_none_when_no_release_branches(self):
        data = {**INDEX_DATA, "branches": {
            "mainline": INDEX_DATA["branches"]["mainline"],
        }}
        idx = parse_index(data)
        assert latest_release_branch(idx) is None

    def test_single_release_branch_returned(self):
        data = {**INDEX_DATA, "branches": {
            "release/0.2.0": INDEX_DATA["branches"]["release/0.2.0"],
        }}
        idx = parse_index(data)
        assert latest_release_branch(idx).branch == "release/0.2.0"

    def test_build_key_ordering_is_lexicographic(self):
        # build_keys are YYYYMMDD-HHMMSS-sha, so lex order == chrono order
        data = {**INDEX_DATA, "branches": {
            "release/0.1.0": {"build_key": "20260101-000000-aaa0001",
                               "hex_url": "x", "manifest_url": "y"},
            "release/0.3.0": {"build_key": "20261231-235959-zzz9999",
                               "hex_url": "x", "manifest_url": "y"},
            "release/0.2.0": {"build_key": "20260601-120000-mmm5555",
                               "hex_url": "x", "manifest_url": "y"},
        }}
        idx = parse_index(data)
        assert latest_release_branch(idx).branch == "release/0.3.0"
