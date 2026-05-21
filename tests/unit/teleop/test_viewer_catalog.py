"""Parsing of browser ``catalog_ids`` in teleop signaling."""

from __future__ import annotations

from teleop.edge.viewer_catalog import parse_viewer_catalog_ids_from_payload


def test_missing_key_means_no_update() -> None:
    assert parse_viewer_catalog_ids_from_payload({"type": "hello"}, max_lines=8) is None


def test_explicit_empty_array_means_revert_sentinel() -> None:
    assert parse_viewer_catalog_ids_from_payload({"catalog_ids": []}, max_lines=8) == []


def test_strings_trimmed_and_capped() -> None:
    got = parse_viewer_catalog_ids_from_payload(
        {"catalog_ids": ["  a ", "b", "c", "d"]},
        max_lines=2,
    )
    assert got == ["a", "b"]


def test_duplicate_ids_preserve_first_only() -> None:
    got = parse_viewer_catalog_ids_from_payload(
        {"catalog_ids": ["front_rgbd", "side_rgbd", "front_rgbd"]},
        max_lines=8,
    )
    assert got == ["front_rgbd", "side_rgbd"]


def test_non_list_returns_none() -> None:
    assert parse_viewer_catalog_ids_from_payload({"catalog_ids": "zed"}, max_lines=8) is None


def test_null_value_returns_none() -> None:
    assert parse_viewer_catalog_ids_from_payload({"catalog_ids": None}, max_lines=8) is None
