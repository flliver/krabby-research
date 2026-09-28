"""The stroke/posture knobs (stride/posture plan, step 3, 2026-09-25), without Isaac.

Every knob is DEFAULT OFF: unset or blank writes NOTHING into the cfg, so a knob-off config dump is
byte-identical to one built before this module existed (the ``motion_knobs``/``clock_knobs``
contract). Covered here:

* the pure knob module (``config/crab_hex/stroke_knobs.py``): module purity, parsers, the applier
  on a stand-in cfg (func swap, added term, the split flag), every refusal, and its re-export
  through ``omni_knobs`` -- including ``KRABBY_OMNI_STROKE_CAT_W`` (per-category stroke plan,
  2026-09-27): grammar, the mirror-tie and composition (needs a non-zero STROKE_W) refusals, and
  the ``cat_scale`` write onto the SAME term ``STROKE_W`` swapped in;
* :func:`crab_hex_omni_stroke_rewards._assert_foot_order` / ``_resolved_body_names`` /
  ``_resolve_cat_scale``, lifted out of the Isaac-importing wrapper module with ``ast`` and exec'd
  against plain stand-ins (``_resolve_cat_scale`` also needs a bare ``import torch``, which needs
  no Isaac/GPU to build a CPU tensor);
* a static text check that neither the forward nor the sideways task's config modules import
  ``stroke_knobs`` (the terms it names, ``RewardFootStroke`` / ``RewardBodyCentring``, do not exist
  on their rewards cfgs).

The driver's wiring (``knob_parsers``, ``OVERLAY_DUMP_CHECKS``, ``RULE_V2_BLOCKS``, and the new
``require_env`` candidates.yaml key) is pinned in ``test_crab_hex_omni_stroke_ladder.py``.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"
OMNI = TASKS / "crab_hex_omni_task"
STROKE_KNOBS_PY = OMNI / "config/crab_hex/stroke_knobs.py"
OMNI_KNOBS_PY = OMNI / "config/crab_hex/omni_knobs.py"
STROKE_REWARDS_PY = OMNI / "mdp/crab_hex_omni_stroke_rewards.py"
FORWARD_CFG = TASKS / "crab_hex_forward_task"
SIDEWAYS_CFG = TASKS / "crab_hex_sideways_task"

ALL_KNOBS = ("KRABBY_OMNI_STROKE_W", "KRABBY_OMNI_CENTRING_W", "KRABBY_OMNI_REWARD_SPLIT",
             "KRABBY_OMNI_STROKE_CAT_W")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _source_of(path: Path, name: str) -> str:
    src = path.read_text()
    tree = ast.parse(src, str(path))
    node = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name][0]
    return textwrap.dedent(" " * node.col_offset + ast.get_source_segment(src, node))


sk = _load(STROKE_KNOBS_PY, "stroke_knobs__test")


# ================================================================= the knob module (pure)
def test_stroke_knobs_module_is_pure():
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('sk', {str(STROKE_KNOBS_PY)!r})\n"
        "mod = importlib.util.module_from_spec(spec); sys.modules['sk'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "assert mod.parse_stroke_w('0.02') == 0.02\n"
        "print(sorted(m for m in sys.modules if m.startswith(('parkour_tasks', 'isaaclab', 'torch'))))\n"
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_variable_names_are_the_contract():
    got = {getattr(sk, a) for a in dir(sk) if a.endswith("_VAR")}
    assert set(ALL_KNOBS) <= got
    assert set(sk.STROKE_KNOB_PARSERS) == set(ALL_KNOBS)
    assert sk.STROKE_TERM == "reward_stride_length"       # the OLD name -- the whole point of the swap
    assert sk.CENTRING_TERM == "reward_body_centring"
    assert sk.EPS_W == 0.001


def test_parse_stroke_w_and_centring_w():
    assert sk.parse_stroke_w("") is None and sk.parse_stroke_w(None) is None and sk.parse_stroke_w("  ") is None
    assert sk.parse_stroke_w("0") == 0.0
    assert sk.parse_stroke_w("0.02") == 0.02
    for bad in ("-0.01", "-1"):
        with pytest.raises(ValueError, match="KRABBY_OMNI_STROKE_W"):
            sk.parse_stroke_w(bad)
    for bad in ("0.0001", "0.0009999"):
        with pytest.raises(ValueError, match="epsilon floor"):
            sk.parse_stroke_w(bad)
    assert sk.parse_stroke_w("0.001") == 0.001            # exactly the floor is fine
    assert sk.parse_centring_w("") is None
    assert sk.parse_centring_w("0") == 0.0
    with pytest.raises(ValueError, match="KRABBY_OMNI_CENTRING_W"):
        sk.parse_centring_w("-0.5")
    with pytest.raises(ValueError, match="epsilon floor"):
        sk.parse_centring_w("0.0005")


def test_parse_reward_split():
    assert sk.parse_reward_split("") is False and sk.parse_reward_split(None) is False
    assert sk.parse_reward_split("0") is False
    assert sk.parse_reward_split("1") is True
    for bad in ("2", "true", "on"):
        with pytest.raises(ValueError, match="KRABBY_OMNI_REWARD_SPLIT"):
            sk.parse_reward_split(bad)


def test_parse_stroke_cat_w_grammar():
    assert sk.parse_stroke_cat_w("") is None
    assert sk.parse_stroke_cat_w(None) is None
    assert sk.parse_stroke_cat_w("  ") is None
    assert sk.parse_stroke_cat_w("side:4.8,diag:1.2,fwd:1,back:1") == {
        "fwd": 1.0, "back": 1.0, "side": 4.8, "diag": 1.2, "turn": 1.0, "move_turn": 1.0,
    }
    # unlisted keys default to 1.0, so 'fwd:2,back:2' alone is a legal, self-describing table
    assert sk.parse_stroke_cat_w("fwd:2,back:2") == {
        "fwd": 2.0, "back": 2.0, "side": 1.0, "diag": 1.0, "turn": 1.0, "move_turn": 1.0,
    }
    assert sk.parse_stroke_cat_w("side:0")["side"] == 0.0     # 0 switches the category off
    # no epsilon floor here (unlike KRABBY_OMNI_STROKE_W): the manager weight stays w_S
    assert sk.parse_stroke_cat_w("side:0.0001")["side"] == 0.0001


def test_parse_stroke_cat_w_refusals():
    with pytest.raises(ValueError, match="unknown category"):
        sk.parse_stroke_cat_w("front:2")
    with pytest.raises(ValueError, match="duplicate key"):
        sk.parse_stroke_cat_w("side:2,side:3")
    for bad in (",", "side:2,", "side:2,,diag:1"):
        with pytest.raises(ValueError, match="non-empty"):
            sk.parse_stroke_cat_w(bad)
    with pytest.raises(ValueError, match="non-negative"):
        sk.parse_stroke_cat_w("side:-1")
    for bad in ("side:nan", "side:inf", "side:-inf"):
        with pytest.raises(ValueError, match="finite"):
            sk.parse_stroke_cat_w(bad)
    with pytest.raises(ValueError, match="'stand'"):
        sk.parse_stroke_cat_w("stand:2")
    with pytest.raises(ValueError, match="fwd == back"):
        sk.parse_stroke_cat_w("fwd:2")                # back defaults to 1.0 != 2.0
    with pytest.raises(ValueError, match="fwd == back"):
        sk.parse_stroke_cat_w("fwd:2,back:1")
    assert sk.parse_stroke_cat_w("fwd:2,back:2") is not None      # the tie satisfied -> accepted


def test_phaseout_names_is_a_light_grammar_check():
    assert sk._phaseout_names("reward_stride_length:0.5:0.001:0:2400", "reward_stride_length")
    assert sk._phaseout_names("reward_clock_schedule:1:0.001:0:2400,reward_stride_length:0.5:0.001:0:2400",
                              "reward_stride_length")
    assert not sk._phaseout_names("reward_clock_schedule:1:0.001:0:2400", "reward_stride_length")
    assert not sk._phaseout_names("", "reward_stride_length")
    assert not sk._phaseout_names(None, "reward_stride_length")


# ------------------------------------------------------------------- stand-in env cfg
class _Bag:
    pass


class _Term:
    def __init__(self, func=None, weight=0.0, params=None):
        self.func, self.weight, self.params = func, weight, params


class _StockStride:
    """Stands in for ``parkour_isaaclab.envs.mdp.rewards.RewardStrideLength``."""


class _OtherFunc:
    """Some other func already owning the term -- the clobber guard's target."""


def _cfg(with_stride=True, stride_func=None, with_centring=False):
    cfg = _Bag()
    cfg.rewards = _Bag()
    if with_stride:
        cfg.rewards.reward_stride_length = _Term(
            func=_StockStride if stride_func is None else stride_func,
            weight=0.5, params={"power": 2.0})
    if with_centring:
        cfg.rewards.reward_body_centring = _Term(func="already-there", weight=0.1, params={})
    return cfg


def _snap(obj):
    if isinstance(obj, (_Bag, _Term)):
        return {k: _snap(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _snap(v) for k, v in obj.items()}
    return obj


BUILDERS = {sk.STROKE_TERM: ("func:stroke", {"p": "stroke"}),
            sk.CENTRING_TERM: ("func:centring", {"p": "centring"})}


def _apply(cfg, env, **kw):
    """The PUBLIC entry point -- only used where no armed knob needs a func-identity check (an
    armed ``KRABBY_OMNI_STROKE_W`` would otherwise reach for the REAL Isaac ``RewardStrideLength``;
    those cases call :func:`_apply_stroke` below, the ``_apply_ang_turn_only_knob`` precedent of
    calling the private applier directly with a stand-in)."""
    sk.apply_stroke_knobs(cfg, env, term_cls=_Term, builders=BUILDERS, **kw)


def _apply_stroke(cfg, env, weight, **kw):
    """``_apply_stroke_w_knob`` directly, with a stand-in ``stock_func`` -- keeps this file
    Isaac-free while still exercising every guard the real applier runs."""
    kw.setdefault("stock_func", _StockStride)
    sk._apply_stroke_w_knob(cfg, env, weight, term_cls=_Term, builders=BUILDERS, **kw)


def test_every_knob_unset_blank_or_off_writes_nothing():
    pristine = _snap(_cfg())
    for env in ({}, {k: "" for k in ALL_KNOBS}, {k: "  " for k in ALL_KNOBS},
                {"KRABBY_OMNI_REWARD_SPLIT": "0"}):
        cfg = _cfg()
        _apply(cfg, env)
        assert _snap(cfg) == pristine, env
        assert not hasattr(cfg, "reward_category_split")


def test_stroke_w_swaps_func_params_and_weight_together():
    cfg = _cfg()
    _apply_stroke(cfg, {}, 0.02)
    term = cfg.rewards.reward_stride_length
    assert (term.func, term.weight, term.params) == ("func:stroke", 0.02, {"p": "stroke"})
    # 0.0 is a legitimate ablation: the swap still happens, at weight 0
    cfg2 = _cfg()
    _apply_stroke(cfg2, {}, 0.0)
    assert cfg2.rewards.reward_stride_length.weight == 0.0
    assert cfg2.rewards.reward_stride_length.func == "func:stroke"


def test_stroke_w_refuses_alongside_krabby_stride_w():
    with pytest.raises(RuntimeError, match="KRABBY_STRIDE_W"):
        _apply_stroke(_cfg(), {"KRABBY_STRIDE_W": "0.5"}, 0.02)
    # blank KRABBY_STRIDE_W is not "also set"
    cfg = _cfg()
    _apply_stroke(cfg, {"KRABBY_STRIDE_W": ""}, 0.02)
    assert cfg.rewards.reward_stride_length.func == "func:stroke"


def test_stroke_w_refuses_a_phaseout_naming_the_term():
    with pytest.raises(RuntimeError, match="KRABBY_PHASEOUT"):
        _apply_stroke(_cfg(), {"KRABBY_PHASEOUT": "reward_stride_length:0.5:0.001:0:2400"}, 0.02)
    # a phaseout naming some OTHER term does not block it
    cfg = _cfg()
    _apply_stroke(cfg, {"KRABBY_PHASEOUT": "reward_clock_schedule:1:0.001:0:2400"}, 0.02)
    assert cfg.rewards.reward_stride_length.func == "func:stroke"


def test_stroke_w_refuses_a_cfg_without_the_term_or_already_swapped():
    no_term = _Bag()
    no_term.rewards = _Bag()
    with pytest.raises(RuntimeError, match="reward_stride_length"):
        _apply_stroke(no_term, {}, 0.02)
    already = _cfg(stride_func=_OtherFunc)
    with pytest.raises(RuntimeError, match="refuses to clobber"):
        _apply_stroke(already, {}, 0.02)


def test_stroke_w_injectable_stock_func_matches_the_real_default():
    """Without an injected ``stock_func``, the applier reaches for the real
    ``parkour_isaaclab.envs.mdp.rewards.RewardStrideLength`` -- checked by source, since importing
    it needs Isaac."""
    src = _source_of(STROKE_KNOBS_PY, "_apply_stroke_w_knob")
    assert "from parkour_isaaclab.envs.mdp.rewards import RewardStrideLength as stock_func" in src


def test_apply_stroke_knobs_applies_cat_w_right_after_stroke_w():
    src = STROKE_KNOBS_PY.read_text()
    body = src[src.index("def apply_stroke_knobs"):]
    assert (body.index("_apply_stroke_w_knob(") < body.index("_apply_stroke_cat_w_knob(")
            < body.index("_apply_centring_w_knob("))


def test_stroke_cat_w_requires_armed_stroke_w():
    """The composition refusal (``no RewardFootStroke to scale``): unset/blank/0.0 STROKE_W all
    refuse CAT_W. Through the PUBLIC entry point for the unset/blank cases (stroke_w stays ``None``,
    so ``_apply_stroke_w_knob`` -- which would otherwise reach for the real Isaac stock func -- is
    never called); directly against the applier for ``0.0`` (a legitimate ablation that WOULD reach
    it, so it is exercised with the parsed value alone, the same way ``_apply_stroke`` sidesteps it).
    """
    cat_w_spec = "side:2,back:1,fwd:1"
    with pytest.raises(RuntimeError, match="KRABBY_OMNI_STROKE_CAT_W"):
        _apply(_cfg(), {"KRABBY_OMNI_STROKE_CAT_W": cat_w_spec})                          # unset
    with pytest.raises(RuntimeError, match="KRABBY_OMNI_STROKE_CAT_W"):
        _apply(_cfg(), {"KRABBY_OMNI_STROKE_W": "", "KRABBY_OMNI_STROKE_CAT_W": cat_w_spec})  # blank
    table = sk.parse_stroke_cat_w(cat_w_spec)
    with pytest.raises(RuntimeError, match="KRABBY_OMNI_STROKE_CAT_W"):
        sk._apply_stroke_cat_w_knob(_cfg(), None, table)
    with pytest.raises(RuntimeError, match="KRABBY_OMNI_STROKE_CAT_W"):
        sk._apply_stroke_cat_w_knob(_cfg(), 0.0, table)                                   # exactly 0.0


def test_stroke_cat_w_writes_cat_scale_onto_the_swapped_term_and_leaves_weight_func_alone():
    cfg = _cfg()
    _apply_stroke(cfg, {}, 0.117)          # arms STROKE_W (stand-in stock func, per _apply_stroke)
    table = sk.parse_stroke_cat_w("side:4.8,diag:1.2,fwd:1,back:1")
    sk._apply_stroke_cat_w_knob(cfg, 0.117, table)
    term = cfg.rewards.reward_stride_length
    assert term.func == "func:stroke" and term.weight == 0.117       # untouched by the cat_w knob
    assert term.params == {"p": "stroke", "cat_scale": {
        "fwd": 1.0, "back": 1.0, "side": 4.8, "diag": 1.2, "turn": 1.0, "move_turn": 1.0}}


def test_stroke_cat_w_refuses_a_cfg_without_the_term():
    no_term = _Bag()
    no_term.rewards = _Bag()
    with pytest.raises(RuntimeError, match="reward_stride_length"):
        sk._apply_stroke_cat_w_knob(no_term, 0.117, sk.parse_stroke_cat_w("side:2,back:1,fwd:1"))


def test_centring_w_adds_the_term_and_refuses_a_double_add():
    cfg = _cfg()
    _apply(cfg, {"KRABBY_OMNI_CENTRING_W": "0.01"})
    term = cfg.rewards.reward_body_centring
    assert (term.func, term.weight, term.params) == ("func:centring", 0.01, {"p": "centring"})
    assert cfg.rewards.reward_stride_length.func is _StockStride       # untouched
    already = _cfg(with_centring=True)
    with pytest.raises(RuntimeError, match="already carries"):
        _apply(already, {"KRABBY_OMNI_CENTRING_W": "0.01"})


def test_centring_w_refusals():
    for bad in ("-0.01", "-1"):
        with pytest.raises(ValueError, match="KRABBY_OMNI_CENTRING_W"):
            _apply(_cfg(), {"KRABBY_OMNI_CENTRING_W": bad})
    with pytest.raises(ValueError, match="epsilon floor"):
        _apply(_cfg(), {"KRABBY_OMNI_CENTRING_W": "0.0005"})


def test_reward_split_flag():
    cfg = _cfg()
    _apply(cfg, {"KRABBY_OMNI_REWARD_SPLIT": "1"})
    assert cfg.reward_category_split is True
    cfg2 = _cfg()
    _apply(cfg2, {"KRABBY_OMNI_REWARD_SPLIT": "0"})
    assert not hasattr(cfg2, "reward_category_split")
    with pytest.raises(ValueError, match="KRABBY_OMNI_REWARD_SPLIT"):
        _apply(_cfg(), {"KRABBY_OMNI_REWARD_SPLIT": "2"})


def test_the_real_term_builders_name_real_classes_and_the_six_foot_bodies():
    """The Isaac-side builder table, checked by SOURCE (building it needs Isaac): both terms point
    at real classes in the wrapper module, with a six-foot, order-preserving SceneEntityCfg."""
    src = STROKE_KNOBS_PY.read_text()
    assert sk._REWARDS_MOD.endswith("crab_hex_omni_task.mdp.crab_hex_omni_stroke_rewards")
    for cls in ("RewardFootStroke", "RewardBodyCentring"):
        assert f"rew.{cls}" in src
    assert "_CRAB_FOOT_BODY_NAMES" in src and "preserve_order=True" in src


# ================================================================= re-export through omni_knobs
def test_omni_knobs_re_exports_the_stroke_variables_and_parsers():
    ok = _load(OMNI_KNOBS_PY, "omni_knobs__stroke_test")
    for var in ALL_KNOBS:
        assert getattr(ok, {"KRABBY_OMNI_STROKE_W": "STROKE_W_VAR",
                            "KRABBY_OMNI_CENTRING_W": "CENTRING_W_VAR",
                            "KRABBY_OMNI_REWARD_SPLIT": "REWARD_SPLIT_VAR",
                            "KRABBY_OMNI_STROKE_CAT_W": "STROKE_CAT_W_VAR"}[var]) == var
    assert ok.parse_stroke_w("0.02") == 0.02
    assert ok.parse_centring_w("0.01") == 0.01
    assert ok.parse_reward_split("1") is True
    assert ok.parse_stroke_cat_w("side:2,back:1,fwd:1")["side"] == 2.0
    assert set(ok.STROKE_KNOB_PARSERS) == set(ALL_KNOBS)
    for name in ("STROKE_W_VAR", "CENTRING_W_VAR", "REWARD_SPLIT_VAR", "STROKE_TERM", "CENTRING_TERM",
                "parse_stroke_w", "parse_centring_w", "parse_reward_split", "STROKE_KNOB_PARSERS",
                "STROKE_CAT_W_VAR", "parse_stroke_cat_w"):
        assert name in ok.__all__


def test_apply_omni_knobs_calls_apply_stroke_knobs_last():
    src = OMNI_KNOBS_PY.read_text()
    body = src[src.index("def apply_omni_knobs"):]
    assert body.index("_mk.apply_motion_knobs(cfg, env)") < body.index("_sk.apply_stroke_knobs(cfg, env)")
    # nothing in apply_omni_knobs runs after it
    tail = body[body.index("_sk.apply_stroke_knobs(cfg, env)"):]
    assert tail.strip().splitlines()[0].strip() == "_sk.apply_stroke_knobs(cfg, env)"


# ================================================================= the Isaac wrapper (ast-lifted)
def _exec_pure(names: tuple[str, ...], *, imports: str = "") -> dict:
    """Lift the given top-level defs out of the Isaac-importing wrapper module and exec them
    against nothing but ``__future__`` annotations (plus ``imports``, for a def that needs a plain
    library -- ``torch`` needs no Isaac/GPU to build a CPU tensor). Neither function otherwise
    touches an isaaclab type by name in its body, only duck-typed ``.body_names`` / ``.body_ids``
    attributes."""
    src = "from __future__ import annotations\n\n" + imports + "\n\n".join(
        _source_of(STROKE_REWARDS_PY, n) for n in names)
    ns: dict = {}
    exec(compile(src, str(STROKE_REWARDS_PY), "exec"), ns)  # noqa: S102
    return ns


_wrap = _exec_pure(("_resolved_body_names", "_assert_foot_order"))
FOOT_ORDER = ("FL", "FR", "ML", "MR", "RL", "RR")
_wrap["FOOT_ORDER"] = FOOT_ORDER

_cat = _exec_pure(("_resolve_cat_scale",), imports="import torch\n\n")
TELEMETRY_CATEGORIES = ("stand", "fwd", "side", "back", "diag", "turn", "move_turn")


class _Entity:
    def __init__(self, body_names):
        self.body_names = body_names


def test_resolved_body_names_reads_a_list_or_a_slice():
    e = _Entity(["a", "b", "c", "d"])
    assert _wrap["_resolved_body_names"](e, [2, 0]) == ["c", "a"]
    assert _wrap["_resolved_body_names"](e, slice(None)) == ["a", "b", "c", "d"]


class _CfgIds:
    def __init__(self, ids):
        self.body_ids = ids


def test_assert_foot_order_passes_when_the_six_names_match_element_for_element():
    names = list(FOOT_ORDER)
    asset = _Entity([f"{leg}_Footpad" for leg in names] + ["Chassis"])
    sensor = _Entity([f"{leg}_Footpad" for leg in names])
    _wrap["_assert_foot_order"](asset, _CfgIds(list(range(6))), sensor, _CfgIds(list(range(6))))


def test_assert_foot_order_refuses_a_mismatched_or_short_pairing():
    asset = _Entity([f"{leg}_Footpad" for leg in FOOT_ORDER])
    sensor_reordered = _Entity(["FR_Footpad", "FL_Footpad", "ML_Footpad", "MR_Footpad", "RL_Footpad",
                               "RR_Footpad"])
    with pytest.raises(RuntimeError, match="must match element for element"):
        _wrap["_assert_foot_order"](asset, _CfgIds(list(range(6))), sensor_reordered, _CfgIds(list(range(6))))
    sensor_short = _Entity([f"{leg}_Footpad" for leg in FOOT_ORDER[:5]])
    with pytest.raises(RuntimeError, match="must match element for element"):
        _wrap["_assert_foot_order"](asset, _CfgIds(list(range(6))), sensor_short, _CfgIds(list(range(5))))
    # both sides agree with each other but NOT with FOOT_ORDER (the per-foot pay's index)
    swapped = ["FR_Footpad", "FL_Footpad", "ML_Footpad", "MR_Footpad", "RL_Footpad", "RR_Footpad"]
    with pytest.raises(RuntimeError, match="must match element for element"):
        _wrap["_assert_foot_order"](_Entity(swapped), _CfgIds(list(range(6))), _Entity(swapped),
                                    _CfgIds(list(range(6))))


# ============================================== _resolve_cat_scale (ast-lifted, pure torch)
def test_resolve_cat_scale_defaults_unlisted_names_to_one_including_stand():
    vec = _cat["_resolve_cat_scale"]({"side": 4.8, "diag": 1.2, "fwd": 1.0, "back": 1.0},
                                     TELEMETRY_CATEGORIES)
    got = {name: pytest.approx(vec[i].item()) for i, name in enumerate(TELEMETRY_CATEGORIES)}
    assert got == {"stand": 1.0, "fwd": 1.0, "back": 1.0, "side": 4.8, "diag": 1.2,
                   "turn": 1.0, "move_turn": 1.0}


def test_resolve_cat_scale_all_ones_is_the_identity_vector():
    vec = _cat["_resolve_cat_scale"]({}, TELEMETRY_CATEGORIES)
    assert (vec == 1.0).all()


def test_resolve_cat_scale_refuses_unknown_names():
    with pytest.raises(RuntimeError, match="unknown categor"):
        _cat["_resolve_cat_scale"]({"sideways": 2.0}, TELEMETRY_CATEGORIES)


def test_resolve_cat_scale_refuses_a_stand_key():
    """``stand`` IS one of ``TELEMETRY_CATEGORIES``, so the unknown-name check alone would let it
    through and silently resolve into a multiplier on a term that pays exactly 0 on a stand step --
    this needs its own refusal, mirroring ``stroke_knobs.parse_stroke_cat_w``'s."""
    with pytest.raises(RuntimeError, match="refuses the key 'stand'"):
        _cat["_resolve_cat_scale"]({"stand": 2.0}, TELEMETRY_CATEGORIES)
    with pytest.raises(RuntimeError, match="refuses the key 'stand'"):
        _cat["_resolve_cat_scale"]({"stand": 1.0, "side": 4.8, "fwd": 1.0, "back": 1.0}, TELEMETRY_CATEGORIES)


def test_resolve_cat_scale_refuses_fwd_back_asymmetry():
    """Defence in depth: the knob that builds ``cat_scale`` already refuses this, so only a
    hand-edited cfg (a stand-in dict here, mimicking one) reaches this check."""
    with pytest.raises(RuntimeError, match="fwd == back"):
        _cat["_resolve_cat_scale"]({"fwd": 2.0, "back": 1.0}, TELEMETRY_CATEGORIES)
    _cat["_resolve_cat_scale"]({"fwd": 2.0, "back": 2.0}, TELEMETRY_CATEGORIES)  # tie satisfied


# ================================================================= forward/sideways isolation
def test_forward_and_sideways_cfg_modules_never_import_stroke_knobs():
    for root in (FORWARD_CFG, SIDEWAYS_CFG):
        for path in root.rglob("*.py"):
            text = path.read_text(errors="ignore")
            assert "stroke_knobs" not in text, path
            assert "crab_hex_omni_stroke_rewards" not in text, path
