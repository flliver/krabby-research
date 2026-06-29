"""Guard for firmware/arduino/command.h parseCommands().

Regression: the parser treated running out of tokens as an error and returned 0,
clearing the whole batch. That silently dropped every command that didn't fill the
buffer exactly — i.e. every single-joint position target and every FULL-only subset
the HAL routes to closed-loop position — so `setTarget` never ran and no joint drove.
End-of-input must instead return the pairs parsed so far."""
import re
from pathlib import Path

ARDUINO = Path(__file__).resolve().parents[3] / "firmware" / "arduino"


def _parse_body() -> str:
    text = (ARDUINO / "command.h").read_text()
    return text[text.index("size_t parseCommands"):]


def test_end_of_input_breaks_not_zero():
    body = _parse_body()
    # a missing name (clean end of input) must break out of the loop, not return 0
    assert re.search(r"name\.length\(\)\s*==\s*0\s*\)\s*\n\s*break", body), \
        "exhausted tokens must break and return the parsed count, not be an error"


def test_dangling_name_is_still_malformed():
    body = _parse_body()
    # a name with no value is still a malformed batch -> clear + return 0
    assert "valStr.length() == 0" in body and "return 0" in body


def test_returns_idx_after_loop():
    body = _parse_body()
    # function ends by returning the count parsed
    assert re.search(r"return\s+idx\s*;", body)
