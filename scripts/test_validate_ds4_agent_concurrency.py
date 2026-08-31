#!/usr/bin/env python3

"""Unit tests for DS4 concurrent-response integrity classification."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType


def load_validator() -> ModuleType:
    path = Path(__file__).with_name("validate-ds4-agent-concurrency.py")
    spec = importlib.util.spec_from_file_location("ds4_agent_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load validator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_validator()


def result_for(content: str, reasoning: str = "") -> dict[str, object]:
    return {
        "content_prefix": content[:1000],
        "tool_call_delta_count": 0,
        "content_indicators": VALIDATOR.count_text_indicators(content, "BETA-731"),
        "reasoning_indicators": VALIDATOR.count_text_indicators(reasoning, "BETA-731"),
    }


class TextIntegrityTest(unittest.TestCase):
    def test_clean_english_with_unicode_punctuation_passes(self) -> None:
        result = result_for(
            "ALPHA-731 REPORT\nThe request is isolated \u2014 caf\u00e9 identifiers remain "
            "attached to their owning stream."
        )

        self.assertEqual(VALIDATOR.integrity_violations(result), [])

    def test_short_cjk_reference_does_not_trigger_burst_watchdog(self) -> None:
        result = result_for("ALPHA-731 REPORT\nThe quoted city name is \u5317\u4eac.")

        self.assertEqual(VALIDATOR.integrity_violations(result), [])

    def test_contiguous_cjk_burst_is_rejected(self) -> None:
        result = result_for(
            "ALPHA-731 REPORT\n\u8bf7\u68c0\u67e5\u8fd9\u4e2a\u8f93\u51fa"
        )

        self.assertIn("content.max_cjk_run=7", VALIDATOR.integrity_violations(result))

    def test_scattered_cjk_volume_is_rejected(self) -> None:
        text = "ALPHA-731 REPORT\n" + " word \u4e2d" * 20
        result = result_for(text)

        violations = VALIDATOR.integrity_violations(result)
        self.assertTrue(
            any(item.startswith("content.cjk_fraction=") for item in violations)
        )

    def test_raw_token_marker_is_rejected(self) -> None:
        result = result_for(
            "ALPHA-731 REPORT\nUnexpected <|tool_calls_section_begin|>."
        )

        self.assertTrue(
            any(
                "content.raw_token" in item
                for item in VALIDATOR.integrity_violations(result)
            )
        )

    def test_forbidden_request_identity_is_rejected(self) -> None:
        result = result_for("ALPHA-731 REPORT\nState from BETA-731 was observed.")

        self.assertIn(
            "content.forbidden_marker_count=1",
            VALIDATOR.integrity_violations(result),
        )


if __name__ == "__main__":
    unittest.main()
