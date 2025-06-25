#
#
#
#
#
#     assert info["bars_since_last"] == bars
#     assert math.isclose(info["score"], score)
#     assert not triggered
#     assert not triggered
#     assert triggered is True
#     bars = strategy.FORCED_ENTRY_BAR_THRESHOLD
#     bars = strategy.FORCED_ENTRY_BAR_THRESHOLD
#     bars = strategy.FORCED_ENTRY_BAR_THRESHOLD - 1
#     score = strategy.FORCED_ENTRY_MIN_SIGNAL_SCORE + 0.1
#     score = strategy.FORCED_ENTRY_MIN_SIGNAL_SCORE + 0.5
#     score = strategy.FORCED_ENTRY_MIN_SIGNAL_SCORE - 0.1
#     triggered, _ = strategy.check_forced_trigger(bars, score)
#     triggered, _ = strategy.check_forced_trigger(bars, score)
#     triggered, info = strategy.check_forced_trigger(bars, score)
# def test_check_forced_trigger_false_bars():
# def test_check_forced_trigger_false_score():
# def test_check_forced_trigger_true():
# from src import strategy
# import math