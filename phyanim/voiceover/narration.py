"""Narration player for engine-mode voiceover.

When a PhysicsAnimation is rendered in engine mode with a narration list,
the NarrationPlayer replaces the single linear ``self.play(tracker.animate...)``
call with a segmented playback that interleaves voiceover narration with
tracker advancement.

Each narration segment specifies:
- ``text``: the narration text to speak.
- ``at``: the render-time second at which the narration begins.

Between narration segments (gaps), the tracker advances without voiceover.
During a narration segment, the tracker advances while voiceover plays, and
if the voiceover duration exceeds the segment's time span, the scene waits
for the voiceover to finish.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from manim import ValueTracker


@dataclass
class NarrationSegment:
    """A single narration cue for engine mode.

    Parameters
    ----------
    text:
        The narration text to be spoken by TTS.
    at:
        The render-time second at which this narration begins.
    subtitle:
        Optional subtitle text (defaults to *text*).
    """

    text: str
    at: float
    subtitle: str | None = None


class NarrationPlayer:
    """Plays narration segments synchronized with a ValueTracker.

    The player walks through the narration segments in order. For each
    segment it:
    1. Advances the tracker from the previous position to ``seg.at``
       (the narration start time).
    2. Opens a voiceover context for ``seg.text``.
    3. Advances the tracker during the voiceover by the minimum of the
       time remaining before the next narration and the voiceover duration.
    4. If the voiceover is longer than the remaining time, waits for it
       to finish.
    """

    def __init__(
        self,
        narration: list[dict | NarrationSegment],
        total_time: float,
    ) -> None:
        self.segments: list[NarrationSegment] = []
        for item in narration:
            if isinstance(item, dict):
                self.segments.append(
                    NarrationSegment(
                        text=item["text"],
                        at=float(item["at"]),
                        subtitle=item.get("subtitle"),
                    )
                )
            else:
                self.segments.append(item)
        self.total_time = total_time

    def play(
        self,
        scene: Any,
        tracker: ValueTracker,
    ) -> None:
        """Play the narration segments, advancing *tracker* from 0 to total_time."""
        from manim import linear

        if not self.segments:
            # No narration — single linear play.
            current = tracker.get_value()
            remaining = self.total_time - current
            if remaining > 0:
                scene.play(
                    tracker.animate.set_value(self.total_time),
                    run_time=remaining,
                    rate_func=linear,
                )
            return

        current_t = tracker.get_value()

        for i, seg in enumerate(self.segments):
            # 1. Gap before this narration: advance tracker to seg.at.
            if seg.at > current_t:
                gap = seg.at - current_t
                scene.play(
                    tracker.animate.set_value(seg.at),
                    run_time=gap,
                    rate_func=linear,
                )
                current_t = seg.at

            # 2. Determine how far the tracker should advance during voiceover.
            if i + 1 < len(self.segments):
                next_at = self.segments[i + 1].at
            else:
                next_at = self.total_time
            available = next_at - current_t

            # 3. Open voiceover and advance the tracker.
            with scene.voiceover(text=seg.text) as vo:
                vo_duration = vo.duration
                if available > 0:
                    advance = min(available, vo_duration)
                    if advance > 0:
                        scene.play(
                            tracker.animate.set_value(current_t + advance),
                            run_time=advance,
                            rate_func=linear,
                        )
                    current_t += advance
                    remaining_vo = vo_duration - advance
                else:
                    remaining_vo = vo_duration

                # 4. If voiceover is longer, wait for it to finish.
                if remaining_vo > 0:
                    scene.wait(remaining_vo)

        # 5. Gap after the last narration.
        if self.total_time > current_t:
            remaining = self.total_time - current_t
            scene.play(
                tracker.animate.set_value(self.total_time),
                run_time=remaining,
                rate_func=linear,
            )
