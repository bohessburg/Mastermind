"""Torch-free self-play work descriptors shared with spawned workers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelfPlaySegment:
    """A contiguous pool work item for NN or scripted-opponent self-play."""

    n_games: int
    seat0_model_id: int
    seat1_model_id: int
    scripted_kind: str | None = None
    nn_player: int = 0
    # ``None`` means full random kingdom pool (or the configured fixed
    # kingdom); a non-empty list contains native DefIds for this segment.
    kingdom_pool: list[int] | None = None
    # ``None`` preserves SelfPlayConfig.kingdom_mode. Curriculum segments set
    # this explicitly so a random phase can override a fixed base campaign.
    kingdom_mode: str | None = None
    # Zero uses SelfPlayConfig.sims_per_move; a positive value creates a
    # higher-budget runner for this segment only.
    sims_override: int = 0
    # Checkpoint basename for metrics and strength-matched league sampling.
    # It is set only on true two-model league segments.
    league_opponent: str | None = None
    # Assigned after the final segment plan is composed. It identifies the
    # first global game covered by this contiguous segment and survives worker
    # splitting, so a slot's seed never depends on which worker owns it.
    game_index: int | None = None

    @property
    def is_league(self) -> bool:
        return not self.is_scripted and self.seat0_model_id != self.seat1_model_id

    @property
    def is_scripted(self) -> bool:
        return self.scripted_kind is not None

    @property
    def is_normal_mirror(self) -> bool:
        """True only for ordinary current-best-versus-current-best games."""
        return not self.is_scripted and self.seat0_model_id == self.seat1_model_id == 0
