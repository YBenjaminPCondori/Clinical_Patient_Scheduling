"""Gymnasium-compatible version of the Part 1b hospital bed environment.

This mirrors the old three-action hospital bed allocation setting:
allocate ICU, allocate general bed, or delay the patient.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


PATIENT_TYPES = {
    0: "emergency",
    1: "urgent",
    2: "elective",
}

ACTION_MEANINGS = {
    0: "allocate_icu",
    1: "allocate_general",
    2: "delay_patient",
}

ICU_MAX = 2
GENERAL_MAX = 3
MAX_STEPS = 500


class OldHospitalBedPPOEnv(gym.Env):
    """Part 1b HospitalEnv adapted to the Gymnasium reset/step API."""

    metadata = {"name": "OldHospitalBedPPOEnv"}

    def __init__(self, max_steps: int = MAX_STEPS) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.asarray([0, 0, 0], dtype=np.float32),
            high=np.asarray([ICU_MAX, GENERAL_MAX, 2], dtype=np.float32),
            dtype=np.float32,
        )
        self._np_random = np.random.default_rng()
        self.timestep = 0
        self.icu_free = ICU_MAX
        self.gen_free = GENERAL_MAX
        self.ptype = 2

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._np_random = np.random.default_rng(seed)
        self.timestep = 0
        self.icu_free = int(self._np_random.integers(0, ICU_MAX + 1))
        self.gen_free = int(self._np_random.integers(0, GENERAL_MAX + 1))
        self.ptype = int(self._np_random.choice([0, 1, 2]))
        return self._get_observation(), self._get_info()

    def step(self, action: int):
        action = int(np.clip(action, 0, self.action_space.n - 1))
        self.timestep += 1

        reward = 0.0
        icu = self.icu_free
        gen = self.gen_free
        patient_type = self.ptype
        events: list[str] = []

        if action == 0:
            if icu > 0:
                self.icu_free -= 1
                events.append("icu_allocated")
                if patient_type in [0, 1]:
                    reward += 5
                else:
                    reward -= 3
            else:
                reward -= 10
                events.append("icu_allocation_failed")

        elif action == 1:
            if gen > 0:
                self.gen_free -= 1
                events.append("general_allocated")
                if patient_type == 2:
                    reward += 5
                elif patient_type in [0, 1]:
                    if icu > 0:
                        reward -= 3
                    else:
                        reward += 2
            else:
                reward -= 10
                events.append("general_allocation_failed")

        elif action == 2:
            reward -= 1
            events.append("patient_delayed")
            if patient_type == 0:
                reward -= 10

        if self._np_random.random() < 0.3:
            self.icu_free = min(self.icu_free + 1, ICU_MAX)
            events.append("icu_bed_released")
        if self._np_random.random() < 0.4:
            self.gen_free = min(self.gen_free + 1, GENERAL_MAX)
            events.append("general_bed_released")

        emergency_rejection = 0
        if patient_type == 0:
            if action == 0 and icu == 0:
                emergency_rejection = 1
            elif action == 1 and gen == 0:
                emergency_rejection = 1
            elif action == 2:
                emergency_rejection = 1

        icu_occupied = 1 if self.icu_free < ICU_MAX else 0
        icu_utilisation = 1.0 - (self.icu_free / ICU_MAX)
        general_utilisation = 1.0 - (self.gen_free / GENERAL_MAX)

        self.ptype = int(self._np_random.choice([0, 1, 2], p=[0.2, 0.3, 0.5]))

        terminated = False
        truncated = self.timestep >= self.max_steps
        info = self._get_info(
            action=action,
            action_meaning=ACTION_MEANINGS[action],
            events=events,
            treated_patient_type=patient_type,
            emergency_rejection=emergency_rejection,
            icu_occupied=icu_occupied,
            icu_utilisation=icu_utilisation,
            general_utilisation=general_utilisation,
        )
        return self._get_observation(), float(reward), terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        return np.asarray([self.icu_free, self.gen_free, self.ptype], dtype=np.float32)

    def _get_info(
        self,
        action: int | None = None,
        action_meaning: str | None = None,
        events: list[str] | None = None,
        treated_patient_type: int | None = None,
        emergency_rejection: int = 0,
        icu_occupied: int | None = None,
        icu_utilisation: float | None = None,
        general_utilisation: float | None = None,
    ) -> dict[str, Any]:
        return {
            "timestep": self.timestep,
            "action": action,
            "action_meaning": action_meaning,
            "events": events or [],
            "patient_type": self.ptype,
            "patient_type_meaning": PATIENT_TYPES[self.ptype],
            "treated_patient_type": treated_patient_type,
            "treated_patient_type_meaning": PATIENT_TYPES[treated_patient_type]
            if treated_patient_type is not None
            else None,
            "icu_free": self.icu_free,
            "general_free": self.gen_free,
            "emergency_rejection": int(emergency_rejection),
            "icu_occupied": int(self.icu_free < ICU_MAX) if icu_occupied is None else int(icu_occupied),
            "icu_utilisation": float(1.0 - (self.icu_free / ICU_MAX))
            if icu_utilisation is None
            else float(icu_utilisation),
            "general_utilisation": float(1.0 - (self.gen_free / GENERAL_MAX))
            if general_utilisation is None
            else float(general_utilisation),
        }
