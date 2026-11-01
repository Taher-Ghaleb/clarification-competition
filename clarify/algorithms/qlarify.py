# SPDX-FileCopyrightText: 2026 Taher Ghaleb <taherghaleb@trentu.ca>
#
# SPDX-License-Identifier: MIT

"""Qlarify: Contrastive Clarification Through Skeptical Assumption Auditing.

Qlarify explores competing interpretations of programming specifications,
challenges implicit assumptions, and identifies when clarification is
valuable for improving solution correctness.

Team: TaG
Team Members: Taher Ghaleb
Main Contact: taherghaleb@trentu.ca
"""

from clarify.baselines import ClarificationAlgorithmBase


class Qlarify(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {}

    def run(self, env, problem) -> str:
        prompt = f"""
You are a contrastive clarification algorithm.

Analyze the following programming problem:

{problem["prompt"]}

Entry point: {problem["entry_point"]}

Consider alternative interpretations, identify implicit assumptions,
and determine the most reasonable approach to solving the problem.

Provide a complete Python implementation.
"""
        return env.llm([{"role": "user", "content": prompt}])
