"""Electronic-temperature conditioning for the ASE calculators.

A model conditioned on Tel carries an ``elec_temp`` entry in its
``embedding_specs``, and expects the temperature in Kelvin: the normalisation
(Tel/10000) lives in that spec's ``scale`` and therefore inside the checkpoint,
so nothing out here has to know the convention or can disagree with it.

Kept beside the calculators rather than inside them so the feature stays in one
place, and so ``mace.py`` gains only the call.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

ELEC_TEMP_KEY = "elec_temp"


def models_expect_elec_temp(models: Iterable[Any]) -> bool:
    """True when any model carries an ``elec_temp`` embedding spec."""
    return any(
        ELEC_TEMP_KEY in (getattr(model, "embedding_specs", None) or {})
        for model in models
    )


def apply_default_elec_temp(
    config, default: Optional[float], models: Iterable[Any], warned: bool
) -> bool:
    """Fill in the electronic temperature where a frame does not carry one.

    A frame's own ``elec_temp`` wins over the calculator's default, so a single
    calculator can still sweep Tel frame by frame.

    Warns once, and returns the updated "already warned" flag, when the model
    was conditioned on Tel but nothing supplied a value, so it falls back to
    0 K. That is a confident ground-state answer for a hot system -- a wrong
    number rather than an error, which is worth saying out loud. It warns rather
    than refusing because a model may legitimately have been trained with 0 K
    meaning the ground state.
    """
    if config.properties.get(ELEC_TEMP_KEY) is not None:
        return warned
    if default is not None:
        config.properties[ELEC_TEMP_KEY] = float(default)
        config.property_weights[ELEC_TEMP_KEY] = 1.0
        return warned
    if not warned and models_expect_elec_temp(models):
        logging.warning(
            "This model carries an `elec_temp` embedding but no electronic "
            "temperature was supplied, so 0 K will be used. Pass elec_temp=<K> "
            "to the calculator, or set atoms.info['elec_temp']."
        )
        return True
    return warned
