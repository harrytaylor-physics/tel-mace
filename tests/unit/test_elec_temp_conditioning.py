"""Electronic-temperature (Tel) conditioning: file stamping through to E(Tel).

The pathway under test is deliberately assembled from what the tree already
has rather than from a parallel Tel field: the electronic temperature travels
as the existing ``elec_temp`` property, is declared to the model as a
continuous per-graph entry in ``embedding_specs``, and enters the network as
``node_feats += joint_embedding(...)`` on the initial node features. Tel is a
rotational scalar, so no irreps change and equivariance is untouched.

What these tests pin, in order: the value survives the round trip from an
extxyz info line onto a batch; a zero-initialised pathway reproduces the
unconditioned model exactly; gradient reaches the pathway and depends on Tel;
and after a short training the energy varies smoothly with Tel at fixed
geometry.
"""

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import data, modules, tools
from mace.data.atomic_data import get_data_loader
from mace.cli.stamp_elec_temp import main as stamp_main
from mace.cli.stamp_elec_temp import stamp_info_line, tel_from_filename

Z_TABLE = tools.AtomicNumberTable([1, 8])
CUTOFF = 5.0

# A fixed water geometry; every energy below is evaluated on exactly this one,
# so anything that moves is Tel and not the structure.
POSITIONS = np.array(
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.93, 0.0, -0.24]], dtype=float
)
NUMBERS = np.array([8, 1, 1])

# Tel/10000: the training data spans 4000-8000 K, so this puts the input in
# [0.4, 0.8] instead of the thousands.
TEL_SCALE = 1.0e-4


@pytest.fixture(autouse=True)
def _float64():
    """Tel conditioning is asserted at 1e-10; float32 cannot carry that."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def spec(**overrides):
    base = {
        "type": "continuous",
        "per": "graph",
        "in_dim": 1,
        "emb_dim": 8,
        "scale": TEL_SCALE,
        "zero_init": True,
    }
    base.update(overrides)
    return {"elec_temp": base}


def build_model(embedding_specs, seed=0):
    torch.manual_seed(seed)
    return modules.ScaleShiftMACE(
        r_max=CUTOFF,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=3,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=len(Z_TABLE.zs),
        hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.zeros(len(Z_TABLE.zs)),
        avg_num_neighbors=3.0,
        atomic_numbers=Z_TABLE.zs,
        correlation=3,
        radial_type="bessel",
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        use_reduced_cg=False,
        embedding_specs=embedding_specs,
    )


def batch_at(tel):
    config = data.Configuration(
        atomic_numbers=NUMBERS,
        positions=POSITIONS,
        properties={
            "energy": 0.0,
            "forces": np.zeros((3, 3)),
            "elec_temp": tel,
        },
        property_weights={"energy": 1.0, "forces": 1.0},
        cell=np.eye(3) * 10.0,
        pbc=(False, False, False),
    )
    atomic_data = data.AtomicData.from_config(
        config, z_table=Z_TABLE, cutoff=CUTOFF
    )
    loader = get_data_loader([atomic_data], batch_size=1, shuffle=False)
    return next(iter(loader))


def energy_at(model, tel):
    return model(batch_at(tel).to_dict(), training=False)["energy"]


def unconditioned_energy(model, tel):
    """Energy from the same weights with the Tel pathway removed.

    `models.py` gates the pathway on `hasattr(self, "joint_embedding")`, so
    detaching the module is exactly the unconditioned model -- same parameters,
    same seed, no separately built baseline whose weights could drift.
    """
    pathway = model.joint_embedding
    del model.joint_embedding
    try:
        return model(batch_at(tel).to_dict(), training=False)["energy"]
    finally:
        model.joint_embedding = pathway


# --- the file stamping -------------------------------------------------------


def write_frames(path, temperatures):
    """A two-atom extxyz with a per-frame ionic temperature `T`."""
    lines = []
    for ionic_t in temperatures:
        lines.append("2\n")
        lines.append(
            'Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" '
            "Properties=species:S:1:pos:R:3 "
            f"T={ionic_t} REF_energy=-1.5 pbc=\"T T T\"\n"
        )
        lines.append("O 0.0 0.0 0.0\n")
        lines.append("H 0.0 0.0 0.96\n")
    path.write_text("".join(lines))


def test_tel_is_read_from_the_filename():
    from pathlib import Path

    assert tel_from_filename(Path("train_Tel4000.xyz")) == 4000.0
    assert tel_from_filename(Path("train_Tel_6000_slice.xyz")) == 6000.0
    assert tel_from_filename(Path("tel-8000.5.xyz")) == 8000.5
    with pytest.raises(ValueError, match="cannot infer"):
        tel_from_filename(Path("train.xyz"))
    with pytest.raises(ValueError, match="conflicting"):
        tel_from_filename(Path("Tel4000_to_Tel6000.xyz"))


def test_stamping_preserves_every_other_byte(tmp_path):
    source = tmp_path / "train_Tel4000.xyz"
    write_frames(source, [6931.6, 3183.965, 6197.004])
    original = source.read_text()

    assert stamp_main([str(source), "--in-place"]) == 0

    stamped = source.read_text()
    assert stamped != original
    # Everything but the appended key is untouched, so REF_energy and the
    # coordinates keep the exact digits they were written with.
    assert stamped.replace(" elec_temp=4000.0", "") == original


def test_stamped_temperature_reaches_the_batch(tmp_path):
    import ase.io

    source = tmp_path / "train_Tel6000.xyz"
    write_frames(source, [6931.6, 3183.965])
    assert stamp_main([str(source), "--in-place"]) == 0

    frames = ase.io.read(source, ":")
    assert len(frames) == 2
    # The ionic temperature is a different quantity and must survive intact.
    assert [f.info["T"] for f in frames] == pytest.approx([6931.6, 3183.965])

    keyspec = data.KeySpecification(
        info_keys={"energy": "REF_energy", "elec_temp": "elec_temp"},
        arrays_keys={},
    )
    for frame in frames:
        config = data.config_from_atoms(frame, key_specification=keyspec)
        assert config.properties["elec_temp"] == pytest.approx(6000.0)
        atomic_data = data.AtomicData.from_config(
            config, z_table=Z_TABLE, cutoff=CUTOFF
        )
        assert atomic_data.elec_temp.item() == pytest.approx(6000.0)

    loaded = next(
        iter(
            get_data_loader(
                [
                    data.AtomicData.from_config(
                        data.config_from_atoms(f, key_specification=keyspec),
                        z_table=Z_TABLE,
                        cutoff=CUTOFF,
                    )
                    for f in frames
                ],
                batch_size=2,
                shuffle=False,
            )
        )
    )
    # Collated per-graph, one entry per frame -- the shape the embedding indexes
    # with `feat[batch]`.
    assert loaded.elec_temp.shape == (2,)
    assert loaded.elec_temp.tolist() == pytest.approx([6000.0, 6000.0])


def test_double_stamping_is_refused_unless_forced():
    line = 'Properties=species:S:1:pos:R:3 T=6931.6\n'
    once = stamp_info_line(line, "elec_temp", 4000.0, force=False)
    assert once.rstrip().endswith("elec_temp=4000.0")

    with pytest.raises(ValueError, match="already carries"):
        stamp_info_line(once, "elec_temp", 4000.0, force=False)

    forced = stamp_info_line(once, "elec_temp", 8000.0, force=True)
    assert forced.rstrip().endswith("elec_temp=8000.0")
    assert forced.count("elec_temp=") == 1
    assert "T=6931.6" in forced


# --- the zero-initialised pathway --------------------------------------------


def test_zero_init_reproduces_the_base_model_at_any_tel():
    model = build_model(spec())

    for tel in (0.0, 4000.0, 6000.0, 8000.0, 1.0e6):
        conditioned = energy_at(model, tel)
        base = unconditioned_energy(model, tel)
        difference = (conditioned - base).abs().max().item()
        assert difference <= 1e-10, f"Tel={tel} moved the energy by {difference}"
        # Stronger than the contract and true by construction: the zeroed output
        # layer emits exactly 0, the bias-free projection maps that to 0, and
        # SiLU(0) is 0, so the added term is a hard zero rather than a small one.
        assert difference == 0.0


def test_pathway_is_live_without_zero_init():
    """The identity above is the zero-init doing its job, not a dead wire."""
    model = build_model(spec(zero_init=False))
    difference = (
        (energy_at(model, 6000.0) - unconditioned_energy(model, 6000.0))
        .abs()
        .max()
        .item()
    )
    assert difference > 1e-3


def test_scale_puts_kelvin_and_normalised_input_on_the_same_footing():
    kelvin = build_model(spec(zero_init=False))
    normalised = build_model(spec(scale=1.0, zero_init=False))
    normalised.load_state_dict(kelvin.state_dict())

    # Tel=6000 K through scale=1e-4 is Tel=0.6 through no scaling at all.
    assert energy_at(kelvin, 6000.0).item() == pytest.approx(
        energy_at(normalised, 0.6).item(), abs=1e-12
    )
    # And the scaling is not a no-op: unnormalised Kelvin saturates the MLP.
    assert (
        energy_at(normalised, 6000.0) - energy_at(kelvin, 6000.0)
    ).abs().item() > 1.0


def test_gradient_flows_through_the_tel_pathway():
    model = build_model(spec())
    zeroed = model.joint_embedding.embedders["elec_temp"][-1]

    grads = {}
    for tel in (4000.0, 8000.0):
        model.zero_grad()
        model(batch_at(tel).to_dict(), training=True)["energy"].sum().backward()
        grads[tel] = zeroed.weight.grad.clone()
        assert zeroed.weight.grad.abs().max().item() > 0.0
        assert zeroed.bias.grad.abs().max().item() > 0.0

    # The gradient carries Tel, which is what makes the pathway learnable rather
    # than merely connected.
    assert (grads[4000.0] - grads[8000.0]).abs().max().item() > 1e-6

    # At init the layers on the far side of the zero see nothing: their gradient
    # is proportional to the zeroed activations. That is inherent to getting an
    # exact identity out of a multiplicative chain, so it is pinned rather than
    # treated as a defect -- and it lasts exactly one step.
    assert model.joint_embedding.project[0].weight.grad.abs().max().item() == 0.0

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.zero_grad()
    model(batch_at(6000.0).to_dict(), training=True)["energy"].sum().backward()
    optimizer.step()
    model.zero_grad()
    model(batch_at(6000.0).to_dict(), training=True)["energy"].sum().backward()
    for name, parameter in model.joint_embedding.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.abs().max().item() > 0.0, name


# --- what it learns ----------------------------------------------------------


TRAINING_TARGETS = {4000.0: -1.0, 6000.0: 0.0, 8000.0: 1.0}


def train_toy_tel_dependence(model, steps=120, lr=2.0e-2):
    """Fit E(Tel) at fixed geometry to three targets; returns the final loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = None
    for _ in range(steps):
        optimizer.zero_grad()
        loss = sum(
            (
                model(batch_at(tel).to_dict(), training=True)["energy"].squeeze()
                - target
            )
            ** 2
            for tel, target in TRAINING_TARGETS.items()
        )
        loss.backward()
        optimizer.step()
    return loss.item()


def test_energy_varies_smoothly_with_tel_after_training():
    torch.manual_seed(0)
    model = build_model(spec())

    assert train_toy_tel_dependence(model) < 1.0e-2

    for tel, target in TRAINING_TARGETS.items():
        assert energy_at(model, tel).item() == pytest.approx(target, abs=0.1)

    grid = np.linspace(4000.0, 8000.0, 41)
    energies = np.array([energy_at(model, float(t)).item() for t in grid])

    # Smoothness on a fixed geometry: with the endpoints two energy units apart,
    # curvature over a 100 K step stays four orders of magnitude below that, so
    # the surface has no steps or kinks a thermostat sweep could trip over.
    span = abs(energies[-1] - energies[0])
    assert span > 1.0
    second_difference = np.abs(np.diff(energies, 2)).max()
    assert second_difference < 1.0e-3 * span

    # Monotone in Tel here, which the three targets asked for.
    assert np.all(np.diff(energies) > 0.0)


def test_tel_conditioning_leaves_forces_equivariant():
    """Tel is a rotational scalar, so conditioning must not disturb forces."""
    torch.manual_seed(0)
    model = build_model(spec(zero_init=False))

    output = model(batch_at(6000.0).to_dict(), training=False)
    forces = output["forces"].detach().numpy()

    rotation = o3.rand_matrix(dtype=torch.float64).numpy()
    rotated = data.Configuration(
        atomic_numbers=NUMBERS,
        positions=POSITIONS @ rotation.T,
        properties={
            "energy": 0.0,
            "forces": np.zeros((3, 3)),
            "elec_temp": 6000.0,
        },
        property_weights={"energy": 1.0, "forces": 1.0},
        cell=np.eye(3) * 10.0,
        pbc=(False, False, False),
    )
    loader = get_data_loader(
        [data.AtomicData.from_config(rotated, z_table=Z_TABLE, cutoff=CUTOFF)],
        batch_size=1,
        shuffle=False,
    )
    rotated_output = model(next(iter(loader)).to_dict(), training=False)

    assert rotated_output["energy"].item() == pytest.approx(
        output["energy"].item(), abs=1e-10
    )
    np.testing.assert_allclose(
        rotated_output["forces"].detach().numpy(), forces @ rotation.T, atol=1e-10
    )


# --- the ASE calculator ------------------------------------------------------


def water_atoms():
    from ase import Atoms

    return Atoms(
        "OH2", positions=POSITIONS, cell=np.eye(3) * 10.0, pbc=False
    )


def calculator_energy(model, atoms, **kwargs):
    from mace.calculators import MACECalculator

    atoms = atoms.copy()
    atoms.calc = MACECalculator(
        models=[model], device="cpu", default_dtype="float64", **kwargs
    )
    return atoms.get_potential_energy()


def test_calculator_conditions_on_elec_temp():
    torch.manual_seed(0)
    model = build_model(spec(zero_init=False))
    atoms = water_atoms()

    cold = calculator_energy(model, atoms, elec_temp=4000.0)
    hot = calculator_energy(model, atoms, elec_temp=8000.0)
    assert abs(cold - hot) > 1e-3

    # The calculator takes Kelvin: the /10000 lives in the spec's `scale`, which
    # is part of the model, so inference cannot disagree with training about it.
    assert cold == pytest.approx(energy_at(model, 4000.0).item(), abs=1e-10)
    assert hot == pytest.approx(energy_at(model, 8000.0).item(), abs=1e-10)


def test_frame_elec_temp_overrides_the_calculator_default():
    torch.manual_seed(0)
    model = build_model(spec(zero_init=False))

    hot_frame = water_atoms()
    hot_frame.info["elec_temp"] = 8000.0
    # Constructed cold, but the frame says hot, so one calculator can sweep Tel.
    assert calculator_energy(model, hot_frame, elec_temp=4000.0) == pytest.approx(
        calculator_energy(model, water_atoms(), elec_temp=8000.0), abs=1e-12
    )


def test_missing_elec_temp_warns_rather_than_passing_silently(caplog):
    torch.manual_seed(0)
    model = build_model(spec(zero_init=False))

    with caplog.at_level("WARNING"):
        fallback = calculator_energy(model, water_atoms())

    assert "elec_temp" in caplog.text
    # It falls back to 0 K rather than refusing: a model may legitimately have
    # been trained with 0 K meaning the ground state.
    assert fallback == pytest.approx(energy_at(model, 0.0).item(), abs=1e-10)


def test_calculator_without_the_embedding_is_untouched(caplog):
    """A plain model gains neither a warning nor a new required argument."""
    torch.manual_seed(0)
    model = build_model(None)

    with caplog.at_level("WARNING"):
        energy = calculator_energy(model, water_atoms())

    assert np.isfinite(energy)
    assert "elec_temp" not in caplog.text
