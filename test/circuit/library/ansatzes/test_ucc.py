# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the UCC Ansatz."""

from test import QiskitNatureTestCase

from ddt import data, ddt, unpack

from qiskit_nature import QiskitNatureError
from qiskit_nature.circuit.library import UCC
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.operators.second_quantization import FermionicOp


def assert_ucc_like_ansatz(test_case, ansatz, num_spin_orbitals, expected_ops):
    """Assertion utility."""
    
    # excitation_ops = ansatz.excitation_ops()[::2] if include_imaginary else ansatz.excitation_ops()
    excitation_ops = ansatz.excitation_ops()
    test_case.assertEqual(len(excitation_ops), len(expected_ops))
    for op, exp in zip(excitation_ops, expected_ops):
        test_case.assertListEqual(op.to_list(), exp.to_list())

    ansatz._build()
    test_case.assertEqual(ansatz.num_qubits, num_spin_orbitals)


@ddt
class TestUCC(QiskitNatureTestCase):
    """Tests for the UCC Ansatz."""

    # Note: many variations of this class are tested by its sub-classes UCCSD, PUCCD and SUCCD.
    # Thus, the tests here mainly cover edge cases which those classes cannot account for.

    @unpack
    @data(
        (
            "t",
            8,
            (2, 2),
            [
                FermionicOp([("++--+I-I", 1j), ("--++-I+I", 1j)], display_format="dense"),
                FermionicOp([("++--+II-", 1j), ("--++-II+", 1j)], display_format="dense"),
                FermionicOp([("++--I+-I", 1j), ("--++I-+I", 1j)], display_format="dense"),
                FermionicOp([("++--I+I-", 1j), ("--++I-I+", 1j)], display_format="dense"),
                FermionicOp([("+I-I++--", 1j), ("-I+I--++", 1j)], display_format="dense"),
                FermionicOp([("+II-++--", 1j), ("-II+--++", 1j)], display_format="dense"),
                FermionicOp([("I+-I++--", 1j), ("I-+I--++", 1j)], display_format="dense"),
                FermionicOp([("I+I-++--", 1j), ("I-I+--++", 1j)], display_format="dense"),
            ],
            False,
        ),
        (
            "sd",
            8,
            (2, 2),
            [FermionicOp([('+I-IIIII', 1j), ('-I+IIIII', 1j)], display_format='dense'),
             FermionicOp([('+I-IIIII', (-1+0j)), ('-I+IIIII', (1-0j))], display_format='dense'),
             FermionicOp([('+II-IIII', 1j), ('-II+IIII', 1j)], display_format='dense'),
             FermionicOp([('+II-IIII', (-1+0j)), ('-II+IIII', (1-0j))], display_format='dense'),
             FermionicOp([('I+-IIIII', 1j), ('I-+IIIII', 1j)], display_format='dense'),
             FermionicOp([('I+-IIIII', (-1+0j)), ('I-+IIIII', (1-0j))], display_format='dense'),
             FermionicOp([('I+I-IIII', 1j), ('I-I+IIII', 1j)], display_format='dense'),
             FermionicOp([('I+I-IIII', (-1+0j)), ('I-I+IIII', (1-0j))], display_format='dense'),
             FermionicOp([('IIII+I-I', 1j), ('IIII-I+I', 1j)], display_format='dense'),
             FermionicOp([('IIII+I-I', (-1+0j)), ('IIII-I+I', (1-0j))], display_format='dense'),
             FermionicOp([('IIII+II-', 1j), ('IIII-II+', 1j)], display_format='dense'),
             FermionicOp([('IIII+II-', (-1+0j)), ('IIII-II+', (1-0j))], display_format='dense'),
             FermionicOp([('IIIII+-I', 1j), ('IIIII-+I', 1j)], display_format='dense'),
             FermionicOp([('IIIII+-I', (-1+0j)), ('IIIII-+I', (1-0j))], display_format='dense'),
             FermionicOp([('IIIII+I-', 1j), ('IIIII-I+', 1j)], display_format='dense'),
             FermionicOp([('IIIII+I-', (-1+0j)), ('IIIII-I+', (1-0j))], display_format='dense'),
             FermionicOp([('++--IIII', 1j), ('--++IIII', (-0-1j))], display_format='dense'),
             FermionicOp([('++--IIII', (-1+0j)), ('--++IIII', (-1+0j))], display_format='dense'),
             FermionicOp([('+I-I+I-I', 1j), ('-I+I-I+I', (-0-1j))], display_format='dense'),
             FermionicOp([('+I-I+I-I', (-1+0j)), ('-I+I-I+I', (-1+0j))], display_format='dense'),
             FermionicOp([('+I-I+II-', 1j), ('-I+I-II+', (-0-1j))], display_format='dense'),
             FermionicOp([('+I-I+II-', (-1+0j)), ('-I+I-II+', (-1+0j))], display_format='dense'),
             FermionicOp([('+I-II+-I', 1j), ('-I+II-+I', (-0-1j))], display_format='dense'),
             FermionicOp([('+I-II+-I', (-1+0j)), ('-I+II-+I', (-1+0j))], display_format='dense'),
             FermionicOp([('+I-II+I-', 1j), ('-I+II-I+', (-0-1j))], display_format='dense'),
             FermionicOp([('+I-II+I-', (-1+0j)), ('-I+II-I+', (-1+0j))], display_format='dense'),
             FermionicOp([('+II-+I-I', 1j), ('-II+-I+I', (-0-1j))], display_format='dense'),
             FermionicOp([('+II-+I-I', (-1+0j)), ('-II+-I+I', (-1+0j))], display_format='dense'),
             FermionicOp([('+II-+II-', 1j), ('-II+-II+', (-0-1j))], display_format='dense'),
             FermionicOp([('+II-+II-', (-1+0j)), ('-II+-II+', (-1+0j))], display_format='dense'),
             FermionicOp([('+II-I+-I', 1j), ('-II+I-+I', (-0-1j))], display_format='dense'),
             FermionicOp([('+II-I+-I', (-1+0j)), ('-II+I-+I', (-1+0j))], display_format='dense'),
             FermionicOp([('+II-I+I-', 1j), ('-II+I-I+', (-0-1j))], display_format='dense'),
             FermionicOp([('+II-I+I-', (-1+0j)), ('-II+I-I+', (-1+0j))], display_format='dense'),
             FermionicOp([('I+-I+I-I', 1j), ('I-+I-I+I', (-0-1j))], display_format='dense'),
             FermionicOp([('I+-I+I-I', (-1+0j)), ('I-+I-I+I', (-1+0j))], display_format='dense'),
             FermionicOp([('I+-I+II-', 1j), ('I-+I-II+', (-0-1j))], display_format='dense'),
             FermionicOp([('I+-I+II-', (-1+0j)), ('I-+I-II+', (-1+0j))], display_format='dense'),
             FermionicOp([('I+-II+-I', 1j), ('I-+II-+I', (-0-1j))], display_format='dense'),
             FermionicOp([('I+-II+-I', (-1+0j)), ('I-+II-+I', (-1+0j))], display_format='dense'),
             FermionicOp([('I+-II+I-', 1j), ('I-+II-I+', (-0-1j))], display_format='dense'),
             FermionicOp([('I+-II+I-', (-1+0j)), ('I-+II-I+', (-1+0j))], display_format='dense'),
             FermionicOp([('I+I-+I-I', 1j), ('I-I+-I+I', (-0-1j))], display_format='dense'),
             FermionicOp([('I+I-+I-I', (-1+0j)), ('I-I+-I+I', (-1+0j))], display_format='dense'),
             FermionicOp([('I+I-+II-', 1j), ('I-I+-II+', (-0-1j))], display_format='dense'),
             FermionicOp([('I+I-+II-', (-1+0j)), ('I-I+-II+', (-1+0j))], display_format='dense'),
             FermionicOp([('I+I-I+-I', 1j), ('I-I+I-+I', (-0-1j))], display_format='dense'),
             FermionicOp([('I+I-I+-I', (-1+0j)), ('I-I+I-+I', (-1+0j))], display_format='dense'),
             FermionicOp([('I+I-I+I-', 1j), ('I-I+I-I+', (-0-1j))], display_format='dense'),
             FermionicOp([('I+I-I+I-', (-1+0j)), ('I-I+I-I+', (-1+0j))], display_format='dense'),
             FermionicOp([('IIII++--', 1j), ('IIII--++', (-0-1j))], display_format='dense'),
             FermionicOp([('IIII++--', (-1+0j)), ('IIII--++', (-1+0j))], display_format='dense')]
            ,
            True,
        ),
        
        
        
        
        
        (
            "t",
            8,
            (2, 1),
            [
                FermionicOp([("++--+-II", 1j), ("--++-+II", 1j)], display_format="dense"),
                FermionicOp([("++--+I-I", 1j), ("--++-I+I", 1j)], display_format="dense"),
                FermionicOp([("++--+II-", 1j), ("--++-II+", 1j)], display_format="dense"),
            ],
            False,
        ),
        (
            "q",
            8,
            (2, 2),
            [FermionicOp([("++--++--", 1j), ("--++--++", -1j)], display_format="dense")],
            False,
        ),
        
        # TODO: add more edge cases?
    )
    
    def test_ucc_ansatz(self, excitations, num_spin_orbitals, num_particles, expect, include_imaginary):
        """Tests the UCC Ansatz."""
        
        converter = QubitConverter(JordanWignerMapper())

        ansatz = UCC(
            qubit_converter=converter,
            num_particles=num_particles,
            num_spin_orbitals=num_spin_orbitals,
            excitations=excitations,
            include_imaginary=include_imaginary
        )

        assert_ucc_like_ansatz(self, ansatz, num_spin_orbitals, expect)


    @unpack
    @data(
        # Excitations not a list of pairs
        (
            8,
            (2, 2),
            [((0, 1, 4), (2, 3, 6), (2, 3, 7))],
        ),
        # Excitation pair has not same length
        (
            8,
            (2, 2),
            [((0, 1, 4), (2, 3, 6, 7))],
        ),
        # Excitation pair with non-unique indices
        (
            8,
            (2, 2),
            [((0, 1, 4), (2, 4, 6))],
        ),
        (
            8,
            (2, 2),
            [((0, 1, 1), (2, 3, 6))],
        ),
    )
    def test_custom_excitations(self, num_spin_orbitals, num_particles, excitations):
        """Tests if an error is raised when the excitations have a wrong format"""
        converter = QubitConverter(JordanWignerMapper())

        # pylint: disable=unused-argument
        def custom_excitations(num_spin_orbitals, num_particles):
            return excitations

        ansatz = UCC(
            qubit_converter=converter,
            num_particles=num_particles,
            num_spin_orbitals=num_spin_orbitals,
            excitations=custom_excitations,
        )

        with self.assertRaises(QiskitNatureError):
            ansatz.excitation_ops()
