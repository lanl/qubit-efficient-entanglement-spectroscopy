#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is a library of functions which generate circuits to perform entanglement spectrsocopy
and which process the results from those circuits.

See https://arxiv.org/abs/2010.03080

Authors:
	Justin Yirka	yirka@utexas.edu
	Yigit Subasi	ysubasi@lanl.gov
"""

######################
###### Imports #######
######################
import numpy as np
from scipy.special import comb

from idle_scheduler import schedule_idle_gates
# available at https://github.com/gadial/qiskit-aer/blob/ff56889c3cf0486b1ad094634e88d7e756b6db3c/qiskit/providers/aer/noise/utils/idle_scheduler.py

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import thermal_relaxation_error, ReadoutError, pauli_error, depolarizing_error
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller

###############################
###### Helper functions #######
###############################
def mergeRegisters(*qregs):
	"""
	Returns a single register containing all the qubits (or cbits) from multiple registers
	"""
	bigReg = []
	for qreg in qregs:
		bigReg += [q for q in qreg]
	return bigReg

def reverseMergeRegisters(*qregs):
	"""
	Returns a merged register after reversing the ordering of *regs and the ordering of each register
	"""
	bigReg = []
	for qreg in reversed(qregs):
		bigReg += [q for q in reversed(qreg)]
	return bigReg

def generate_cswap_instruction(k = 1):
	"""
	Returns an Instruction which implements a CSWAP between two registers of size k

	When calling the instruction, the register must be in the order [cont, targ1, targ2].

	We use this method to be sure we use a specific number and sequence of gates.
	In fact, this is the default decomposition of circ.cswap(), but we use to be sure it's this exact circuit.
	"""
	c = QuantumRegister(1, "c")
	t1 = QuantumRegister(k, "t1")
	t2 = QuantumRegister(k, "t2")
	circ = QuantumCircuit(c,t1,t2, name="CSWAP'")
	circ.cx(t2,t1)
	circ.h(t2)
	circ.cx(t1,t2)
	circ.tdg(t2)
	circ.cx(c,t2)
	circ.t(t2)
	circ.cx(t1,t2)
	circ.t(t1)
	circ.tdg(t2)
	circ.cx(c,t2)
	circ.cx(c,t1)
	circ.t(t2)
	circ.t(c)
	circ.tdg(t1)
	circ.h(t2)
	circ.cx(t2,t1)
	circ.cx(c,t1)
	cswap_instruc = circ.to_instruction()
	return cswap_instruc

def generate_default_prep_state_instruction(theta, k = 1, theta2 = None):
	"""
	Returns an Instruction which prepares a state on 2k, depending on the theta parameters

	For k=1, we only use the first theta parameters.
	Our algorithm for k=1 produces a state such that
		Tr(rho_A^n) = 2^(-(n-1)) * [sum (n choose k) sin(theta)^k for even k up to n].

	Undefined for k>1.
	"""
	if k == 1:
		qr = QuantumRegister(1, "a")
		qr2 = QuantumRegister(1, "b")
		prep_state_circ = QuantumCircuit(qr, qr2, name='Prep_State')
		prep_state_circ.h(qr)
		prep_state_circ.u2(theta - np.pi / 2, np.pi / 2, qr2)
		prep_state_circ.cx(qr, qr2)
		prep_state = prep_state_circ.to_instruction()
	else:
		raise Exception("generate_default_prep_state_instruction is only defined for k= 1.")
	return prep_state

def computeExactTraces_forDefaultStatePrep(n, theta_list, k = 1):
	"""
	Computes list of Tr(rho_A^n) for rho prepared with generate_default_prep_state_instruction and theta_list.

	`k` is such that rho is a state on 2k qubits, and rho_A is k qubits.

	For k=1, the formula is 2^(-(n-1)) * [sum (n choose k) sin(theta)^k for even k up to n].

	Only defined for k=1 right now.
	"""
	if k == 1:
		return [2**(-(n-1)) * sum(comb(n, k) * np.sin(theta)**k for k in range(0, n+1, 2)) for theta in theta_list]
	else:
		raise Exception("computeExactValues_forDefaultStatePrep is only defined for k=1.")


#####################################
###### Noise Helper functions #######
#####################################
DEFAULT_OP_TIMES = {
	"measure" : 3,
	"reset" : 2,
	"u1" : 1,
	"u2" : 1,
	"u3" : 1,
	"id" : 1,
	"cx" : 5
}

T1 = 200
T2 = 200
THERMAL_POPULATION_1 = 10**(-7)

DEPOLARIZATION_PROB_SINGLE_QUBIT = 0.001
DEPOLARIZATION_PROB_TWO_QUBIT = 0.005

PAULI_ERROR_PROB_SINGLE_QUBIT = 0.001
PAULI_ERROR_PROB_TWO_QUBIT = 0.005

MEASUREMENT_ERROR_PROB_0_given_1 = 0.02
MEASUREMENT_ERROR_PROB_1_given_0 = 0.02

DEPOLARIZATION_PROB_RESET = 0
PAULI_ERROR_PROB_RESET = 0

TWO_QUBIT_GATES = ["cx"]
ONE_QUBIT_GATES = ["u1", "u2", "u3"]
OTHER_ONE_QUBIT_OPS = ["measure", "reset", "id"]
ALL_OPS = TWO_QUBIT_GATES + ONE_QUBIT_GATES + OTHER_ONE_QUBIT_OPS

def construct_noise_model_full(num_qubits,
								  two_qubit_gates = TWO_QUBIT_GATES, one_qubit_gates = ONE_QUBIT_GATES,
								  other_one_qubit_ops = OTHER_ONE_QUBIT_OPS, op_times = DEFAULT_OP_TIMES,
								  t1 = T1, t2 = T2, thermal_population_1 = THERMAL_POPULATION_1,
								  depolarization_prob_single_qubit = DEPOLARIZATION_PROB_SINGLE_QUBIT,
								  depolarization_prob_two_qubit = DEPOLARIZATION_PROB_TWO_QUBIT,
								  pauli_error_prob_single_qubit = PAULI_ERROR_PROB_SINGLE_QUBIT,
								  pauli_error_prob_two_qubit = PAULI_ERROR_PROB_TWO_QUBIT,
								  measurement_error_prob0_given_1 = MEASUREMENT_ERROR_PROB_0_given_1,
								  measurement_error_prob1_given_0 = MEASUREMENT_ERROR_PROB_1_given_0,
								  depolarization_prob_reset=DEPOLARIZATION_PROB_RESET,
								  pauli_error_prob_reset=PAULI_ERROR_PROB_RESET):
	"""
	Returns a NoiseModel with thermal, gate (depol. and pauli), and readout noise.

	This noise model is fairly flexible, but you can customize it more if you write your own. For example,
	IBM has other types of noise.

	All error parameters can be adjusted, but if no parameters are given, they default to the values set above in this file.
	Applies errors to all qubits and all gates.

	For most errors, you can just set them to 0 if they're not desired.
	But, to get rid of thermal noise, use the function `construct_noise_model_noThermal`.
	And even though you can just set thing to 0, the unnecessary code below could be slow, so checkout `construct_noise_model_readoutOnly`
	and `construct_noise_model_thermalOnly` below to speed things up.

	Note that thermal noise is tricky! Empty wires do not receive noise by default.
	Look at `construct_modified_circuit_for_thermal_noise` for details.
	"""
	noise_model = NoiseModel()

	noise_model.add_all_qubit_readout_error(ReadoutError([
		[1 - measurement_error_prob1_given_0, measurement_error_prob1_given_0],
		[measurement_error_prob0_given_1, 1 - measurement_error_prob0_given_1]
	]), warnings=False)
	noise_model.add_all_qubit_quantum_error(pauli_error([
			("X", pauli_error_prob_reset),
			("Y", pauli_error_prob_reset),
			("Z", pauli_error_prob_reset),
			("I", 1 - 3 * pauli_error_prob_reset)
		]), "reset", warnings=False)
	noise_model.add_all_qubit_quantum_error(depolarizing_error(depolarization_prob_reset, 1), "reset", warnings=False)
	for g in one_qubit_gates + other_one_qubit_ops:
		noise_model.add_all_qubit_quantum_error(thermal_relaxation_error(t1,t2, op_times[g], thermal_population_1), g, warnings=False)
	for g in one_qubit_gates:
		noise_model.add_all_qubit_quantum_error(pauli_error([
			("X", pauli_error_prob_single_qubit),
			("Y", pauli_error_prob_single_qubit),
			("Z", pauli_error_prob_single_qubit),
			("I", 1 - 3 * pauli_error_prob_single_qubit)
		]), g, warnings=False)
		noise_model.add_all_qubit_quantum_error(depolarizing_error(depolarization_prob_single_qubit, 1), g, warnings=False)
	for g in two_qubit_gates:
		thermalErrorTwo = thermal_relaxation_error(t1,t2, op_times[g], thermal_population_1).expand(thermal_relaxation_error(t1,t2,op_times[g],thermal_population_1))
		pauliError = pauli_error([
			("X", pauli_error_prob_two_qubit),
			("Y", pauli_error_prob_two_qubit),
			("Z", pauli_error_prob_two_qubit),
			("I", 1 - 3 * pauli_error_prob_two_qubit)
		], g)
		pauliErrorTwo = pauliError.expand(pauliError)
		depolarizingErrorTwo = depolarizing_error(depolarization_prob_two_qubit, 2)
		for i in range(num_qubits):
			for j in range(num_qubits):
				if i != j:
					noise_model.add_quantum_error(thermalErrorTwo, g, [i,j], warnings=False)
					noise_model.add_quantum_error(pauliErrorTwo, g, [i,j], warnings=False)
					noise_model.add_quantum_error(depolarizingErrorTwo, g, [i,j], warnings=False)
	return noise_model

def construct_noise_model_noThermal(num_qubits,
								  two_qubit_gates = TWO_QUBIT_GATES, one_qubit_gates = ONE_QUBIT_GATES,
								  depolarization_prob_single_qubit = DEPOLARIZATION_PROB_SINGLE_QUBIT,
								  depolarization_prob_two_qubit = DEPOLARIZATION_PROB_TWO_QUBIT,
								  pauli_error_prob_single_qubit = PAULI_ERROR_PROB_SINGLE_QUBIT,
								  pauli_error_prob_two_qubit = PAULI_ERROR_PROB_TWO_QUBIT,
								  measurement_error_prob0_given_1 = MEASUREMENT_ERROR_PROB_0_given_1,
								  measurement_error_prob1_given_0 = MEASUREMENT_ERROR_PROB_1_given_0,
								  depolarization_prob_reset=DEPOLARIZATION_PROB_RESET,
								  pauli_error_prob_reset=PAULI_ERROR_PROB_RESET, **kwargs):
	"""
	Returns a NoiseModel with gate (depol. and pauli) and readout noise.

	All error parameters can be adjusted. If no parameters are given, they default to the values set above in this file.
	Applies errors to all qubits and all gates.

	For all of these errors, can just set parameter to 0 if error is not desired.
	"""
	noise_model = NoiseModel()

	noise_model.add_all_qubit_readout_error(ReadoutError([
		[1 - measurement_error_prob1_given_0, measurement_error_prob1_given_0],
		[measurement_error_prob0_given_1, 1 - measurement_error_prob0_given_1]
	]), warnings=False)
	noise_model.add_all_qubit_quantum_error(pauli_error([
			("X", pauli_error_prob_reset),
			("Y", pauli_error_prob_reset),
			("Z", pauli_error_prob_reset),
			("I", 1 - 3 * pauli_error_prob_reset)
		]), "reset", warnings=False)
	noise_model.add_all_qubit_quantum_error(depolarizing_error(depolarization_prob_reset, 1), "reset", warnings=False)
	for g in one_qubit_gates:
		noise_model.add_all_qubit_quantum_error(pauli_error([
			("X", pauli_error_prob_single_qubit),
			("Y", pauli_error_prob_single_qubit),
			("Z", pauli_error_prob_single_qubit),
			("I", 1 - 3 * pauli_error_prob_single_qubit)
		]), g, warnings=False)
		noise_model.add_all_qubit_quantum_error(depolarizing_error(depolarization_prob_single_qubit, 1), g, warnings=False)
	for g in two_qubit_gates:
		pauliError = pauli_error([
			("X", pauli_error_prob_two_qubit),
			("Y", pauli_error_prob_two_qubit),
			("Z", pauli_error_prob_two_qubit),
			("I", 1 - 3 * pauli_error_prob_two_qubit)
		], g)
		pauliErrorTwo = pauliError.expand(pauliError)
		depolarizingErrorTwo = depolarizing_error(depolarization_prob_two_qubit, 2)
		for i in range(num_qubits):
			for j in range(num_qubits):
				if i != j:
					noise_model.add_quantum_error(pauliErrorTwo, g, [i,j], warnings=False)
					noise_model.add_quantum_error(depolarizingErrorTwo, g, [i,j], warnings=False)
	return noise_model

def construct_noise_model_readoutOnly(numQubits, measurement_error_prob1_given_0 = MEASUREMENT_ERROR_PROB_1_given_0,
							   measurement_error_prob0_given_1 = MEASUREMENT_ERROR_PROB_0_given_1, **kwargs):
	noise_model = NoiseModel()
	noise_model.add_all_qubit_readout_error(ReadoutError([
		[1 - measurement_error_prob1_given_0, measurement_error_prob1_given_0],
		[measurement_error_prob0_given_1, 1 - measurement_error_prob0_given_1]
	]), warnings=False)
	return noise_model

def construct_noise_model_thermalOnly(num_qubits,
								  two_qubit_gates = TWO_QUBIT_GATES, one_qubit_gates = ONE_QUBIT_GATES,
								  other_one_qubit_ops = OTHER_ONE_QUBIT_OPS, op_times = DEFAULT_OP_TIMES,
								  t1 = T1, t2 = T2, thermal_population_1 = THERMAL_POPULATION_1, **kwargs):
	noise_model = NoiseModel()
	for g in one_qubit_gates + other_one_qubit_ops:
		noise_model.add_all_qubit_quantum_error(thermal_relaxation_error(t1, t2, op_times[g], thermal_population_1), g, warnings=False)
	for g in two_qubit_gates:
		thermalErrorTwo = thermal_relaxation_error(t1, t2, op_times[g], thermal_population_1).expand(thermal_relaxation_error(t1,t2,op_times[g],thermal_population_1))
		for i in range(num_qubits):
			for j in range(num_qubits):
				if i != j:
					noise_model.add_quantum_error(thermalErrorTwo, g, [i,j], warnings=False)
	return noise_model

def construct_modified_circuit_for_thermal_noise(circ, op_times = DEFAULT_OP_TIMES, ops = ALL_OPS):
	"""
	Returns a circuit padded with identities for use with thermal noise.

	Qiskit does not apply thermal noise to "empty" wires.
	We need to fill these wires with identity gates and apply noise to those identities.

	We unroll the circuit into the basic operations in `ops`, and apply identities according to the
	passed `op_times`, using the `idle_scheduler` class.

	`idle_scheduler` is available at https://github.com/gadial/qiskit-aer/blob/ff56889c3cf0486b1ad094634e88d7e756b6db3c/qiskit/providers/aer/noise/utils/idle_scheduler.py
	"""
	urPass = Unroller(ops)
	pm = PassManager(urPass)
	circ_unrolled = pm.run(circ)

	circ_withId = schedule_idle_gates(circ_unrolled, op_times)
	return circ_withId


########################################
###### H-Test Helper Functions #########
########################################
def hTest_computeAnswer(counts):
	"""
	Computes the expectation value from the Hadamard-test.

	`counts` should be the dictionary returned by qiskit of the form {state1:count1, state2:count2, ...}.
	We assume that the first qubit is the output qubit,
	and so the final output bit in each state_i is the output bit (since IBM prints little-endian).
	"""
	results = []
	total = 0
	totalShots = 0
	for state, count in counts.items():
		total += count * (-1) ** int(state[-1]) # IBM prints little-endian, i.e. c[0] is on RHS
		totalShots += count
	expectation = total / totalShots
	return expectation

def hTest_postSelectedCounts(n, counts, k = 1):
	"""
	Returns `counts` with some entries deleted according to postselection described in paper [SCC19].

	[SCC19] is "Entanglement spectroscopy with a depth-two quantum circuit".

	`k` is such that rho is a state on 2k qubits, and rho_A is k qubits.

	`counts` should be the dictionary returned by qiskit of the form {state1:count1, state2:count2, ...}.
	We assume that the first qubit is the output qubit, and that the registers are of the form
	rho1A, rho1B, rho2A, ..., each k qubits.
	And because IBM is little-endian, the bits in each state_i in counts will be the reverse order.
	"""
	goodCounts = {}
	for state, count in counts.items():
		trimmedState = state.replace(" ", "") # If used many classical regs, it added whitespace
		aset = set(trimmedState[-(2*k*j + j + 1)] for j in range(k) for i in range(n))
		bset = set(trimmedState[-(2*k*j + j + k + 1)] for j in range(k) for i in range(n))
		if trimmedState[-1] != 1 or aset != bset:
			goodCounts.update({state:count})
	return goodCounts

def hTest_computeUncertainty(numShots, confidenceLevel = 0.68):
	"""
	Computes a +- error value for the Hadamard-test according to some confidence level.

	We use Hoeffding's inequality for the error in computing the expectation value of a single
	random variable according to a certain confidenceLevel in that error interval.
	"""
	epsilon = np.sqrt(np.log((1 - confidenceLevel) / 2) / (-2 * numShots))
	return 2 * epsilon

def hTest_computeErrorBars(numShots, answers, confidenceLevel):
	"""
	Returns a nested list of the form [lowerErrorBarValues, upperErrorBarValues] for use in pyplot.

	The upperErrors exactly equal the value computed by hTest_computeUncertainty.

	The lowerErrors are the same, with some post processing:
	If answer is negative, we give a lower uncertainty of 0.
	If the answer-error is less than zero, we give a lower uncertainty equal to the answer.
	"""
	upperErrors = [hTest_computeUncertainty(numShots, confidenceLevel) for ans in answers]
	lowerErrors = []
	for ans, error in zip(answers, upperErrors):
		if ans < 0:
			lowerErrors.append(0)
		elif ans - error < 0:
			lowerErrors.append(ans)
		else:
			lowerErrors.append(error)
	return [lowerErrors, upperErrors]

##################################
###### H-Test (original) #########
##################################
def hTest_original_circuit(n, k, prep_state):
	"""
	Implements Hadamard test with cyclic Permutation gate on A subsystems for Tr(rho_A^n)

	`k` is such that rho is a state on 2k qubits, and rho_A is k qubits.
	`prep_state` should be a gate or instruction that accepts a single register of 2k qubits,
	ordered with subsystem A then subsystem B qubits, which prepares the desired state rho.
	"""
	cswap_custom = generate_cswap_instruction(k)

	qCont = QuantumRegister(1)
	qregs = [[QuantumRegister(k, "qa_" + str(i+1)), QuantumRegister(k, "qb_" + str(i+1))] for i in range(n)]
	cCont = ClassicalRegister(1)
	cregs = [[ClassicalRegister(k, "a_" + str(i+1)), ClassicalRegister(k, "b_" + str(i+1))] for i in range(n)]
	circ = QuantumCircuit(qCont, cCont)
	for qpair in qregs:
		for qr in qpair:
			circ.add_register(qr)
	for cpair in cregs:
		for cr in cpair:
			circ.add_register(cr)

	for qpair in qregs:
		circ.append(prep_state, mergeRegisters(qpair[0], qpair[1]))

	circ.h(qCont)

	for i in range(n-1):
		circ.append(cswap_custom, mergeRegisters(qCont, qregs[i][0], qregs[i+1][0]))

	circ.h(qCont)

	circ.measure(qCont, cCont)
	for i in range(n):
		circ.measure(qregs[i][0], cregs[i][0])
		circ.measure(qregs[i][1], cregs[i][1])

	return circ

#########################################################################################
###### H-Test, Qubit efficient, 4k+1 Qubits #######
#########################################################################################
def hTest_qubitEfficient4k_circuit(n, k, prep_state):
	"""
	Implements a H-test for Tr(rho_A^n) using 4 registers and 1 ancilla.

	This implementation never resets two of those registers.

	`k` is such that rho is a state on 2k qubits, and rho_A is k qubits.
	`prep_state` should be a gate or instruction that accepts a single register of 2k qubits,
	ordered with subsystem A then subsystem B qubits, which prepares the desired state rho.
	"""
	cswap_custom = generate_cswap_instruction(k)

	contQ = QuantumRegister(1, "q0")
	contC = ClassicalRegister(1, "c0")
	qlabels = ["topA", "topB", "bottomA", "bottomB"]
	qregs = [QuantumRegister(k, label) for label in qlabels]
	circ = QuantumCircuit(contQ, contC)
	for qr in qregs:
		circ.add_register(qr)

	topReg = mergeRegisters(qregs[0], qregs[1])
	bottomReg = mergeRegisters(qregs[2], qregs[3])
	swapRegs = mergeRegisters((contQ), qregs[0], qregs[2])

	circ.h(contQ)

	circ.append(prep_state, topReg) # Prep copy 1 in top register
	for i in range(n - 1):
		circ.append(prep_state, bottomReg) # Prep copy in bottom register
		circ.append(cswap_custom, swapRegs) # Swap the A partitions of each register
		circ.reset(qregs[2]) # Reset the bottom register
		circ.reset(qregs[3])
	# The final reset is unnecessary, but doesn't hurt anything

	circ.h(contQ)
	circ.measure(contQ, contC)

	return circ

#########################################################################################
###### H-Test, Qubit efficient, 3k+1 Qubits #######
#########################################################################################
def hTest_qubitEfficient3k_circuit(n, k, prep_state):
	"""
	Implements a H-test for Tr(rho_A^n) using 3 registers and 1 ancilla.

	This implementation never resets two of those registers

	`k` is such that rho is a state on 2k qubits, and rho_A is k qubits.
	`prep_state` should be a gate or instruction that accepts a single register of 2k qubits,
	ordered with subsystem A then subsystem B qubits, which prepares the desired state rho.
	"""
	cswap_custom = generate_cswap_instruction(k)

	contQ = QuantumRegister(1, "q0")
	contC = ClassicalRegister(1, "c0")
	qlabels = ["reg0", "reg1", "reg2"]
	qregs = [QuantumRegister(k, label) for label in qlabels]
	circ = QuantumCircuit(contQ, contC)
	for qr in qregs:
		circ.add_register(qr)

	topRegs = mergeRegisters(qregs[0], qregs[1])
	bottomRegs = mergeRegisters(qregs[1], qregs[2])
	swapRegs = mergeRegisters((contQ), qregs[0], qregs[1])

	circ.h(contQ)

	circ.append(prep_state, topRegs) # Prep copy 1 in top two registers
	circ.reset(qregs[1])
	for i in range(n - 1):
		circ.append(prep_state, bottomRegs) # Prep copy in bottom registers
		circ.append(cswap_custom, swapRegs) # Swap the A partitions of each register
		circ.reset(qregs[1])
		circ.reset(qregs[2])
	# The final resets are unnecessary, but don't hurt anything

	circ.h(contQ)
	circ.measure(contQ, contC)

	return circ

################################################################################
###### H-Test, Qubit Efficient, Alternative implementation, 4k+1 Qubits #######
################################################################################
def hTest_qubitEfficient_alternative4k_circ(n, k, prep_state):
	"""
	Implements a qubit efficient Hadamard test for Tr(rho_A^n) using 4 registers and 1 ancilla

	This implementation alternates which registers it resets.
	In our noisy simulations, it actually performs worse than the other 4k circuit.

	`k` is such that rho is a state on 2k qubits, and rho_A is k qubits.
	`prep_state` should be a gate or instruction that accepts a single register of 2k qubits,
	ordered with subsystem A then subsystem B qubits, which prepares the desired state rho.
	"""
	cswap_custom = generate_cswap_instruction(k)

	contQ = QuantumRegister(1, "q0")
	contC = ClassicalRegister(1, "c0")
	qlabels = ["topA", "topB", "bottomA", "bottomB"]
	qregs = [QuantumRegister(k, label) for label in qlabels]
	circ = QuantumCircuit(contQ, contC)
	for qr in qregs:
		circ.add_register(qr)

	topReg = mergeRegisters(qregs[0], qregs[1])
	bottomReg = mergeRegisters(qregs[2], qregs[3])
	swapRegs = mergeRegisters(contQ, qregs[0], qregs[2])

	circ.h(contQ)

	circ.append(prep_state, topReg) # Prep copy 1 in top register
	circ.append(prep_state, bottomReg) # Prep copy 2 in bottom register
	circ.append(cswap_custom, swapRegs) # Swap the A partitions of each register
	for i in range(3, n, 2):
		circ.reset(qregs[0]) # Reset the top register
		circ.reset(qregs[1])
		circ.append(prep_state, topReg) # Prep copy i in top register
		circ.append(cswap_custom, swapRegs)

		circ.reset(qregs[2]) # Reset the bottom register
		circ.reset(qregs[3])
		circ.append(prep_state, bottomReg) # Prep copy i+1 in bottom register
		circ.append(cswap_custom, swapRegs)
	if n % 2 == 1:
		circ.reset(qregs[0])
		circ.reset(qregs[1])
		circ.append(prep_state, topReg) # Prep copy n in top register
		circ.append(cswap_custom, swapRegs)
	# We skip the final reset. It's unnecessary

	circ.h(contQ)
	circ.measure(contQ, contC)

	return circ

################################################################################
###### H-Test, Qubit Efficient, alternative implementation, 3k+1 Qubits #######
################################################################################
def hTest_qubitEfficient_alternative3k_circ(n, k, prep_state):
	"""
	Implements a qubit efficient Hadamard test for Tr(rho_A^n) using 3 registers and 1 ancilla

	This implementation alternates which registers it resets.
	In our noisy simulations, it actually performs worse than the other 3k circuit.

	`k` is such that rho is a state on 2k qubits, and rho_A is k qubits.
	`prep_state` should be a gate or instruction that accepts a single register of 2k qubits,
	ordered with subsystem A then subsystem B qubits, which prepares the desired state rho.
	"""
	cswap_custom = generate_cswap_instruction(k)

	contQ = QuantumRegister(1, "q0")
	contC = ClassicalRegister(1, "c0")
	qlabels = ["reg0", "reg1", "reg2"]
	qregs = [QuantumRegister(k, label) for label in qlabels]
	circ = QuantumCircuit(contQ, contC)
	for qr in qregs:
		circ.add_register(qr)

	circ.h(contQ)

	circ.append(prep_state, mergeRegisters(qregs[0],qregs[1])) # Prep copy 1 in reg0,reg1
	circ.reset(qregs[1]) # Reset reg1, tracing out psi_1_B
	circ.append(prep_state, mergeRegisters(qregs[1],qregs[2])) # Prep copy 2 in reg1,reg2
	circ.append(cswap_custom, mergeRegisters((contQ),qregs[0],qregs[1]))
	for i in range(3, n, 2):
		circ.reset(qregs[2]) # Trace out psi_i-1_B
		circ.reset(qregs[0]) # Trace out psi_i-2_A, so all of psi_i-2 is gone
		circ.append(prep_state, mergeRegisters(qregs[0], qregs[2])) # Prep psi_i in reg0,reg2
		circ.append(cswap_custom, mergeRegisters((contQ),qregs[0], qregs[1]))

		circ.reset(qregs[2]) # Trace out psi_i_B
		circ.reset(qregs[1]) # Trace out psi_i-1_A, so all of psi_i-1 is gone
		circ.append(prep_state, mergeRegisters(qregs[1], qregs[2])) # Prep psi_i+1 in reg1,2
		circ.append(cswap_custom, mergeRegisters((contQ),qregs[0], qregs[1]))
	if n % 2 == 1:
		circ.reset(qregs[2])
		circ.reset(qregs[0])
		circ.append(prep_state, mergeRegisters(qregs[0], qregs[2])) # Prep psi_n
		circ.append(cswap_custom, mergeRegisters((contQ),qregs[0], qregs[1]))
		# We skip the final reset. It's unnecessary

	circ.h(contQ)
	circ.measure(contQ, contC)

	return circ


###############################################
###### Two-Copy Test Helper Functions #########
###############################################
def twoCopyTest_computeAnswer(n, k, counts):
	"""
	Computes Tr(rho_A^n) using the results of the two-copy test

	`k` is such that rho is a state on 2k qubits, and rho_A is k qubits.

	`counts` should be the dictionary returned by qiskit of the form {state1:count1, state2:count2, ...}.
	The qubits should be ordered by copies rho1, rho2, ..., rho2n, with each rho_i ordered with
	the A qubits before the B qubits.
	Then, because IBM is little-endian, the classical bits in each state_i should be in the opposite order.

	This calculation requires taking a square root. If the value before taking the square root is negative
	(due to noise in the simulation), then we return -square_root(|value|).
	"""
	total = 0
	totalShots = 0
	qubitsPerNCopies = 2 * n * k
	for state, count in counts.items():
		trimmedState = state.replace(" ", "") # Using many classical regs added whitespace
		value = 1
		for i in range(qubitsPerNCopies):
			index1 = 2 * qubitsPerNCopies - 1 - i # IBM prints little-endian, i.e. c[0] is on RHS
			index2 = (qubitsPerNCopies - 1) - ((2 * k + i) % (qubitsPerNCopies))
			value *= (-1)**(int(trimmedState[index1]) * int(trimmedState[index2]))
		total += count * value
		totalShots += count
	expectation = total / totalShots
	if expectation < 0:
		finalValue = -np.sqrt(abs(expectation))
	else:
		finalValue = np.sqrt(expectation)
	return finalValue

def twoCopyTest_computeLowerErrorBound(numShots, answer, confidenceLevel = 0.68):
	"""
	Computes the difference between answer and the smallest ideal answer we would expect given numShots

	We use Hoeffding's inequality to find the potential error in computing the expectation value of a single
	random variable such that we can have `confidenceLevel` in that error bound.

	If the `answer` is negative, then we return 0 lower uncertainty.
	If the `answer` is positive and the value of answer after error has been factored in is negative,
	then we return a lower uncertainty that equals the answer (so the error bar extends down to 0).
	"""
	rawAnsError = 2 * np.sqrt(np.log((1 - confidenceLevel) / 2) / (-2 * numShots))

	rawAns = answer ** 2
	sign = np.sign(answer)

	lowerRawAns = sign*rawAns - rawAnsError
	if answer <= 0:
		return 0
	elif lowerRawAns <= 0:
		return answer
	else:
		return abs(answer - np.sqrt(lowerRawAns))

def twoCopyTest_computeUpperErrorBound(numShots, answer, confidenceLevel = 0.68):
	"""
	Computes the difference between answer and the largest ideal answer we would expect given numShots

	We use Hoeffding's inequality to find the potential error in computing the expectation value of a single
	random variable such that we can have `confidenceLevel` in that error bound.

	In this calculation, we undo the square root from `twoCopyTest_computeAnswer` (keeping the same sign as `answer`)
	and then add error to the raw value.
	If this raw value with error factored in is negative (which implies `answer` was also negative to begin with),
	then we proceed as in `twoCopyTest_computeAnswer` and compute -square_root(|value|).
	"""
	rawAnsError = 2 * np.sqrt(np.log((1 - confidenceLevel) / 2) / (-2 * numShots))

	rawAns = answer ** 2
	sign = np.sign(answer)

	upperRawAns = sign * rawAns + rawAnsError
	if upperRawAns >= 0:
		return abs(answer - np.sqrt(upperRawAns))
	else:
		return abs(abs(answer) - np.sqrt(abs(upperRawAns)))

def twoCopyTest_computeErrorBars(numShots, answers, confidenceLevel):
	"""
	Returns a nested list of the form [lowerErrorBarValues, upperErrorBarValues] for use in pyplot.
	"""
	lowerErrors = [twoCopyTest_computeLowerErrorBound(numShots, ans, confidenceLevel) for ans in answers]
	upperErrors = [twoCopyTest_computeUpperErrorBound(numShots, ans, confidenceLevel) for ans in answers]
	return [lowerErrors, upperErrors]

#######################################
###### Two-copy Test (original) #######
#######################################
def twoCopyTest_original_circuit(n, k, prep_state):
	"""
	Implements the Two-copy test for Tr(rho_A^n)

	`k` is such that rho is a state on 2k qubits, and rho_A is k qubits.
	`prep_state` should be a gate or instruction that accepts a single register of 2k qubits,
	ordered with subsystem A then subsystem B qubits, which prepares the desired state rho.
	"""
	qregs1 = [[QuantumRegister(k, "qa_" + str(i+1)), QuantumRegister(k, "qb_" + str(i+1))] for i in range(n)]
	qregs2 = [[QuantumRegister(k, "qa2_" + str(i+1)), QuantumRegister(k, "qb2_" + str(i+1))] for i in range(n)]
	cregs1 = [[ClassicalRegister(k, "a_" + str(i+1)), ClassicalRegister(k, "b_" + str(i+1))] for i in range(n)]
			  # For states 1[A,B], 2[A,B], ..  n[A,B]
	cregs2 = [[ClassicalRegister(k, "a2_" + str(i+1)), ClassicalRegister(k, "b2_" + str(i+1))] for i in range(n)]
			  # For states 1'[A,B], 2'[A,B], .. n'[A,B]
	circ = QuantumCircuit()
	for qpair in qregs1:
		for qr in qpair:
			circ.add_register(qr)
	for qpair in qregs2:
		for qr in qpair:
			circ.add_register(qr)
	for cpair in cregs1:
		for cr in cpair:
			circ.add_register(cr)
	for cpair in cregs2:
		for cr in cpair:
			circ.add_register(cr)

	for i in range(n):
		circ.append(prep_state, mergeRegisters(qregs1[i][0], qregs1[i][1]))
	for i in range(n):
		circ.append(prep_state, mergeRegisters(qregs2[i][0], qregs2[i][1]))

	for i in range(n):
		circ.cx(qregs1[i][0], qregs2[i-1][0]) # CNOT(i_A, i-1'_A)
		circ.cx(qregs1[i][1], qregs2[i][1]) # CNOT(i_B, i'_B)

	for qpair in qregs1:
		for qr in qpair:
			  circ.h(qr)

	for i in range(n):
		circ.measure(qregs1[i][0], cregs1[i][0])
		circ.measure(qregs1[i][1], cregs1[i][1])
	for i in range(n):
		circ.measure(qregs2[i][0], cregs2[i][0])
		circ.measure(qregs2[i][1], cregs2[i][1])

	return circ

########################################################
###### Two-copy test, Qubit efficient, 6k Qubits #######
########################################################
def twoCopyTest_qubitEfficient6k_circuit(n, k, prep_state):
	"""
	Implements a qubit efficient Two-copy test for Tr(rho_A^n) using 6 registers

	`k` is such that rho is a state on 2k qubits, and rho_A is k qubits.
	`prep_state` should be a gate or instruction that accepts a single register of 2k qubits,
	ordered with subsystem A then subsystem B qubits, which prepares the desired state rho.
	"""
	qlabels = ["reg0", "reg1", "reg2", "reg3", "reg4", "reg5"]
	qregs = [QuantumRegister(k, label) for label in qlabels]
	cregs1 = [[ClassicalRegister(k, "a_" + str(i+1)), ClassicalRegister(k, "b_" + str(i+1))] for i in range(n)]
		# For states 1[A,B], 2[A,B], ..  n[A,B]
	cregs2 = [[ClassicalRegister(k, "a2_" + str(i+1)), ClassicalRegister(k, "b2_" + str(i+1))] for i in range(n)]
		# For states 1'[A,B], 2'[A,B], .. n'[A,B]
	circ = QuantumCircuit()
	for qr in qregs:
		circ.add_register(qr)
	for cpair in cregs1:
		for cr in cpair:
			circ.add_register(cr)
	for cpair in cregs2:
		for cr in cpair:
			circ.add_register(cr)

	circ.append(prep_state, mergeRegisters(qregs[0],qregs[1])) # Prep copy 1'
	circ.append(prep_state, mergeRegisters(qregs[2],qregs[3])) # Prep copy 1
	circ.append(prep_state, mergeRegisters(qregs[4],qregs[5])) # Prep copy n'
	circ.cx(qregs[2], qregs[4]) # CNOT(1_A, n'_A)
	circ.cx(qregs[3], qregs[1]) # CNOT(1_B, 1'_B)
	circ.h(qregs[2])
	circ.h(qregs[3])
	circ.measure(qregs[2], cregs1[0][0]) # Measure psi_1
	circ.measure(qregs[3], cregs1[0][1])
	circ.measure(qregs[1], cregs2[0][1]) # Measure psi_1'_B
	circ.measure(qregs[4], cregs2[n - 1][0]) # Measure psi_n'_A
	circ.reset(qregs[2])
	circ.reset(qregs[3])
	circ.reset(qregs[1])
	circ.reset(qregs[4])
	for i in range(2, n // 2 + 1):
		# Each iteration will prep copies i, i', n-i+2, n-i+1',
		# and will interact with copies i-1'_A, n-i+2'_B, prepared in previous iteration.
		circ.append(prep_state, mergeRegisters(qregs[1],qregs[2])) # Prep copy i
		circ.append(prep_state, mergeRegisters(qregs[3],qregs[4])) # Prep copy n-i+2
		circ.cx(qregs[1],qregs[0]) # CNOT(i_A, i-1'_A)
		circ.h(qregs[1])
		circ.cx(qregs[4],qregs[5]) # CNOT(n-i+2_B, n-i+2'_B)
		circ.h(qregs[4])
		circ.measure(qregs[0], cregs2[i - 1 - 1][0]) # i-1'_A
		circ.measure(qregs[1], cregs1[i - 1][0]) # i_A
		circ.measure(qregs[4], cregs1[n - i + 2 - 1][1]) # n-i+2_B
		circ.measure(qregs[5], cregs2[n - i + 2 - 1][1]) # n-i+2'_B
		circ.reset(qregs[0])
		circ.reset(qregs[1])
		circ.reset(qregs[4])
		circ.reset(qregs[5])

		circ.append(prep_state, mergeRegisters(qregs[0],qregs[1])) # Prep copy i'
		circ.append(prep_state, mergeRegisters(qregs[4],qregs[5])) # Prep copy n-i+1'
		circ.cx(qregs[2],qregs[1]) # CNOT(i_B, i'_B)
		circ.h(qregs[2])
		circ.cx(qregs[3],qregs[4]) # CNOT(n-i+2_A, n-i+1'_A)
		circ.h(qregs[3])
		circ.measure(qregs[1], cregs2[i - 1][1]) # i'_B
		circ.measure(qregs[2], cregs1[i - 1][1]) #  i_B
		circ.measure(qregs[3], cregs1[n - i + 2 - 1][0]) # n-i+2_A
		circ.measure(qregs[4], cregs2[n - i + 1 - 1][0]) # n-i+1'_A
		circ.reset(qregs[1])
		circ.reset(qregs[2])
		circ.reset(qregs[3]) # These last 2 resets are unnecessary in last iteration if N even
		circ.reset(qregs[4])
	if n % 2 == 0:
		# If N even, then final iteration leaves copies i'_A, n-i+1'_B = n/2'_A, n/2+1'_B
		# behind. We now load a copy n/2+1, which matches with the above and we are done.
		# i'_A is in qregs[0]
		# n-i+1'_B is in qregs[5]
		circ.append(prep_state, mergeRegisters(qregs[1],qregs[2])) # Prep copy n/2+1
		circ.cx(qregs[1],qregs[0]) # CNOT(n/2+1_A, n/2'_A)
		circ.h(qregs[1])
		circ.cx(qregs[2],qregs[5]) # CNOT(n/2+1_B, n/2+1'_B)
		circ.h(qregs[2])
		circ.measure(qregs[0], cregs2[n // 2 - 1][0]) # n/2'_A
		circ.measure(qregs[1], cregs1[n // 2 + 1 - 1][0]) # n/2+1_A
		circ.measure(qregs[2], cregs1[n // 2 + 1 - 1][1]) # n/2+1_B
		circ.measure(qregs[5], cregs2[n // 2 + 1 - 1][1]) # n/2+1'_B
	else:
		# If N odd, then final iteration leaves floor(n/2)'_A, ceil(n/2)+1'_B.
		# We now load ceil(n/2), ceil(n/2)+1, and ceil(n/2)', and we are done.
		circ.append(prep_state, mergeRegisters(qregs[1],qregs[2])) # Prep copy ceil(n/2)
		circ.append(prep_state, mergeRegisters(qregs[3],qregs[4])) # Prep copy ceil(n/2)+1
		circ.cx(qregs[1],qregs[0]) # CNOT(ceil(n/2)_A, floor(n/2)'_A)
		circ.h(qregs[1])
		circ.cx(qregs[4], qregs[5]) #CNOT(ceil(n/2)+1_B, ceil(n/2)+1'_B)
		circ.h(qregs[4])
		circ.measure(qregs[0], cregs2[n // 2 - 1][0]) # floor(n/2)'_A
		circ.measure(qregs[1], cregs1[n // 2 + 1 - 1][0]) # ceil(n/2)_A
		circ.measure(qregs[4], cregs1[n // 2 + 2 - 1][1]) # ceil(n/2)+1_B
		circ.measure(qregs[5], cregs2[n // 2 + 2 - 1][1]) # ceil(n/2)+1'_B
		circ.reset(qregs[0])
		circ.reset(qregs[1])
		# Resetting 4,5 would be unnecessary

		circ.append(prep_state, mergeRegisters(qregs[0],qregs[1])) # Prep copy ceil(n/2)'
		circ.cx(qregs[2], qregs[1]) # CNOT(ceil(n/2)_B, ceil(n/2)'_B)
		circ.h(qregs[2])
		circ.cx(qregs[3], qregs[0]) # CNOT(ceil(n/2)+1_A, ceil(n/2)'_A)
		circ.h(qregs[3])
		circ.measure(qregs[0], cregs2[n // 2 + 1 - 1][0]) # ceil(n/2)'_A
		circ.measure(qregs[1], cregs2[n // 2 + 1 - 1][1]) # ceil(n/2)'_B
		circ.measure(qregs[2], cregs1[n // 2 + 1 - 1][1]) # ceil(n/2)_B
		circ.measure(qregs[3], cregs1[n // 2 + 2 - 1][0]) # ceil(n/2)+1_A

	return circ

########################################################
###### Two-copy Test, Qubit efficient, 4k Qubits #######
########################################################
def twoCopyTest_qubitEfficient4k_circuit(n, k, prep_state):
	"""
	Implements a qubit efficient Two-copy test for Tr(rho_A^n) using 4 registers

	`k` is such that rho is a state on 2k qubits, and rho_A is k qubits.
	`prep_state` should be a gate or instruction that accepts a single register of 2k qubits,
	ordered with subsystem A then subsystem B qubits, which prepares the desired state rho.
	"""
	qlabels = ["reg0", "reg1", "reg2", "reg3"]
	qregs = [QuantumRegister(k, label) for label in qlabels]
	cregs1 = [[ClassicalRegister(k, "a_"+ str(i+1)), ClassicalRegister(k, "b_" + str(i+1))] for i in range(n)]
		# For states 1[A,B], 2[A,B], ..  n[A,B]
	cregs2 = [[ClassicalRegister(k, "a2_" + str(i+1)), ClassicalRegister(k, "b2_" + str(i+1))] for i in range(n)]
		# For states 1'[A,B], 2'[A,B], .. n'[A,B]
	circ = QuantumCircuit()
	for qr in qregs:
		circ.add_register(qr)
	for cpair in cregs1:
		for cr in cpair:
			circ.add_register(cr)
	for cpair in cregs2:
		for cr in cpair:
			circ.add_register(cr)

	circ.append(prep_state, mergeRegisters(qregs[0],qregs[1])) # Prep copy 1
	circ.append(prep_state, reverseMergeRegisters(qregs[2],qregs[3])) # Prep copy 1' - *upside down*
	circ.cx(qregs[1],qregs[2]) # CNOT(1_B, 1'_B)
	circ.h(qregs[1])
	circ.measure(qregs[1], cregs1[0][1]) # Measure psi_1_B
	circ.measure(qregs[2], cregs2[0][1]) # Measure psi_1'_B
	circ.reset(qregs[1])
	circ.reset(qregs[2])
	for i in range(2, (n + 1) // 2 + 1): # i = 2 .. ceil(N/2) + 1
		# Each iteration will prep copies i, i', n-i+2, n-i+2',
		# and will interact with copies i-1'_A and n-i+3_A (rollsover to 1_A for i=2), prepared in previous iteration.
		circ.append(prep_state, mergeRegisters(qregs[1],qregs[2])) # Prep copy n-i+2'
		circ.cx(qregs[0],qregs[1]) # CNOT(n-i+3_A, n-i+2'_A)
		circ.h(qregs[0])
		circ.measure(qregs[0], cregs1[((n - i + 3) % n) - 1][0])
			# Measure n-i+3_A (rollsover to 1_A for i=2)
		circ.measure(qregs[1], cregs2[n - i + 2 - 1][0]) # n-i+2'_A
		circ.reset(qregs[0])
		circ.reset(qregs[1])

		circ.append(prep_state, mergeRegisters(qregs[0],qregs[1])) # Prep copy i
		circ.cx(qregs[0],qregs[3]) # CNOT(i_A, i-1'_A)
		circ.h(qregs[0])
		circ.measure(qregs[0], cregs1[i - 1][0]) # i_A
		circ.measure(qregs[3], cregs2[i - 1 - 1][0]) # i-1'_A
		circ.reset(qregs[0])
		circ.reset(qregs[3])

		circ.append(prep_state, mergeRegisters(qregs[0],qregs[3])) # Prep copy n-i+2
		circ.cx(qregs[3],qregs[2]) # CNOT(n-i_2_B, n-i+2'_B)
		circ.h(qregs[3])
		circ.measure(qregs[2], cregs2[n - i + 2 - 1][1]) # n-i+2'_B
		circ.measure(qregs[3], cregs1[n - i + 2 - 1][1]) # n-i+2_B
		circ.reset(qregs[2])
		circ.reset(qregs[3])

		circ.append(prep_state, reverseMergeRegisters(qregs[2],qregs[3])) # Prep copy i' - *upside down*
		circ.cx(qregs[1],qregs[2]) # CNOT(i_B, i'_B)
		circ.h(qregs[1])
		circ.measure(qregs[1], cregs1[i - 1][1]) # i_B
		circ.measure(qregs[2], cregs2[i - 1][1]) # i'_B
		circ.reset(qregs[1]) # These two resets are unnecessary in last iteration if N odd
		circ.reset(qregs[2])
	if n % 2 == 0:
		# If N even, then final iteration leaves n/2+2_A on [0] and n/2'_A on [3]
		circ.append(prep_state, mergeRegisters(qregs[1],qregs[2])) # Prep copy n/2+1'
		circ.cx(qregs[0], qregs[1]) # CNOT(n/2+2_A, n/2+1'_A)
		circ.h(qregs[0])
		if n == 2: # Special case, where n/2+2 wraps around to 1.
			circ.measure(qregs[0], cregs1[0][0]) # 1_A
			circ.measure(qregs[1], cregs2[1][0]) # 2'_A
		else:
			circ.measure(qregs[0], cregs1[n // 2 + 2 - 1][0]) # n/2+2_A
			circ.measure(qregs[1], cregs2[n // 2 + 1 - 1][0]) # n/2+1'_A
		circ.reset(qregs[0])
		circ.reset(qregs[1])

		circ.append(prep_state, mergeRegisters(qregs[0],qregs[1])) # Prep copy n/2+1
		circ.cx(qregs[0], qregs[3]) # CNOT(n/2+1_A, n/2'_A)
		circ.h(qregs[0])
		circ.measure(qregs[0], cregs1[n // 2 + 1 - 1][0]) # n/2+1_A
		circ.measure(qregs[3], cregs2[n // 2 - 1][0]) # n/2'_A
		# No more resets necessary

		circ.cx(qregs[1], qregs[2]) # CNOT(n/2+1_B, n/2+1'_B)
		circ.h(qregs[1])
		circ.measure(qregs[1], cregs1[n // 2 + 1 - 1][1]) # n/2+1_B
		circ.measure(qregs[2], cregs2[n // 2 + 1 - 1][1]) # n/2+1'_B
	else:
		# If N odd, then final iteration leaves floor(n/2)+2_A on [0] and ceil(N/2)'_A on [3]
		circ.cx(qregs[0],qregs[3]) # CNOT(floor(n/2)+2_A, ceil(N/2)'_A)
		circ.h(qregs[0])
		circ.measure(qregs[0], cregs1[n // 2 + 2 - 1][0]) # floor(n/2)+2_A
		circ.measure(qregs[3], cregs2[n // 2 + 1 - 1][0]) # ceil(n/2)'_A
		# No reset necessary

	return circ
