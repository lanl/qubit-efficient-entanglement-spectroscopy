"""
This file runs simulations of various entanglement spectroscopy circuits using
qiskit and saves and plots the results.

See https://arxiv.org/abs/2010.03080

Authors:
	Justin Yirka	yirka@utexas.edu
	Yigit Subasi	ysubasi@lanl.gov
"""

"""
This file is an example of how to use the module entSpectroscopy.py.
Most of the experiments for https://arxiv.org/abs/2010.03080 were run using scripts
like this.

You can run it from the command line like:
    python simulateCircuitsScript.py 2 20 1 1000 0.68 -f './folder'
or
    python3 simulateCircuitsScript.py 2 6 1 20000 0.68 0.95 -f './folder' $'Graph Title' 1 5 3 2 800 800 10**(-7) 0.01 0.0005 0.0025 0 &
Change the parameters after `simulateCircuitsScript.py` based on the description below.

This script needs `entSpectroscopy.py` and `idle_scheduler.py` available to import.
`idle_scheduler` is available at https://github.com/gadial/qiskit-aer/blob/ff56889c3cf0486b1ad094634e88d7e756b6db3c/qiskit/providers/aer/noise/utils/idle_scheduler.py

You can adjust the number of qubits and the paramters for the noise model from the command line, like above.
Other changes will require modifying this script. Again, this is just one example of using entSpectroscopy.

If you want to change which circuits are simulated, then you'll have to edit the tuple `all_circuits`
in this file.

If you want a noise model different than the models provided in entSpectroscopy, then you'll have
to write you own.
If you want to simulate the circuits on different quantum states than the default state provided in
entSpectroscopy, then you'll have to write you own function.
"""

"""
Command line arguments:
    1: Min n
    2: Max n (exclusive, i.e. we compute up to maxN - 1)
    3: k (size of rho_A) (entSpectroscopy is really only defined for k=1 right now)
    4: Number of shots
    5: Confidence level for error bars (as a decimal)
    6: Noise choice: -f -t -r -g or -n
        for full noise model, thermal noise only, readout noise only, gate and readout noise only, or no noise.
        The most general is -f.
        -t and -r are for convenience. You could achieve -r with -f if you set many
        parameters to 0, but this makes it easier.
    7: Output directory (what folder should the output files be placed in?) (don't include a slash at the end)

Optional:
    8: Plot subtitle

Optional, but if give one, must give all:
    If none of these are given, then we use the default error parameters listed in `entSpectroscopy.py`.
    Note that while we've made this to be flexible, we have made assumptions; for example, this script
    only takes 1 parameter for readout error, assuming that 1 should flip to 0 with the same probability
    that 0 flips to 1. If you want more customization... write your own script.

    8: Single qubit gate time
    10: CX time
    11: Measure time
    12: Reset time
    13: T1
    14: T2
    15: Thermal Population
    16: Readout error probability
    17: Single qubit gate error
    18: CX gate error
    19: Reset gate error
"""

######################
###### Imports #######
######################
import sys, os
from time import perf_counter
import numpy as np
from scipy.stats import linregress
from scipy.stats import t

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from entSpectroscopy import *

from qiskit import execute
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import thermal_relaxation_error, ReadoutError

###########################################
###### Command Line Args/Parameters #######
###########################################
args = sys.argv
N_LIST = range(int(args[1]), int(args[2]))
K = int(args[3])
SHOTS = int(args[4])
CONFIDENCE_LEVEL = float(args[5])
NOISE_MODEL_CHOICE = args[6]
OUTPUT_DIRECTORY = args[7]
if len(args) >= 9:
    PLOT_SUBTITLE_STRING = "\n" + args[8]
else:
    PLOT_SUBTITLE_STRING = ""
if len(args) >= 10:
    op_times = {
        "u1" : int(args[9]),
        "u2" : int(args[9]),
        "u3" : int(args[9]),
        "cx" : int(args[10]),
        "measure" : int(args[11]),
        "reset" : int(args[12]),
        "id" : 1
    }

    noiseArgs = {
        "op_times" : op_times,
        "t1" : float(args[13]),
        "t2" : float(args[14]),
        "thermal_population_1" : float(args[15]),
        "measurement_error_prob0_given_1" : float(args[16]),
        "measurement_error_prob1_given_0" : float(args[16]),
        "depolarization_prob_single_qubit" : float(args[17]),
        "pauli_error_prob_single_qubit" : float(args[17]),
        "depolarization_prob_two_qubit" : float(args[18]),
        "pauli_error_prob_two_qubit" : float(args[18]),
        "depolarization_prob_reset" : float(args[19]),
        "pauli_error_prob_reset" : float(args[19])
    }
else:
    op_times = DEFAULT_OP_TIMES
    noiseArgs = {}

if NOISE_MODEL_CHOICE == "-f":
    noise_model_generator = construct_noise_model_full
elif NOISE_MODEL_CHOICE == "-t":
    noise_model_generator = construct_noise_model_thermalOnly
elif NOISE_MODEL_CHOICE == "-r":
    noise_model_generator = construct_noise_model_readoutOnly
elif NOISE_MODEL_CHOICE == "-g":
    noise_model_generator = construct_noise_model_noThermal
elif NOISE_MODEL_CHOICE == "-n":
    noise_model_generator = None
else:
    raise Exception("Did not recognize the noise flag you passed. Only -f,-t,-r,-g or -n are accepted.")

#####################################
###### Which circuits to run? #######
#####################################
# Order of tuples: Generating function, Plot label, Name to write in results file, Results, Plot color, Type of circuit it is, Slopes, Slope StdErr
# All of the generating functions needs to accept parameters (n, k, prep_state)
hTestOrig = (hTest_original_circuit, "H-Test Original", "H-Test Original", [], 'c', 'h', [], [])
hTestEff4k = (hTest_qubitEfficient4k_circuit, "H-Test Q.Eff. 4k", "H-Test Qubit Eff 4k+1", [], 'm', 'h', [], [])
hTestEff3k = (hTest_qubitEfficient3k_circuit, "H-Test Q.Eff. 3k", "H-Test Qubit Eff 3k+1", [], 'k', 'h', [], [])
hTestEff_alt4k = (hTest_qubitEfficient_alternative4k_circ, "H-Test 4k Alt", "H-Test Qubit Eff Alternative 4k", [], 'r', 'h', [], [])
hTestEff_alt3k = (hTest_qubitEfficient_alternative3k_circ, "H-Test 3k Alt", "H-Test Qubit Eff Alternative 3k", [], 'y', 'h', [], [])
twoCopyOrig = (twoCopyTest_original_circuit, "Two-Copy Orig", "Two-copy test Original", [], 'g', 't', [], [])
twoCopyEff6k = (twoCopyTest_qubitEfficient6k_circuit, "Two-Copy Q.Eff. 6k", "Two-copy test Qubit Eff 6k", [], 'r', 't', [], [])
twoCopyEff4k = (twoCopyTest_qubitEfficient4k_circuit, "Two-Copy Q.Eff. 4k", "Two-copy test Qubit Eff 4k", [], 'y', 't', [], [])

# Edit this list to specify which circuits to run or not to run:
all_circuits = (hTestOrig, hTestEff4k, hTestEff3k, hTestEff_alt4k, hTestEff_alt3k, twoCopyOrig, twoCopyEff6k, twoCopyEff4k)


######################################
###### Thetas and Exact Values #######
######################################
# Change these functions if you want to change the inputs and ideal outputs

NUM_THETAS = 20

def getThetaList(n):
    """
    Returns list of 20 thetas which give evenly spaced values of Tr(rho_A^n)

    Hardcoded values for convenience.

    Assumes we're using the default state prep function from spectroscopy, with k=1

    These values were generated using this Mathematica code:
        n = 7;
        numIntervals = 19;
        numDigitsPrecision = 10;
        outs = Subdivide[2^(-(n-1)), 1, numIntervals];
        trace[t] = Sum[ Binomial[n,i]*Sin[t]^i, {i,0,n, 2}] / 2^(n-1);
        result = N[Map[Reduce[{trace[t] == #, 0 <= t <= Pi/2}, t, Reals]&, outs], numDigitsPrecision];
        StringReplace[ToString[result], {"t == "->"","{"->"[","}"->"]"}]
    """
    if n == 2:
        theta_list = [0, 0.2314773640, 0.3304226479, 0.4086378551, 0.4766796116, 0.5386634661, 0.5967431472, 0.6522511548, 0.7061190231, 0.7590702093, 0.8117261175, 0.8646773037, 0.9185451720, 0.9740531796, 1.032132861, 1.094116715, 1.162158472, 1.240373679, 1.339318963, 1.570796327]
    elif n == 3:
        theta_list = [0, 0.2314773640, 0.3304226479, 0.4086378551, 0.4766796116, 0.5386634661, 0.5967431472, 0.6522511548, 0.7061190231, 0.7590702093, 0.8117261175, 0.8646773037, 0.9185451720, 0.9740531796, 1.032132861, 1.094116715, 1.162158472, 1.240373679, 1.339318963, 1.570796327]
    elif n == 4:
        theta_list = [0, 0.2491203095, 0.3543433131, 0.4366865759, 0.5076378996, 0.5716852714, 0.6311768187, 0.6875602412, 0.7418396403, 0.7947846259, 0.8470441546, 0.8992213155, 0.9519358743, 1.005893804, 1.061988158, 1.121480238, 1.186392369, 1.260572559, 1.353876403, 1.570796327]
    elif n == 5:
        theta_list = [0, 0.2794021424, 0.3935921523, 0.4808709997, 0.5546242519, 0.6201165657, 0.6801030414, 0.7362716601, 0.7897777908, 0.8414889475, 0.8921167248, 0.9423010413, 0.9926770691, 1.043945273, 1.096968442, 1.152941697, 1.213757347, 1.282990475, 1.369767528, 1.570796327]
    elif n == 6:
        theta_list = [0, 0.3199493272, 0.4426490985, 0.5332414006, 0.6079996178, 0.6732468307, 0.7322299041, 0.7868946105, 0.8385407416, 0.8881187231, 0.9363862185, 0.9840045982, 1.031611489, 1.079892021, 1.129672738, 1.182081773, 1.238888833, 1.303420250, 1.384147665, 1.570796327]
    elif n == 7:
        theta_list = [0, 0.3678313758, 0.4959520472, 0.5872307295, 0.6610521651, 0.7246517493, 0.7816281774, 0.8340827170, 0.8833879597, 0.9305272669, 0.9762694175, 1.021272932, 1.066161499, 1.111594881, 1.158359191, 1.207517971, 1.260730410, 1.321105684, 1.396551522, 1.570796327]
    elif n == 8:
        theta_list = [0, 0.4190980610, 0.5487218956, 0.6385481712, 0.7102223398, 0.7714770612, 0.8260582020, 0.8761131478, 0.9230247583, 0.9677718001, 1.011111010, 1.053683798, 1.096091665, 1.138965366, 1.183051185, 1.229353728, 1.279435279, 1.336218322, 1.407129918, 1.570796327]
    elif n == 9:
        theta_list = [0, 0.4699483644, 0.5980580911, 0.6852691420, 0.7543035643, 0.8130292719, 0.8651960287, 0.9129307474, 0.9575922479, 1.000135893, 1.041295848, 1.081690781, 1.121897674, 1.162518577, 1.204262943, 1.248083017, 1.295456906, 1.349146792, 1.416169103, 1.570796327]
    elif n == 10:
        theta_list = [0, 0.5178871080, 0.6428229977, 0.7269938361, 0.7933187827, 0.8495895696, 0.8994862332, 0.9450840890, 0.9877032578, 1.028268498, 1.067488042, 1.105956733, 1.144227554, 1.182875819, 1.222577841, 1.264239767, 1.309266550, 1.360282005, 1.423949179, 1.570796327]
    elif n == 11:
        theta_list = [0, 0.5618311107, 0.6829308316, 0.7640349883, 0.8277746957, 0.8817666042, 0.9295904649, 0.9732586582, 1.014048186, 1.052851733, 1.090351606, 1.127119732, 1.163686736, 1.200603658, 1.238517270, 1.278293144, 1.321272326, 1.369958243, 1.430707030, 1.570796327]
    elif n == 12:
        theta_list = [0, 0.6016113123, 0.7187601932, 0.7969478414, 0.8582967603, 0.9102117319, 0.9561635201, 0.9980997455, 1.037254578, 1.074489673, 1.110462801, 1.145724613, 1.180785326, 1.216174132, 1.252511494, 1.290627125, 1.331805837, 1.378445385, 1.436632811, 1.570796327]
    elif n == 13:
        theta_list = [0, 0.6374947672, 0.7508314977, 0.8263148487, 0.8854801079, 0.9355135269, 0.9797781857, 1.020159260, 1.057850304, 1.093683938, 1.128295237, 1.162215377, 1.195936075, 1.229966875, 1.264904742, 1.301547495, 1.341130084, 1.385956703, 1.441876305, 1.570796327]
    elif n == 14:
        theta_list = [0, 0.6699013348, 0.7796628239, 0.8526634711, 0.9098410717, 0.9581700326, 1.000911168, 1.039891076, 1.076265547, 1.110840399, 1.144229891, 1.176947451, 1.209468031, 1.242283462, 1.275969611, 1.311295749, 1.349452268, 1.392659851, 1.446555022, 1.570796327]
    elif n == 15:
        theta_list = [0, 0.6992688969, 0.8057167271, 0.8764438904, 0.9318105945, 0.9785912864, 1.019951286, 1.057662855, 1.092846933, 1.126284704, 1.158571373, 1.190204182, 1.221642854, 1.253363183, 1.285921989, 1.320062813, 1.356935956, 1.398686995, 1.450761486, 1.570796327]
    elif n == 16:
        theta_list = [0, 0.7259992723, 0.8293876787, 0.8980308021, 0.9517428409, 0.9971115976, 1.037213790, 1.073771462, 1.107873437, 1.140278259, 1.171563669, 1.202212111, 1.232669418, 1.263396793, 1.294933762, 1.328000541, 1.363711112, 1.404143059, 1.454569088, 1.570796327]
    elif n == 17:
        theta_list = [0, 0.7504420305, 0.8510056543, 0.9177332773, 0.9699277528, 1.014003334, 1.052954644, 1.088457310, 1.121570524, 1.153031985, 1.183403375, 1.213153560, 1.242715652, 1.272537518, 1.303142892, 1.335230726, 1.369881908, 1.409112095, 1.458036592, 1.570796327]
    elif n == 18:
        theta_list = [0, 0.7728943275, 0.8708449345, 0.9358060701, 0.9866032111, 1.029489282, 1.067382764, 1.101916298, 1.134121676, 1.164717338, 1.194250156, 1.223176488, 1.251917747, 1.280909547, 1.310661141, 1.341851985, 1.375532656, 1.413662103, 1.461211518, 1.570796327]
    elif n == 19:
        theta_list = [0, 0.7936065797, 0.8891335947, 0.9524599192, 1.001965421, 1.043752841, 1.080669888, 1.114309291, 1.145677424, 1.175474885, 1.204234833, 1.232402061, 1.260387176, 1.288614493, 1.317579915, 1.347944949, 1.380732269, 1.417848650, 1.464132692, 1.570796327]
    elif n == 20:
        theta_list = [0, 0.8127895528, 0.9060620298, 0.9678701650, 1.016177317, 1.056946117, 1.092958313, 1.125769477, 1.156362330, 1.185420882, 1.213465554, 1.240930408, 1.268216032, 1.295736274, 1.323974685, 1.353576174, 1.385537623, 1.421717585, 1.466832141, 1.570796327]
    else:
        raise Exception("Thetas have only been prepared for n = 2 to 20.")
    return theta_list

def getExactTraces(n):
    """
    Returns list of NUM_THETAS evenly spaced values from 0 to 2^(-(n-1)).

    So this assumes you pick thetas such that the exact values are this evenly spaced list.

    See `computeExactTraces_forDefaultStatePrep` in spectroscopy to calculate the traces for
    arbitrary thetas.
    """
    return np.linspace(2 ** (-(n-1)), 1, NUM_THETAS)


###########################
###### *** MAIN *** #######
###########################
backend = QasmSimulator()

os.makedirs(os.path.dirname(OUTPUT_DIRECTORY + "/results.txt"), exist_ok=True)
resultsFile = open(OUTPUT_DIRECTORY + "/results.txt", "w", buffering=1)
logFile = open(OUTPUT_DIRECTORY + "/results_log.txt", "w", buffering=1)

resultsFile.write("Arguments: \n" + str(args) + "\n\n")

for r in all_circuits:
    resultsFile.write(r[1] + "\n")
resultsFile.write("\n")

logFile.write("STARTING \n")

nStartTime = perf_counter()
lastTime = nStartTime
newTime = nStartTime

for n in N_LIST:
    for r in all_circuits:
        r[3].clear()

    theta_list = getThetaList(n)
    exact_values = getExactTraces(n)
    resultsFile.write("n = " + str(n) + "\n")
    resultsFile.write("Thetas: " + str(theta_list) + "\n")
    resultsFile.write("Exact Values: " + str(exact_values) + "\n")

    for thetaCounter, theta in enumerate(theta_list):
        prep_state = generate_default_prep_state_instruction(theta, K)

        for circTuple in all_circuits:
            circuit = circTuple[0](n, K, prep_state)
            if NOISE_MODEL_CHOICE != "-n": # Noisy
                circuitThermalReady = construct_modified_circuit_for_thermal_noise(circuit, op_times)
                noise = noise_model_generator(len(circuit.qubits), **noiseArgs)
                counts = execute(circuitThermalReady, backend=backend, shots=SHOTS, basis_gates=noise.basis_gates, noise_model=noise).result().get_counts()
            else: # Noiseless
                counts = execute(circuit, backend=backend, shots=SHOTS).result().get_counts()

            if circTuple[5] == "h":
                answer = hTest_computeAnswer(counts)
            elif circTuple[5] == "t":
                answer = twoCopyTest_computeAnswer(n, K, counts)
            circTuple[3].append(answer)

            logFile.write(circTuple[2] + ". N = " + str(n) + ". Theta number " + str(thetaCounter) + "\n")
            newTime = perf_counter()
            logFile.write("Time to complete this simulation: " + str(newTime - lastTime) + "\n")
            lastTime = newTime

    for r in all_circuits:
        resultsFile.write(str(r[3]) + "\n")
    resultsFile.write("\n")

    newTime = perf_counter()
    logFile.write("Total time taken for N=" + str(n) + " was " + str(newTime - nStartTime) + "\n")
    logFile.write("\n")
    nStartTime = newTime


    # Normal Plot
    plt.clf()
    plt.axis([-.02, 1.6, 0, 1.02])
    plt.plot(theta_list, exact_values, 'b')
    for r in all_circuits:
        if r[5] == 'h':
            plt.errorbar(theta_list, r[3], yerr = hTest_computeErrorBars(SHOTS, r[3], CONFIDENCE_LEVEL), color = r[4], linestyle = '--')
        elif r[5] == 't':
            plt.errorbar(theta_list, r[3], yerr = twoCopyTest_computeErrorBars(SHOTS, r[3], CONFIDENCE_LEVEL), color = r[4], linestyle = '-')
        else:
            print("ERROR! Unknown algorithm type in r[4]. Don't know how to plot.")
    plt.legend(["exact"] + [r[1] for r in all_circuits])
    plt.title("N = " + str(n) + " " + PLOT_SUBTITLE_STRING)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIRECTORY + "/plot_n" +str(n)+ ".png", dpi=300)

    # Linear Plot
    plt.clf()
    plt.axis([-.02, 1.02, 0, 1.02])
    plt.plot(exact_values, exact_values, 'b')
    for r in all_circuits:
        if r[5] == 'h':
            plt.errorbar(exact_values, r[3], yerr = hTest_computeErrorBars(SHOTS, r[3], CONFIDENCE_LEVEL), color = r[4], linestyle = '--')
        elif r[5] == 't':
            plt.errorbar(exact_values, r[3], yerr = twoCopyTest_computeErrorBars(SHOTS, r[3], CONFIDENCE_LEVEL), color = r[4], linestyle = '-')
        else:
            print("ERROR! Unknown algorithm type in r[4]. Don't know how to plot.")
    plt.legend(["exact"] + [r[1] for r in all_circuits])
    plt.title("N = " + str(n) + " " + PLOT_SUBTITLE_STRING)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIRECTORY + "/linearPlot_n" +str(n)+ ".png", dpi=300)

    # Calculate slopes
    resultsFile.write("Slopes, Std Err, R squared: \n")
    for r in all_circuits:
        regression = linregress(exact_values, r[3])
        resultsFile.write(str(regression[0]) + " , " + str(regression[4]) + " , " + str(regression[2]) + "\n")
        r[6].append(regression[0]) # slope
        r[7].append(regression[4]) # stderr
    resultsFile.write("\n")

resultsFile.close()
logFile.close()

# Plot Slopes
def calculateSlopeError(numPoints, stderr, confidence_level):
    """
    Calculates the error in the slope according to a given confidence level.

    data : number of points the linregress was based on
    stderr : generally, the error output by the linregress function
    confidence_level : a decimal, such as 0.95

    Calculations are based on these instructions:
    https://stattrek.com/regression/slope-confidence-interval.aspx
    """
    score = t.ppf(1 - ((1 - confidence_level) / 2), numPoints - 2)
    margin_of_error = score * stderr
    return margin_of_error

plt.clf()
plt.axis([N_LIST[0], N_LIST[-1] + 1, -0.1, 1.1])
plt.plot(N_LIST, [1]*len(N_LIST), 'b')
for circTuple in all_circuits:
    slopes = circTuple[6]
    stdErrors = circTuple[7]
    errors = [calculateSlopeError(NUM_THETAS, stderr, CONFIDENCE_LEVEL) for stderr in stdErrors]
    if circTuple[5] == "h":
        linestyle = '--'
    elif circTuple[5] == "t":
        linestyle = '-'
    plt.errorbar(N_LIST, slopes, yerr = errors, color = circTuple[4], linestyle = linestyle)
plt.legend(["exact"] + [r[1] for r in all_circuits])
plt.title("Slopes. " + PLOT_SUBTITLE_STRING)
plt.tight_layout()
plt.savefig(OUTPUT_DIRECTORY + "/slopePlot.png", dpi=300)
