"""
imdd_lib.py
===========
Simulation library for IM-DD optical access links.
Project: "Optical IM-DD / PAM Access-Link Trade-off Study"

Structure:
    1. Imports & GPU check
    2. Transmitter block
    3. Fiber channel block
    4. Receiver block
    5. Single-run simulation  (run_link)
    6. Parameter sweep functions
    7. Plot / visualization functions
    8. Utility functions
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt

from optic.models.devices import mzm, photodiode
from optic.models.channels import linearFiberChannel
from optic.comm.modulation import modulateGray, demodulateGray, grayMapping
from optic.comm.sources import bitSource
from optic.dsp.core import upsample, pulseShape, pnorm, anorm, signalPower
from optic.utils import parameters, dBm2W
from optic.plot import eyediagram

try:
    from optic.dsp.coreGPU import checkGPU
    if checkGPU():
        from optic.dsp.coreGPU import firFilter
    else:
        from optic.dsp.core import firFilter
except ImportError:
    from optic.dsp.core import firFilter


# =============================================================================
# 2. TRANSMITTER BLOCK
# =============================================================================

def build_transmitter(Pi_dBm, M, SpS, Rs, paramBits, paramPulse, paramMZM, seed=None):
    """
    Build the optical transmitter signal chain.

    Bit source → Gray-coded PAM modulation → upsample → NRZ pulse shaping → MZM.

    Parameters
    ----------
    Pi_dBm    : float      — Laser input power to MZM [dBm]
    M         : int        — Modulation order (2 = OOK, 4 = PAM4)
    SpS       : int        — Samples per symbol
    Rs        : float      — Symbol rate [Hz]
    paramBits : parameters — Bit source config (nBits, mode, order)
    paramPulse: parameters — Pulse shaping config (pulseType, SpS)
    paramMZM  : parameters — MZM config (Vpi, Vb)
    seed      : int|None   — RNG seed for the bit source (None = not fixed)

    Returns
    -------
    sigTxo : ndarray — Complex optical signal at MZM output
    bitsTx : ndarray — Transmitted bit sequence (reference for BER)
    symbTx : ndarray — Transmitted symbol sequence (reference for PAM4 decision)
    """
    Pi = dBm2W(Pi_dBm)

    if seed is not None:
        paramBits.seed = seed

    # Generate PRBS bit sequence
    bitsTx = bitSource(paramBits)

    # Gray-coded PAM modulation (M=2 → OOK/2-PAM, M=4 → 4-PAM)
    symbTx = modulateGray(bitsTx, M, 'pam')
    symbTx = pnorm(symbTx)  # power normalization

    # Upsample and NRZ pulse shaping
    symbolsUp = upsample(symbTx, SpS)
    pulse     = pulseShape(paramPulse)
    sigTx     = firFilter(pulse, symbolsUp)
    sigTx     = anorm(sigTx)

    # Optical intensity modulation via MZM
    Ai     = np.sqrt(Pi)
    sigTxo = mzm(Ai, sigTx, paramMZM)

    return sigTxo, bitsTx, symbTx


# =============================================================================
# 3. FIBER CHANNEL BLOCK
# =============================================================================

def build_channel(sigTxo, paramCh):
    """
    Propagate the optical signal through a linear fiber channel.

    No optical amplification is applied — fiber loss directly reduces received power,
    which is controlled via Pi_dBm at the transmitter to sweep received power levels.

    Parameters
    ----------
    sigTxo  : ndarray    — Complex optical signal at the MZM output
    paramCh : parameters — Fiber channel config:
                               L   [km]       total link length
                               α   [dB/km]    attenuation coefficient
                               D   [ps/nm/km] dispersion parameter
                               Fc  [Hz]       central optical frequency
                               Fs  [Hz]       simulation sampling frequency

    Returns
    -------
    sigCh : ndarray — Complex optical signal after fiber propagation
    """
    sigCh = linearFiberChannel(sigTxo, paramCh)
    return sigCh


# =============================================================================
# 4. RECEIVER BLOCK
# =============================================================================

def build_receiver(sigCh, bitsTx, M, SpS, paramPD, discard=100):
    """
    Receiver chain: photodiode → normalization → sampling → decision → BER.
    Supports OOK (M=2) and 4-PAM (M=4) via a unified multi-level slicer.

    Parameters
    ----------
    sigCh   : ndarray    — Optical signal at the photodiode input
    bitsTx  : ndarray    — Reference transmitted bit sequence
    M       : int        — Modulation order (2 or 4)
    SpS     : int        — Samples per symbol
    paramPD : parameters — Photodiode config (ideal, B, Fs, ipd_sat, ...)
    discard : int        — Number of symbols to discard at each end when counting BER

    Returns
    -------
    dict with keys:
        'BER'    : float   — Measured bit error rate
        'Pb'     : float   — Approx. theoretical BER from worst-case Q-factor
                             (exact for OOK, approximate for 4-PAM)
        'Q'      : float   — Worst-case (minimum) eye Q-factor across all
                             adjacent decision levels
        'I_Rx'   : ndarray — Full-rate photodiode current (used for eye diagrams)
        'I_dec'  : ndarray — Symbol-rate samples at decision point
        'bitsRx' : ndarray — Decided bit sequence
    """
    if M not in (2, 4):
        raise ValueError(f"Unsupported modulation order M={M}. Use M=2 (OOK) or M=4 (PAM4).")

    # Optical-to-electrical conversion
    I_Rx      = photodiode(sigCh, paramPD)
    I_Rx_full = I_Rx.copy()  # keep full-rate copy for eye diagram

    # Normalize
    I_Rx = I_Rx / np.std(I_Rx)

    # Sample at symbol center (one sample per symbol)
    I_dec = I_Rx[0::SpS]

    symbTx_ideal = pnorm(modulateGray(bitsTx, M, 'pam')).real

    # Normalized levels: match the scale of I_dec, used only for thresholding
    levels_norm = np.sort(np.unique(np.round(pnorm(grayMapping(M, 'pam')).real, 10)))

    # Raw (un-normalized) levels: exactly what demodulateGray expects
    levels_raw = np.sort(np.unique(np.round(grayMapping(M, 'pam').real, 10)))

    symb_idx = np.argmin(np.abs(symbTx_ideal[:, None] - levels_norm[None, :]), axis=1)

    means = np.array([I_dec[symb_idx == k].mean() for k in range(M)])
    stds = np.array([I_dec[symb_idx == k].std() for k in range(M)])

    thr = (stds[:-1] * means[1:] + stds[1:] * means[:-1]) / (stds[:-1] + stds[1:])

    decided_idx = np.digitize(I_dec, thr)
    symbDec = levels_raw[decided_idx]  # <-- use RAW levels here
    bitsRx = demodulateGray(symbDec, M, 'pam').astype(int)

    Q = np.min((means[1:] - means[:-1]) / (stds[1:] + stds[:-1]))
    Pb = (2 * (M - 1) / M) * 0.5 * erfc(Q / np.sqrt(2)) / np.log2(M)

    err = np.logical_xor(
        bitsRx[discard: bitsRx.size - discard],
        bitsTx[discard: bitsTx.size - discard],
    )
    BER = np.mean(err)

    return {
        'BER': BER, 'Pb': Pb, 'Q': Q,
        'I_Rx': I_Rx_full, 'I_dec': I_dec, 'bitsRx': bitsRx,
    }


# =============================================================================
# 5. SINGLE-RUN SIMULATION
# =============================================================================

def run_link(
    Pi_dBm,
    M           = 2,
    SpS         = 16,
    Rs          = 10e9,
    fiber_L     = 10,
    fiber_alpha = 0.2,
    fiber_D     = 16,
    Fc          = 193.1e12,
    rx_bandwidth = None,
    rx_ideal     = False,
    nBits        = 100000,
    seed         = None,
    discard      = 100,
):
    """
    Run a complete IM-DD link simulation: Tx → Fiber → Rx → BER.

    Parameters
    ----------
    Pi_dBm       : float     — Laser power at MZM input [dBm]
                               (controls received power after fiber loss)
    M            : int       — Modulation order: 2 = OOK, 4 = PAM4
    SpS          : int       — Samples per symbol (default 16)
    Rs           : float     — Symbol rate [Hz] (default 10 Gbaud)
    fiber_L      : float     — Fiber length [km] (default 10 km)
    fiber_alpha  : float     — Fiber attenuation [dB/km] (default 0.2)
    fiber_D      : float     — Fiber dispersion [ps/nm/km] (default 16)
    Fc           : float     — Central optical frequency [Hz]
    rx_bandwidth : float|None — Photodiode bandwidth [Hz]
                                None → matched filter: B = Rs
    rx_ideal     : bool      — True = noiseless, unlimited-bandwidth photodiode
    nBits        : int       — Number of bits to simulate
    seed         : int|None  — RNG seed (None = not fixed)
    discard      : int       — Guard symbols excluded from BER count

    Returns
    -------
    result : dict
        Performance metrics : 'BER', 'Pb', 'Q'
        Signals             : 'I_Rx', 'I_dec', 'sigTxo', 'sigCh'
        Bit sequences       : 'bitsTx', 'bitsRx', 'symbTx'
        Sim parameters      : 'SpS', 'Rs', 'Fs', 'Pi_dBm', 'M'
    """
    Fs = Rs * SpS

    # --- Block configurations ---
    paramBits        = parameters()
    paramBits.nBits  = nBits
    paramBits.mode   = 'prbs'
    paramBits.order  = 23

    paramPulse           = parameters()
    paramPulse.pulseType = 'nrz'
    paramPulse.SpS       = SpS

    paramMZM     = parameters()
    paramMZM.Vpi = 2
    paramMZM.Vb  = -paramMZM.Vpi / 2

    paramCh    = parameters()
    paramCh.L  = fiber_L
    paramCh.α  = fiber_alpha
    paramCh.D  = fiber_D
    paramCh.Fc = Fc
    paramCh.Fs = Fs

    paramPD       = parameters()
    paramPD.ideal = rx_ideal
    paramPD.Fs    = Fs
    paramPD.B     = rx_bandwidth if rx_bandwidth is not None else Rs

    # --- Run each block in sequence ---
    sigTxo, bitsTx, symbTx = build_transmitter(
        Pi_dBm, M, SpS, Rs, paramBits, paramPulse, paramMZM, seed=seed
    )

    sigCh = build_channel(sigTxo, paramCh)

    rx_result = build_receiver(sigCh, bitsTx, M, SpS, paramPD, discard=discard)

    # Merge all results into a single dict
    result = rx_result
    result.update({
        'bitsTx' : bitsTx,
        'symbTx' : symbTx,
        'sigTxo' : sigTxo,
        'sigCh'  : sigCh,
        'SpS'    : SpS,
        'Rs'     : Rs,
        'Fs'     : Fs,
        'Pi_dBm' : Pi_dBm,
        'M'      : M,
    })
    return result


# =============================================================================
# 6. PARAMETER SWEEP FUNCTIONS
# =============================================================================

def sweep_ber_vs_power(power_range, M=2, Rs=10e9, SpS=16, fiber_L=10,
                       rx_bandwidth=None, nBits=100000, verbose=True, **kwargs):
    """
    Sweep BER over a range of received optical power levels.

    Each power point uses a different RNG seed to avoid statistical correlation.

    Parameters
    ----------
    power_range  : array-like — Input power values [dBm], e.g. np.arange(-30, -14)
    M            : int        — Modulation order (2 or 4)
    Rs           : float      — Symbol rate [Hz]
    SpS          : int        — Samples per symbol
    fiber_L      : float      — Fiber length [km]
    rx_bandwidth : float|None — Photodiode bandwidth [Hz] (None = Rs)
    nBits        : int        — Bits per simulation run
    verbose      : bool       — Show tqdm progress bar
    **kwargs                  — Additional arguments forwarded to run_link()

    Returns
    -------
    dict:
        'power' : ndarray — Power sweep values [dBm]
        'BER'   : ndarray — Simulated BER at each point
        'Pb'    : ndarray — Theoretical BER at each point
        'Q'     : ndarray — Q-factor at each point
    """
    from tqdm import tqdm

    power_range = np.asarray(power_range)
    BER = np.zeros(power_range.shape)
    Pb  = np.zeros(power_range.shape)
    Q   = np.zeros(power_range.shape)

    iterator = tqdm(enumerate(power_range), total=len(power_range),
                    desc='Sweep: power') if verbose else enumerate(power_range)

    for i, Pi_dBm in iterator:
        res    = run_link(Pi_dBm=Pi_dBm, M=M, Rs=Rs, SpS=SpS,
                          fiber_L=fiber_L, rx_bandwidth=rx_bandwidth,
                          nBits=nBits, seed=12335 + i, **kwargs)
        BER[i] = res['BER']
        Pb[i]  = res['Pb']
        Q[i]   = res['Q']

    return {'power': power_range, 'BER': BER, 'Pb': Pb, 'Q': Q}


def sweep_ber_vs_bandwidth(bw_range, Pi_dBm=-20, M=2, Rs=10e9, SpS=16,
                           fiber_L=10, nBits=100000, verbose=True, **kwargs):
    """
    Sweep BER over a range of receiver bandwidths.

    Parameters
    ----------
    bw_range : array-like — Bandwidth values [Hz], e.g. np.linspace(0.5*Rs, 2*Rs, 20)
    Pi_dBm   : float      — Fixed transmit power [dBm]
    (remaining parameters same as sweep_ber_vs_power)

    Returns
    -------
    dict:
        'bandwidth' : ndarray — Bandwidth sweep values [Hz]
        'BER'       : ndarray
        'Pb'        : ndarray
        'Q'         : ndarray
    """
    from tqdm import tqdm

    bw_range = np.asarray(bw_range)
    BER = np.zeros(bw_range.shape)
    Pb  = np.zeros(bw_range.shape)
    Q   = np.zeros(bw_range.shape)

    iterator = tqdm(enumerate(bw_range), total=len(bw_range),
                    desc='Sweep: bandwidth') if verbose else enumerate(bw_range)

    for i, bw in iterator:
        res    = run_link(Pi_dBm=Pi_dBm, M=M, Rs=Rs, SpS=SpS,
                          fiber_L=fiber_L, rx_bandwidth=bw,
                          nBits=nBits, seed=12335, **kwargs)
        BER[i] = res['BER']
        Pb[i]  = res['Pb']
        Q[i]   = res['Q']

    return {'bandwidth': bw_range, 'BER': BER, 'Pb': Pb, 'Q': Q}


def sweep_ber_vs_fiber_length(length_range, Pi_dBm=-20, M=2, Rs=10e9, SpS=16,
                               rx_bandwidth=None, nBits=100000, verbose=True, **kwargs):
    """
    Sweep BER over a range of fiber lengths.

    Parameters
    ----------
    length_range : array-like — Fiber length values [km], e.g. np.arange(0, 80, 5)
    Pi_dBm       : float      — Fixed transmit power [dBm]
    (remaining parameters same as sweep_ber_vs_power)

    Returns
    -------
    dict:
        'length' : ndarray — Length sweep values [km]
        'BER'    : ndarray
        'Pb'     : ndarray
        'Q'      : ndarray
    """
    from tqdm import tqdm

    length_range = np.asarray(length_range)
    BER = np.zeros(length_range.shape)
    Pb  = np.zeros(length_range.shape)
    Q   = np.zeros(length_range.shape)

    iterator = tqdm(enumerate(length_range), total=len(length_range),
                    desc='Sweep: fiber length') if verbose else enumerate(length_range)

    for i, L in iterator:
        res    = run_link(Pi_dBm=Pi_dBm, M=M, Rs=Rs, SpS=SpS,
                          fiber_L=L, rx_bandwidth=rx_bandwidth,
                          nBits=nBits, seed=12335, **kwargs)
        BER[i] = res['BER']
        Pb[i]  = res['Pb']
        Q[i]   = res['Q']

    return {'length': length_range, 'BER': BER, 'Pb': Pb, 'Q': Q}


def sweep_ber_vs_dispersion(dispersion_range, Pi_dBm=-20, M=2, Rs=10e9, SpS=16,
                             fiber_L=10, rx_bandwidth=None, nBits=100000,
                             verbose=True, **kwargs):
    """
    Sweep BER over a range of fiber dispersion coefficients.

    Parameters
    ----------
    dispersion_range : array-like — Dispersion values [ps/nm/km],
                                    e.g. np.arange(0, 20, 1)
    Pi_dBm           : float      — Fixed transmit power [dBm]
    fiber_L          : float      — Fixed fiber length [km]
    (remaining parameters same as sweep_ber_vs_power)

    Returns
    -------
    dict:
        'dispersion' : ndarray — Dispersion sweep values [ps/nm/km]
        'BER'        : ndarray
        'Pb'         : ndarray
        'Q'          : ndarray
    """
    from tqdm import tqdm

    dispersion_range = np.asarray(dispersion_range)
    BER = np.zeros(dispersion_range.shape)
    Pb  = np.zeros(dispersion_range.shape)
    Q   = np.zeros(dispersion_range.shape)

    iterator = tqdm(enumerate(dispersion_range), total=len(dispersion_range),
                    desc='Sweep: dispersion') if verbose else enumerate(dispersion_range)

    for i, D in iterator:
        res    = run_link(Pi_dBm=Pi_dBm, M=M, Rs=Rs, SpS=SpS,
                          fiber_L=fiber_L, fiber_D=D, rx_bandwidth=rx_bandwidth,
                          nBits=nBits, seed=12335, **kwargs)
        BER[i] = res['BER']
        Pb[i]  = res['Pb']
        Q[i]   = res['Q']

    return {'dispersion': dispersion_range, 'BER': BER, 'Pb': Pb, 'Q': Q}


# =============================================================================
# 7. PLOT / VISUALIZATION FUNCTIONS
# =============================================================================

def plot_eye_diagrams(result, discard=50):
    """
    Plot Tx and Rx eye diagrams from a run_link() result.

    The Tx eye uses an ideal (noiseless) photodiode applied to the MZM output.
    The Rx eye uses the actual photodiode current stored in result['I_Rx'].

    Parameters
    ----------
    result  : dict — Output dict from run_link()
    discard : int  — Number of symbols to discard at each end before plotting
    """
    SpS    = result['SpS']
    sigTxo = result['sigTxo']
    I_Rx   = result['I_Rx']

    # Ideal photodiode for Tx eye (no noise, no bandwidth limit)
    paramPD_ideal       = parameters()
    paramPD_ideal.ideal = True
    paramPD_ideal.Fs    = result['Fs']
    I_Tx = photodiode(sigTxo.real, paramPD_ideal)

    d = discard * SpS
    eyediagram(I_Tx[d:-d], I_Tx.size - 2*d, SpS, plotlabel='Tx eye', ptype='fancy')
    eyediagram(I_Rx[d:-d], I_Rx.size - 2*d, SpS, plotlabel='Rx eye', ptype='fancy')


def plot_ber_vs_power(results_list, labels=None, target_BER=None, title='BER vs Received Power'):
    """
    Plot BER curves as a function of received optical power.

    Parameters
    ----------
    results_list : list of dict — Each dict is output from sweep_ber_vs_power()
    labels       : list of str  — Legend labels, e.g. ['OOK', 'PAM4']
    title        : str          — Plot title
    """
    if labels is None:
        labels = [f'Config {i+1}' for i in range(len(results_list))]

    plt.figure(figsize=(8, 5))
    for res, label in zip(results_list, labels):
        p = res['power']
        plt.plot(p, np.log10(np.clip(res['Pb'],  1e-12, 1)), '--',
                 label=f'{label} — Pb (theory)')
        plt.plot(p, np.log10(np.clip(res['BER'], 1e-12, 1)), 'o-',
                 label=f'{label} — BER (sim)')

    # Mark FEC threshold if available
    if target_BER is not None:
        plt.axhline(np.log10(target_BER), color='gray', linestyle=':', label=f'BER = {target_BER}')

    plt.xlabel('Pin [dBm]')
    plt.ylabel('log$_{10}$(BER)')
    plt.title(title)
    plt.ylim(-10, 0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_ber_vs_bandwidth(results_list, Rs=10e9, labels=None,
                          title='BER vs Receiver Bandwidth'):
    """
    Plot BER as a function of normalized receiver bandwidth (B / Rs).

    Parameters
    ----------
    results_list : list of dict — Each dict is output from sweep_ber_vs_bandwidth()
    Rs           : float        — Symbol rate [Hz] used for normalization
    labels       : list of str
    title        : str
    """
    if labels is None:
        labels = [f'Config {i+1}' for i in range(len(results_list))]

    plt.figure(figsize=(8, 5))
    for res, label in zip(results_list, labels):
        bw_norm = res['bandwidth'] / Rs
        plt.plot(bw_norm, np.log10(np.clip(res['BER'], 1e-12, 1)), 'o-', label=label)

    plt.xlabel('Bandwidth / Rs')
    plt.ylabel('log$_{10}$(BER)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_ber_vs_length(results_list, labels=None, title='BER vs Fiber Length'):
    """
    Plot BER as a function of fiber length.

    Parameters
    ----------
    results_list : list of dict — Each dict is output from sweep_ber_vs_fiber_length()
    labels       : list of str
    title        : str
    """
    if labels is None:
        labels = [f'Config {i+1}' for i in range(len(results_list))]

    plt.figure(figsize=(8, 5))
    for res, label in zip(results_list, labels):
        plt.plot(res['length'], np.log10(np.clip(res['BER'], 1e-12, 1)), 'o-', label=label)

    plt.xlabel('Fiber Length [km]')
    plt.ylabel('log$_{10}$(BER)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_ber_vs_dispersion(results_list, labels=None, title='BER vs Fiber Dispersion'):
    """
    Plot BER as a function of fiber dispersion coefficient.

    Parameters
    ----------
    results_list : list of dict — Each dict is output from sweep_ber_vs_dispersion()
    labels       : list of str
    title        : str
    """
    if labels is None:
        labels = [f'Config {i+1}' for i in range(len(results_list))]

    plt.figure(figsize=(8, 5))
    for res, label in zip(results_list, labels):
        plt.plot(res['dispersion'], np.log10(np.clip(res['BER'], 1e-12, 1)), 'o-', label=label)

    plt.xlabel('Dispersion [ps/nm/km]')
    plt.ylabel('log$_{10}$(BER)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# 8. UTILITY FUNCTIONS
# =============================================================================

def ber_floor(BER_array, floor=1e-12):
    """Clip a BER array to a minimum floor value to avoid log10(0)."""
    return np.clip(BER_array, floor, 1.0)


def print_summary(result):
    """Print a concise summary of a single run_link() result."""
    mod_name = 'OOK' if result['M'] == 2 else 'PAM4'
    print(f"  Modulation : M={result['M']} ({mod_name})")
    print(f"  Pi_dBm     : {result['Pi_dBm']:.1f} dBm")
    print(f"  Symbol rate: {result['Rs']/1e9:.1f} Gbaud")
    print(f"  Q-factor   : {result['Q']:.2f}")
    print(f"  BER (sim)  : {result['BER']:.2e}")
    print(f"  Pb (theory): {result['Pb']:.2e}")