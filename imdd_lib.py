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

import os
import pandas as pd

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

def build_receiver(sigCh, bitsTx, M, SpS, paramPD, discard=100, n_train=2000):
    """
    Receiver chain: photodiode -> normalization -> pilot-aided estimation of
    sampling phase & decision thresholds -> decision on the payload -> BER.
    Supports OOK (M=2) and 4-PAM (M=4).

    Parameters
    ----------
    sigCh   : ndarray    -- Optical signal at the photodiode input
    bitsTx  : ndarray    -- Transmitted bits. Only the first n_train*log2(M)
                            bits (the pilot) are visible to the receiver
                            algorithm; the remainder is used for scoring.
    M       : int        -- Modulation order (2 or 4)
    SpS     : int        -- Samples per symbol
    paramPD : parameters -- Photodiode config (ideal, B, Fs, ipd_sat, ...)
    discard : int        -- Guard symbols: skipped at the start of the pilot
                            (filter transient) and at the end of the payload
    n_train : int        -- Length of the known pilot, in symbols

    Returns
    -------
    dict with keys:
        'BER'      : float   -- Bit error rate measured on the PAYLOAD only
        'BER_train': float   -- BER on the pilot (in-sample, for reference)
        'Pb'       : float   -- Approx. theoretical BER from the pilot-estimated
                                worst-case Q (exact for OOK, approx. for 4-PAM)
        'Q'        : float   -- Worst-case eye Q-factor estimated on the pilot
        'I_Rx'     : ndarray -- Full-rate photodiode current (for eye diagrams)
        'I_dec'    : ndarray -- Symbol-rate samples at the chosen phase
        'bitsRx'   : ndarray -- Decided bits for the whole frame (pilot+payload)
        'phase'    : int     -- Sampling phase chosen from the pilot
        'means'    : ndarray -- Per-level means estimated on the pilot
        'stds'     : ndarray -- Per-level stds estimated on the pilot
        'thr'      : ndarray -- Decision thresholds estimated on the pilot
        'n_train'  : int     -- Pilot length used [symbols]
    """
    if M not in (2, 4):
        raise ValueError(f"Unsupported modulation order M={M}. Use M=2 (OOK) or M=4 (PAM4).")

    bps = int(np.log2(M))                       # bits per symbol
    n_sym = bitsTx.size // bps                  # symbols in the frame

    if n_train <= 2 * discard:
        raise ValueError(f"n_train={n_train} must be > 2*discard={2 * discard}.")
    if n_train + discard >= n_sym:
        raise ValueError(
            f"Frame too short: n_train={n_train} + discard={discard} >= {n_sym} symbols."
        )

    # Ideal constellation levels in ascending order.
    # OOK : [-1, 1]   |   4-PAM : [-3, -1, 1, 3]
    levels = np.sort(grayMapping(M, 'pam').real)

    # The receiver is allowed to know only pilot
    pilot_bits = bitsTx[: n_train * bps]
    pilot_sym  = modulateGray(pilot_bits, M, 'pam').real
    pilot_idx  = np.argmin(np.abs(pilot_sym[:, None] - levels[None, :]), axis=1)

    # Optical-to-electrical conversion
    I_Rx      = photodiode(sigCh, paramPD)
    I_Rx_full = I_Rx.copy()  # keep full-rate copy for eye diagram

    # Normalize (uses only the received waveform)
    I_Rx = I_Rx / np.std(I_Rx)

    # ---- Pilot-aided estimation of sampling phase and thresholds ----------
    phase, means, stds, thr, Q = _estimate_phase_and_thresholds(
        I_Rx, pilot_idx, M, SpS, guard=discard
    )

    # ---- Sample and decide the whole frame with the pilot-derived settings
    I_dec = I_Rx[phase::SpS][:n_sym]
    decided_idx = np.digitize(I_dec, thr)
    symbDec = levels[decided_idx]
    bitsRx = demodulateGray(symbDec, M, 'pam').astype(int)

    # ---- Metrics ----------------------------------------------------------
    # Approximate theoretical BER from the pilot-estimated worst-case Q
    # (exact for OOK, nearest-neighbour approximation for Gray-coded M-PAM).
    Pb = (2 * (M - 1) / M) * 0.5 * erfc(Q / np.sqrt(2)) / np.log2(M)

    # Simulated BER on the payload only (bits of symbols [n_train, n_sym-discard)),
    # comparing against the ground truth (bitsTx)
    b0, b1 = n_train * bps, (n_sym - discard) * bps
    BER = np.mean(np.logical_xor(bitsRx[b0:b1], bitsTx[b0:b1]))

    # In-sample BER on the pilot itself (optimistic; for diagnostics only).
    g0 = discard * bps
    BER_train = np.mean(np.logical_xor(bitsRx[g0:n_train * bps], pilot_bits[g0:]))

    return {
        'BER': BER, 'BER_train': BER_train, 'Pb': Pb, 'Q': Q,
        'I_Rx': I_Rx_full, 'I_dec': I_dec, 'bitsRx': bitsRx,
        'phase': phase, 'means': means, 'stds': stds, 'thr': thr,
        'n_train': n_train, 'discard': discard,
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
    n_train      = 2000,
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
    discard      : int       — Guard symbols (pilot start / payload end)
    n_train      : int       — Known pilot length [symbols] used by the receiver
                               to estimate sampling phase and thresholds; BER is
                               measured on the remaining payload only

    Returns
    -------
    result : dict
        Performance metrics : 'BER' (payload), 'BER_train', 'Pb', 'Q'
        Rx estimates        : 'phase', 'means', 'stds', 'thr', 'n_train'
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

    # Optical power launched into the fiber
    Ptx_dBm = W2dBm(signalPower(sigTxo))

    sigCh = build_channel(sigTxo, paramCh)

    # Optical power received at the fiber output
    Prx_dBm = W2dBm(signalPower(sigCh))

    rx_result = build_receiver(sigCh, bitsTx, M, SpS, paramPD,
                               discard=discard, n_train=n_train)

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
        'Ptx_dBm': Ptx_dBm,
        'Prx_dBm': Prx_dBm,
        'M'      : M,
    })
    return result


# =============================================================================
# 6. PARAMETER SWEEP FUNCTIONS
# =============================================================================
def sweep_param(param_name, param_range, fixed_params=None, verbose=True, save_path=None, sheet_name=None):
    """
    Generic parameter sweep function for IM-DD link simulations.

    Parameters
    ----------
    param_name   : str        — Target parameter name in run_link()
                                 (e.g., 'Pi_dBm', 'rx_bandwidth', 'fiber_L', 'fiber_D')
    param_range  : array-like — Sequence of values to sweep over.
    fixed_params : dict       — Fixed keyword arguments forwarded directly to run_link().
    verbose      : bool       — Display tqdm progress bar.
    save_path    : str|None   — Path to save Excel file.
    sheet_name   : str|None   — Excel sheet name (defaults to param_name if None).

    Returns
    -------
    dict:
        param_name : ndarray — Swept values
        'BER'      : ndarray — Simulated BER
        'Pb'       : ndarray — Theoretical BER
        'Q'        : ndarray — Eye Q-factor
        'Ptx_dBm'  : ndarray — Transmit power
        'Prx_dBm'  : ndarray — Received power
    """
    from tqdm import tqdm
    import pandas as pd

    if fixed_params is None:
        fixed_params = {}

    param_range = np.asarray(param_range)
    N = len(param_range)

    # Pre-allocate arrays
    BER = np.zeros(N)
    Pb  = np.zeros(N)
    Q   = np.zeros(N)
    Ptx = np.zeros(N)
    Prx = np.zeros(N)

    iterator = tqdm(enumerate(param_range), total=N,
                    desc=f'Sweep: {param_name}') if verbose else enumerate(param_range)

    for i, val in iterator:
        # Construct run arguments dynamically
        kwargs = fixed_params.copy()
        kwargs[param_name] = val

        # Handle power seed variation logic
        if param_name == 'Pi_dBm':
            kwargs.setdefault('seed', 12335 + i)
        else:
            kwargs.setdefault('seed', 12335)

        res = run_link(**kwargs)

        BER[i] = res['BER']
        Pb[i]  = res['Pb']
        Q[i]   = res['Q']
        Ptx[i] = res['Ptx_dBm']
        Prx[i] = res['Prx_dBm']

    # Package output dictionary
    res = {
        param_name: param_range,
        'BER': BER,
        'Pb': Pb,
        'Q': Q,
        'Ptx_dBm': Ptx,
        'Prx_dBm': Prx,
        'Rs': fixed_params.get('Rs', None)  # Save Rs for plot_ber_vs_bandwidth()
    }

    # Optional Excel export
    if save_path:
        df_data = {param_name: param_range, 'Ptx_dBm': Ptx, 'Prx_dBm': Prx, 'BER': BER, 'Pb': Pb, 'Q': Q}
        table = pd.DataFrame(df_data)
        save_sheet = sheet_name or f'BER-{param_name}'
        save_to_excel_sheet(table, save_path, save_sheet)

    return res



# =============================================================================
# 7. PLOT / VISUALIZATION FUNCTIONS
# =============================================================================

def plot_eye_diagrams(result, discard=50,
                      save_path_tx=None,
                      save_path_rx=None,
                      show=True,
                      dpi=300):
    """
    Plot Tx and Rx eye diagrams from a run_link() result.

    Parameters
    ----------
    result : dict
        Output dict from run_link().
    discard : int
        Number of symbols discarded at both ends.
    save_path_tx : str or None
        File path to save Tx eye diagram.
    save_path_rx : str or None
        File path to save Rx eye diagram.
    show : bool
        Whether to display the figures.
    dpi : int
        Figure resolution when saving.
    """
    SpS = result['SpS']
    sigTxo = result['sigTxo']
    I_Rx = result['I_Rx']

    paramPD_ideal = parameters()
    paramPD_ideal.ideal = True
    paramPD_ideal.Fs = result['Fs']

    I_Tx = photodiode(sigTxo.real, paramPD_ideal)

    d = discard * SpS

    # ----- Tx eye -----
    eyediagram(
        I_Tx[d:-d],
        I_Tx.size - 2 * d,
        SpS,
        plotlabel='Tx eye',
        ptype='fancy'
    )

    if save_path_tx is not None:
        plt.savefig(save_path_tx, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    # ----- Rx eye -----
    eyediagram(
        I_Rx[d:-d],
        I_Rx.size - 2 * d,
        SpS,
        plotlabel='Rx eye',
        ptype='fancy'
    )

    if save_path_rx is not None:
        plt.savefig(save_path_rx, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_ber_vs_power(
        results_list,
        labels=None,
        target_BER=None,
        title='BER vs Received Power',
        save_path=None,
        show=True,
        dpi=300):
    if labels is None:
        labels = [f'Config {i + 1}' for i in range(len(results_list))]

    plt.figure(figsize=(8, 5))
    for res, label in zip(results_list, labels):
        # Look for Prx_dBm, Pi_dBm, or legacy power
        p = res.get('Prx_dBm', res.get('Pi_dBm', res.get('power')))
        plt.plot(p, np.log10(np.clip(res['Pb'], 1e-12, 1)), '--',
                 label=f'{label} — Pb (theory)')
        plt.plot(p, np.log10(np.clip(res['BER'], 1e-12, 1)), 'o-',
                 label=f'{label} — BER (sim)')

    if target_BER is not None:
        plt.axhline(np.log10(target_BER), color='gray', linestyle=':', label=f'BER = {target_BER}')

    plt.xlabel('Received Power [dBm]')
    plt.ylabel('log$_{10}$(BER)')
    plt.title(title)
    plt.ylim(-10, 0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_ber_vs_bandwidth(
        results_list,
        labels=None,
        title='BER vs Receiver Bandwidth',
        save_path=None,
        show=True,
        normalize_bw=False,
        dpi=300):
    if labels is None:
        labels = [f'Config {i + 1}' for i in range(len(results_list))]

    plt.figure(figsize=(8, 5))

    for res, label in zip(results_list, labels):
        # Look for rx_bandwidth or legacy bandwidth
        bw_raw = res.get('rx_bandwidth', res.get('bandwidth'))
        rs = res.get('Rs', 10e9)

        norm = rs if normalize_bw else 1e9
        bw = bw_raw / norm

        plt.plot(bw, np.log10(np.clip(res['Pb'], 1e-12, 1)), '--',
                 label=f'{label} — Pb (theory)')
        plt.plot(bw, np.log10(np.clip(res['BER'], 1e-12, 1)), 'o-',
                 label=f'{label} — BER (sim)')

    if normalize_bw:
        plt.xlabel('Normalized Receiver Bandwidth (B/Rs)')
    else:
        plt.xlabel('Receiver Bandwidth (GHz)')

    plt.ylabel(r'$\log_{10}(\mathrm{BER})$')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_ber_vs_length(
        results_list,
        labels=None,
        title='BER vs Fiber Length',
        save_path=None,
        show=True,
        dpi=300):
    if labels is None:
        labels = [f'Config {i + 1}' for i in range(len(results_list))]

    plt.figure(figsize=(8, 5))
    for res, label in zip(results_list, labels):
        # Look for fiber_L or legacy length
        L = res.get('fiber_L', res.get('length'))
        plt.plot(L, np.log10(np.clip(res['Pb'], 1e-12, 1)), '--',
                 label=f'{label} — Pb (theory)')
        plt.plot(L, np.log10(np.clip(res['BER'], 1e-12, 1)), 'o-',
                 label=f'{label} — BER (sim)')

    plt.xlabel('Fiber Length [km]')
    plt.ylabel('log$_{10}$(BER)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_ber_vs_dispersion(
        results_list,
        labels=None,
        title='BER vs Fiber Dispersion',
        save_path=None,
        show=True,
        dpi=300):
    if labels is None:
        labels = [f'Config {i + 1}' for i in range(len(results_list))]

    plt.figure(figsize=(8, 5))
    for res, label in zip(results_list, labels):
        # Look for fiber_D or legacy dispersion
        d = res.get('fiber_D', res.get('dispersion'))
        plt.plot(d, np.log10(np.clip(res['Pb'], 1e-12, 1)), '--',
                 label=f'{label} — Pb (theory)')
        plt.plot(d, np.log10(np.clip(res['BER'], 1e-12, 1)), 'o-',
                 label=f'{label} — BER (sim)')

    plt.xlabel('Dispersion [ps/nm/km]')
    plt.ylabel('log$_{10}$(BER)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


# =============================================================================
# 8. UTILITY FUNCTIONS
# =============================================================================
def calculate_evm(result, M=None):
    """
    Calculate RMS EVM (%) on the payload of an IM-DD OOK/4-PAM link.

    Since I_dec in build_receiver() is only normalized by std(I_Rx), its
    offset/gain does not match the ideal PAM constellation (grayMapping).
    Here, an affine transformation estimated FROM PILOTS (result['means'],
    already available and used for estimating thr/Q) is applied to scale
    I_dec to the exact range [-1,1] / [-3,-1,1,3] before calculating EVM.
    """
    M       = M or result['M']
    I_dec   = result['I_dec']
    means   = result['means']               # ascending, same order as levels
    n_train = result['n_train']
    discard = result.get('discard', 100)    # see patch below

    levels = np.sort(grayMapping(M, 'pam').real)

    # Calibrate gain (a) + offset (b), estimated strictly from pilots
    a, b = np.polyfit(means, levels, 1)
    I_norm = a * I_dec + b

    # Payload: exclude pilots + guard symbols at both ends, aligned with BER calculation
    payload = I_norm[n_train:-discard] if discard > 0 else I_norm[n_train:]

    nearest_idx = np.argmin(np.abs(payload[:, None] - levels[None, :]), axis=1)
    I_ideal = levels[nearest_idx]

    P_avg_ideal = np.mean(levels**2)
    evm_rms = np.sqrt(np.mean((payload - I_ideal)**2) / P_avg_ideal) * 100.0

    return evm_rms

def _cluster_stats(I_dec, idx, M):
    """
    Per-level mean/std of received samples grouped by known symbol index.
    Returns (means, stds), or None if any level has < 2 samples.
    """
    means = np.empty(M)
    stds  = np.empty(M)
    for k in range(M):
        cls = I_dec[idx == k]
        if cls.size < 2:
            return None
        means[k] = cls.mean()
        stds[k]  = cls.std()
    return means, stds


def _estimate_phase_and_thresholds(I_Rx, pilot_idx, M, SpS, guard=100):
    """
    Estimate the sampling phase and decision thresholds from a known pilot.

    For each candidate phase in [0, SpS) the pilot samples are grouped by their
    known symbol level; the phase with the largest worst-case eye Q-factor
    (i.e. smallest ISI/noise penalty) is selected. Thresholds are then the
    standard optimum thresholds between adjacent levels, computed from the
    pilot means/stds at that phase.

    Parameters
    ----------
    I_Rx      : ndarray -- Normalized full-rate received waveform
    pilot_idx : ndarray -- Known pilot symbol indices (0..M-1), length n_train
    guard     : int     -- Pilot symbols skipped at the start (filter transient)

    Returns
    -------
    phase, means, stds, thr, Q  (all estimated on the pilot only)
    """
    n_train = pilot_idx.size
    idx_p = pilot_idx[guard:]

    best = None
    for phase in range(SpS):
        I_p = I_Rx[phase::SpS][guard:n_train]
        stats = _cluster_stats(I_p, idx_p[:I_p.size], M)
        if stats is None:
            continue
        means, stds = stats
        Q = np.min((means[1:] - means[:-1]) / (stds[1:] + stds[:-1]))
        if best is None or Q > best[0]:
            best = (Q, phase, means, stds)

    if best is None:
        raise ValueError(
            "Pilot too short: some symbol level has fewer than 2 training "
            "samples. Increase n_train."
        )

    Q, phase, means, stds = best
    thr = (stds[:-1] * means[1:] + stds[1:] * means[:-1]) / (stds[:-1] + stds[1:])
    thr = np.sort(thr)   # np.digitize needs monotonic thresholds (guards very low SNR)
    return phase, means, stds, thr, Q


def ber_floor(BER_array, floor=1e-12):
    """Clip a BER array to a minimum floor value to avoid log10(0)."""
    return np.clip(BER_array, floor, 1.0)

def W2dBm(P_W):
    """Convert power from Watts to dBm."""
    return 10 * np.log10(np.asarray(P_W) * 1e3)

def print_summary(result):
    """Print a concise summary of a single run_link() result."""
    mod_name = 'OOK' if result['M'] == 2 else 'PAM4'
    print(f"  Modulation : M={result['M']} ({mod_name})")
    print(f"  Pi_dBm     : {result['Pi_dBm']:.1f} dBm")
    print(f"  Prx_dBm    : {result['Prx_dBm']:.1f} dBm")
    print(f"  Symbol rate: {result['Rs']/1e9:.1f} Gbaud")
    print(f"  Q-factor   : {result['Q']:.2f}")
    print(f"  BER (sim)  : {result['BER']:.2e}")
    print(f"  Pb (theory): {result['Pb']:.2e}")

def save_to_excel_sheet(table, file_path, sheet_name):
    """Appends or updates a worksheet in an existing or new Excel workbook."""
    if not os.path.exists(file_path):
        # Create a new workbook if it doesn't exist yet
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            table.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # Append or replace the sheet in an existing workbook
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            table.to_excel(writer, sheet_name=sheet_name, index=False)

# =============================================================================
# 8. PROTOTYPING FUNCTIONS
# =============================================================================
def sweep_ber_vs_bw_and_power(bw_range, power_range, M=2, Rs=10e9, SpS=16,
                               fiber_L=10, nBits=100000, save_path=None,
                               sheet_name='BER-BW-Power', verbose=True, **kwargs):
    """
    2-D sweep: BER surface over (receiver bandwidth, transmit/received power).

    This is the "SNR-aware" extension of sweep_ber_vs_bandwidth(): instead of a
    single BER-vs-bandwidth curve at one fixed power, it runs a full family of
    curves, one per power level, and additionally extracts the bandwidth that
    minimizes BER at every power level (B_opt as a function of SNR).

    Parameters
    ----------
    bw_range    : array-like — Receiver bandwidth values [Hz],
                                e.g. np.linspace(0.3*Rs, 2*Rs, 20)
    power_range : array-like — Laser input power values [dBm] used as the SNR
                                proxy (received power Prx_dBm is recovered from
                                run_link() and stored, since it is the physically
                                meaningful SNR axis once fiber_L/alpha are fixed)
    M           : int        — Modulation order (2 or 4)
    Rs          : float      — Symbol rate [Hz]
    SpS         : int        — Samples per symbol
    fiber_L     : float      — Fiber length [km] (fixed across the sweep)
    nBits       : int        — Bits per simulation run
    save_path   : str|None   — Path to save Excel file (long-format table)
    sheet_name  : str        — Excel sheet name
    verbose     : bool       — Show tqdm progress bar
    **kwargs                 — Additional arguments forwarded to run_link()

    Returns
    -------
    dict:
        'bandwidth' : ndarray (nB,)      — Bandwidth sweep values [Hz]
        'power'     : ndarray (nP,)      — Input power sweep values [dBm]
        'Prx_dBm'   : ndarray (nP,)      — Received optical power per power row [dBm]
        'BER'       : ndarray (nP, nB)   — Simulated BER surface
        'Pb'        : ndarray (nP, nB)   — Theoretical BER surface
        'Q'         : ndarray (nP, nB)   — Q-factor surface
        'B_opt'     : ndarray (nP,)      — argmin_B BER(power, B) per power row [Hz]
        'BER_opt'   : ndarray (nP,)      — BER value at B_opt for each power row
        'Rs'        : float              — Symbol rate (for normalized-bandwidth plots)
        'M'         : int                — Modulation order
    """
    from tqdm import tqdm
    import pandas as pd

    bw_range = np.asarray(bw_range)
    power_range = np.asarray(power_range)
    nB, nP = len(bw_range), len(power_range)

    BER = np.zeros((nP, nB))
    Pb  = np.zeros((nP, nB))
    Q   = np.zeros((nP, nB))
    Prx_dBm = np.zeros(nP)
    EVM = np.zeros((nP, nB))

    total = nP * nB
    pbar = tqdm(total=total, desc='Sweep: BW x Power') if verbose else None

    for ip, Pi_dBm in enumerate(power_range):
        # Same seed reused across bandwidth values within a power row so that
        # the BW sweep is compared on the same bit/noise realization; the seed
        # still changes across power rows to avoid correlation between rows.
        for ib, bw in enumerate(bw_range):
            res = run_link(Pi_dBm=Pi_dBm, M=M, Rs=Rs, SpS=SpS,
                            fiber_L=fiber_L, rx_bandwidth=bw,
                            nBits=nBits, seed=12335 + ip, **kwargs)
            BER[ip, ib] = res['BER']
            Pb[ip, ib]  = res['Pb']
            Q[ip, ib]   = res['Q']
            EVM[ip, ib] = calculate_evm(res)
            if ib == 0:
                Prx_dBm[ip] = res['Prx_dBm']
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    # B_opt(power): bandwidth that minimizes simulated BER at each power level
    B_opt_idx = np.argmin(BER, axis=1)
    B_opt = bw_range[B_opt_idx]
    BER_opt = BER[np.arange(nP), B_opt_idx]

    result = {
        'bandwidth': bw_range,
        'power': power_range,
        'Prx_dBm': Prx_dBm,
        'BER': BER,
        'Pb': Pb,
        'Q': Q,
        'B_opt': B_opt,
        'BER_opt': BER_opt,
        'Rs': Rs,
        'M': M,
        'EVM': EVM,
    }

    # Optional Excel export (long format: one row per (power, bandwidth) pair)
    if save_path is not None:
        bw_grid, p_grid = np.meshgrid(bw_range, power_range)
        prx_grid = np.repeat(Prx_dBm[:, None], nB, axis=1)
        table = pd.DataFrame({
            'Pi_dBm': p_grid.ravel(),
            'Prx_dBm': prx_grid.ravel(),
            'Bandwidth_Hz': bw_grid.ravel(),
            'BER': BER.ravel(),
            'Pb': Pb.ravel(),
            'Q': Q.ravel(),
            'EVM': EVM.ravel()
        })
        save_to_excel_sheet(table, save_path, sheet_name)

    return result


def plot_ber_contour_bw_power(
        result,
        title='BER Map: Bandwidth vs Received Power',
        normalize_bw=True,
        use_sim=True,
        target_BER=None,
        show_Bopt=True,
        save_path=None,
        show=True,
        dpi=300):
    """
    2-D contour/heatmap of log10(BER) over (bandwidth, received power), built
    from the output of sweep_ber_vs_bw_and_power().

    This is the plot that answers "where is B_opt, and how does it move as a
    function of SNR/received power?" — a single BER-vs-bandwidth curve only
    shows a 1-D slice of this surface at one fixed power.

    Parameters
    ----------
    result       : dict — Output of sweep_ber_vs_bw_and_power()
    normalize_bw : bool — x-axis in B/Rs (True) or GHz (False)
    use_sim      : bool — Plot simulated BER surface (True) or theoretical Pb (False)
    target_BER   : float|None — If given, overlay a contour line at this BER
                                 (e.g. FEC threshold 1e-3) to show the usable
                                 (bandwidth, power) region, not just the single
                                 minimum point.
    show_Bopt    : bool — Overlay the B_opt(power) trajectory (white dashed line)
    """
    bw = result['bandwidth']
    Rs = result.get('Rs', 10e9)
    bw_axis = bw / Rs if normalize_bw else bw / 1e9

    y = result['Prx_dBm']

    Z = result['BER'] if use_sim else result['Pb']
    logZ = np.log10(np.clip(Z, 1e-12, 1))

    plt.figure(figsize=(8, 6))
    cf = plt.contourf(bw_axis, y, logZ, levels=30, cmap='viridis')
    cbar = plt.colorbar(cf)
    cbar.set_label(r'$\log_{10}(\mathrm{BER})$')

    if target_BER is not None:
        cs = plt.contour(bw_axis, y, logZ, levels=[np.log10(target_BER)],
                          colors='red', linewidths=2)
        plt.clabel(cs, fmt=lambda v: f'BER={target_BER:.0e}')

    if show_Bopt:
        B_opt_axis = result['B_opt'] / Rs if normalize_bw else result['B_opt'] / 1e9
        plt.plot(B_opt_axis, y, 'w--o', linewidth=2, markersize=4,
                 label=r'$B_{opt}$(SNR)')
        plt.legend(loc='best')

    plt.xlabel('Normalized Receiver Bandwidth (B/Rs)' if normalize_bw
               else 'Receiver Bandwidth (GHz)')
    plt.ylabel('Received Optical Power [dBm]')
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_ber_vs_bandwidth_waterfall(
        result,
        title='BER vs Bandwidth at Multiple Power Levels',
        normalize_bw=True,
        target_BER=None,
        save_path=None,
        show=True,
        dpi=300):
    """
    BER-vs-bandwidth curves for every power level in a sweep_ber_vs_bw_and_power()
    result, color-coded from low to high received power. Complements
    plot_ber_contour_bw_power(): easier to compare curve shape / floor / how the
    optimum region widens or narrows as SNR changes, at the cost of not showing
    the full continuous surface.
    """
    bw = result['bandwidth']
    Rs = result.get('Rs', 10e9)
    bw_axis = bw / Rs if normalize_bw else bw / 1e9
    Prx = result['Prx_dBm']

    plt.figure(figsize=(8, 6))
    cmap = plt.cm.viridis
    n = len(Prx)
    for i, p in enumerate(Prx):
        color = cmap(i / max(n - 1, 1))
        plt.plot(bw_axis, np.log10(np.clip(result['BER'][i], 1e-12, 1)),
                 'o-', color=color, markersize=3, label=f'Prx = {p:.1f} dBm')

    if target_BER is not None:
        plt.axhline(np.log10(target_BER), color='gray', linestyle=':',
                    label=f'BER = {target_BER:.0e}')

    plt.xlabel('Normalized Receiver Bandwidth (B/Rs)' if normalize_bw
               else 'Receiver Bandwidth (GHz)')
    plt.ylabel(r'$\log_{10}(\mathrm{BER})$')
    plt.title(title)
    plt.grid(True)
    plt.legend(fontsize=8, ncol=2, loc='best')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

def sweep_ber_vs_rate_and_power(rate_range, power_range, M=2, SpS=16,
                                 fiber_L=10, nBits=100000, save_path=None,
                                 sheet_name='BER-Rate-Power', verbose=True, **kwargs):
    """
    2-D sweep: BER surface over (symbol rate, transmit/received power).

    Analogous to sweep_ber_vs_bw_and_power(), but sweeps the symbol rate Rs
    instead of the receiver bandwidth. Since rx_bandwidth defaults to Rs
    (matched filter) whenever it isn't explicitly overridden in **kwargs,
    this traces the BER-vs-power trade-off as the link is pushed to higher
    baud rates -- the "how much power do I need to give up to go faster"
    curve family -- and extracts, for every power level, the rate that
    minimizes BER (R_opt as a function of SNR).

    Parameters
    ----------
    rate_range  : array-like -- Symbol rate values [Hz],
                                e.g. np.linspace(5e9, 30e9, 20)
    power_range : array-like -- Laser input power values [dBm] used as the SNR
                                proxy (received power Prx_dBm is recovered from
                                run_link() and stored, since it is the physically
                                meaningful SNR axis once fiber_L/alpha are fixed)
    M           : int        -- Modulation order (2 or 4)
    SpS         : int        -- Samples per symbol
    fiber_L     : float      -- Fiber length [km] (fixed across the sweep)
    nBits       : int        -- Bits per simulation run
    save_path   : str|None   -- Path to save Excel file (long-format table)
    sheet_name  : str        -- Excel sheet name
    verbose     : bool       -- Show tqdm progress bar
    **kwargs                 -- Additional arguments forwarded to run_link()
                                (e.g. rx_bandwidth= to fix an absolute
                                receiver bandwidth instead of tracking Rs)

    Returns
    -------
    dict:
        'rate'      : ndarray (nR,)      -- Symbol rate sweep values [Hz]
        'power'     : ndarray (nP,)      -- Input power sweep values [dBm]
        'Prx_dBm'   : ndarray (nP,)      -- Received optical power per power row [dBm]
        'BER'       : ndarray (nP, nR)   -- Simulated BER surface
        'Pb'        : ndarray (nP, nR)   -- Theoretical BER surface
        'Q'         : ndarray (nP, nR)   -- Q-factor surface
        'R_opt'     : ndarray (nP,)      -- argmin_R BER(power, R) per power row [Hz]
        'BER_opt'   : ndarray (nP,)      -- BER value at R_opt for each power row
        'M'         : int                -- Modulation order
    """
    from tqdm import tqdm
    import pandas as pd

    rate_range = np.asarray(rate_range)
    power_range = np.asarray(power_range)
    nR, nP = len(rate_range), len(power_range)

    BER = np.zeros((nP, nR))
    Pb  = np.zeros((nP, nR))
    Q   = np.zeros((nP, nR))
    Prx_dBm = np.zeros(nP)

    total = nP * nR
    pbar = tqdm(total=total, desc='Sweep: Rate x Power') if verbose else None

    for ip, Pi_dBm in enumerate(power_range):
        # Same seed reused across rate values within a power row so that
        # the rate sweep is compared on the same bit/noise realization; the
        # seed still changes across power rows to avoid correlation between rows.
        for ir, Rs in enumerate(rate_range):
            res = run_link(Pi_dBm=Pi_dBm, M=M, Rs=Rs, SpS=SpS,
                            fiber_L=fiber_L,
                            nBits=nBits, seed=12335 + ip, **kwargs)
            BER[ip, ir] = res['BER']
            Pb[ip, ir]  = res['Pb']
            Q[ip, ir]   = res['Q']
            if ir == 0:
                Prx_dBm[ip] = res['Prx_dBm']
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    # R_opt(power): symbol rate that minimizes simulated BER at each power level
    R_opt_idx = np.argmin(BER, axis=1)
    R_opt = rate_range[R_opt_idx]
    BER_opt = BER[np.arange(nP), R_opt_idx]

    result = {
        'rate': rate_range,
        'power': power_range,
        'Prx_dBm': Prx_dBm,
        'BER': BER,
        'Pb': Pb,
        'Q': Q,
        'R_opt': R_opt,
        'BER_opt': BER_opt,
        'M': M,
    }

    # Optional Excel export (long format: one row per (power, rate) pair)
    if save_path is not None:
        r_grid, p_grid = np.meshgrid(rate_range, power_range)
        prx_grid = np.repeat(Prx_dBm[:, None], nR, axis=1)
        table = pd.DataFrame({
            'Pi_dBm': p_grid.ravel(),
            'Prx_dBm': prx_grid.ravel(),
            'Rate_Hz': r_grid.ravel(),
            'BER': BER.ravel(),
            'Pb': Pb.ravel(),
            'Q': Q.ravel(),
        })
        save_to_excel_sheet(table, save_path, sheet_name)

    return result


def plot_ber_contour_rate_power(
        result,
        title='BER Map: Symbol Rate vs Received Power',
        rate_unit='Gbaud',
        use_sim=True,
        target_BER=None,
        show_Ropt=True,
        save_path=None,
        show=True,
        dpi=300):
    """
    2-D contour/heatmap of log10(BER) over (symbol rate, received power),
    built from the output of sweep_ber_vs_rate_and_power().

    This is the plot that answers "how much received power do I need to hit
    a target BER at a given baud rate, and where does the usable region
    shrink as I push the rate up?" -- a single BER-vs-power curve only shows
    a 1-D slice of this surface at one fixed rate.

    Parameters
    ----------
    result      : dict -- Output of sweep_ber_vs_rate_and_power()
    rate_unit   : str  -- 'Gbaud' (Rs/1e9) or 'Gbps' (Rs*log2(M)/1e9) for the x-axis
    use_sim     : bool -- Plot simulated BER surface (True) or theoretical Pb (False)
    target_BER  : float|None -- If given, overlay a contour line at this BER
                                (e.g. FEC threshold 1e-3) to show the usable
                                (rate, power) region, not just the single
                                minimum point.
    show_Ropt   : bool -- Overlay the R_opt(power) trajectory (white dashed line)
    """
    Rs_arr = result['rate']
    M = result.get('M', 2)
    bits_per_symb = np.log2(M)

    if rate_unit.lower() == 'gbps':
        scale = bits_per_symb / 1e9
        xlabel = 'Bit Rate (Gbps)'
    else:
        scale = 1 / 1e9
        xlabel = 'Symbol Rate (Gbaud)'

    rate_axis = Rs_arr * scale
    y = result['Prx_dBm']

    Z = result['BER'] if use_sim else result['Pb']
    logZ = np.log10(np.clip(Z, 1e-12, 1))

    plt.figure(figsize=(8, 6))
    cf = plt.contourf(rate_axis, y, logZ, levels=30, cmap='viridis')
    cbar = plt.colorbar(cf)
    cbar.set_label(r'$\log_{10}(\mathrm{BER})$')

    if target_BER is not None:
        cs = plt.contour(rate_axis, y, logZ, levels=[np.log10(target_BER)],
                          colors='red', linewidths=2)
        plt.clabel(cs, fmt=lambda v: f'BER={target_BER:.0e}')

    if show_Ropt:
        R_opt_axis = result['R_opt'] * scale
        plt.plot(R_opt_axis, y, 'w--o', linewidth=2, markersize=4,
                 label=r'$R_{opt}$(SNR)')
        plt.legend(loc='best')

    plt.xlabel(xlabel)
    plt.ylabel('Received Optical Power [dBm]')
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_ber_vs_rate_waterfall(
        result,
        title=None,
        x_axis='power',
        rate_unit='Gbaud',
        target_BER=None,
        save_path=None,
        show=True,
        dpi=300):
    """
    Waterfall of BER curves from a sweep_ber_vs_rate_and_power() result.
    Complements plot_ber_contour_rate_power(): easier to compare curve
    shape/floor at the cost of not showing the full continuous surface.

    Parameters
    ----------
    result     : dict — Output of sweep_ber_vs_rate_and_power()
    x_axis     : str  — 'power' (default): x = received optical power
                                [dBm], one curve per swept rate, color-coded
                                low-to-high rate. This is the classic BER
                                waterfall (BER vs Prx).
                         'rate' : x = symbol/bit rate, one curve per swept
                                power, color-coded low-to-high power.
    rate_unit  : str  — 'Gbaud' (Rs/1e9) or 'Gbps' (Rs*log2(M)/1e9), used
                         whenever rate appears (as the x-axis or in labels).
    target_BER : float|None — If given, overlay a horizontal BER threshold line.
    """
    Rs_arr = result['rate']
    M = result.get('M', 2)
    bits_per_symb = np.log2(M)
    Prx = result['Prx_dBm']
    BER = result['BER']

    if rate_unit.lower() == 'gbps':
        rate_scale = bits_per_symb / 1e9
        rate_label = 'Bit Rate (Gbps)'
    else:
        rate_scale = 1 / 1e9
        rate_label = 'Symbol Rate (Gbaud)'
    rate_axis = Rs_arr * rate_scale

    plt.figure(figsize=(8, 6))
    cmap = plt.cm.viridis

    if x_axis.lower() == 'power':
        # x = Prx_dBm, one curve per rate
        n = len(rate_axis)
        for i in range(n):
            color = cmap(i / max(n - 1, 1))
            plt.plot(Prx, np.log10(np.clip(BER[:, i], 1e-12, 1)),
                     'o-', color=color, markersize=3,
                     label=f'{rate_axis[i]:.1f} {rate_unit}')
        xlabel = 'Received Optical Power [dBm]'
        default_title = 'BER vs Received Power at Multiple Rates'
    else:
        # x = rate, one curve per power (original behavior)
        n = len(Prx)
        for i, p in enumerate(Prx):
            color = cmap(i / max(n - 1, 1))
            plt.plot(rate_axis, np.log10(np.clip(BER[i], 1e-12, 1)),
                     'o-', color=color, markersize=3, label=f'Prx = {p:.1f} dBm')
        xlabel = rate_label
        default_title = 'BER vs Symbol Rate at Multiple Power Levels'

    if target_BER is not None:
        plt.axhline(np.log10(target_BER), color='gray', linestyle=':',
                    label=f'BER = {target_BER:.0e}')

    plt.xlabel(xlabel)
    plt.ylabel(r'$\log_{10}(\mathrm{BER})$')
    plt.title(title if title is not None else default_title)
    plt.grid(True)
    plt.legend(fontsize=8, ncol=2, loc='best')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

def sweep_dlimit_vs_length_and_rate(length_range, rate_range, M=2,
                                     save_path=None,
                                     sheet_name='Dlimit-Length-Rate'):
    """
    2-D map of the dispersion-limited tolerance D_limit(length, rate),
    computed analytically (no link simulation) from the classical GVD
    pulse-broadening rule of thumb:

        D_limit = 100000 / (Rs_Gbaud^2 * L)

    where Rs_Gbaud is the *symbol* rate in Gbaud, L is the fiber length in
    km, and D_limit comes out in ps/nm/km. Equivalently, this is the usual
    dispersion-limited-distance relation L_max = 1e5 / (D * Rs^2) solved
    for D instead of L.

    IMPORTANT: the rule of thumb bounds pulse broadening relative to the
    *symbol* (baud) period, not the bit period, so it must be applied to
    Rs, not to the bit rate Rs*log2(M). This is what lets M-PAM show its
    expected advantage over OOK at equal bit rate: at the same bit rate,
    M-PAM's lower baud rate gives longer symbols and therefore a higher
    D_limit than OOK. M is kept only as metadata for converting the rate
    axis to Gbps in plot_dlimit_waterfall() -- it does not enter this
    formula.

    This plays the same role as sweep_ber_vs_rate_and_power() /
    sweep_dlimit_vs_length_and_rate()'s simulation-based counterpart, but
    is a closed-form estimate rather than a BER sweep -- useful as a fast
    sanity-check curve to compare against simulated results.

    Parameters
    ----------
    length_range : array-like -- Fiber length values [km], e.g. np.linspace(1, 80, 10)
    rate_range    : array-like -- Symbol rate values [Hz], e.g. np.linspace(5e9, 30e9, 6)
    M             : int        -- Modulation order (2 or 4); stored as metadata for
                                   Gbps-axis conversion in plot_dlimit_waterfall(),
                                   not used in the D_limit formula itself.
    save_path     : str|None   -- Path to save Excel file (long-format D_limit table)
    sheet_name    : str        -- Excel sheet name

    Returns
    -------
    dict:
        'length'  : ndarray (nL,)    -- Fiber length sweep values [km]
        'rate'    : ndarray (nR,)    -- Symbol rate sweep values [Hz]
        'D_limit' : ndarray (nL, nR) -- Dispersion tolerance [ps/nm/km]
        'M'       : int
    """
    import pandas as pd

    length_range = np.asarray(length_range)
    rate_range   = np.asarray(rate_range)

    Rs_Gbaud = rate_range / 1e9    # symbol rate in Gbaud, shape (nR,)
    L_km     = length_range        # shape (nL,)

    # D_limit[il, ir] = 1e5 / (Rs_Gbaud[ir]^2 * L_km[il])
    D_limit = 1e5 / (Rs_Gbaud[None, :] ** 2 * L_km[:, None])

    result = {
        'length': length_range,
        'rate': rate_range,
        'D_limit': D_limit,
        'M': M,
    }

    # Optional Excel export (long format: one row per (length, rate) pair)
    if save_path is not None:
        l_grid, r_grid = np.meshgrid(length_range, rate_range, indexing='ij')
        table = pd.DataFrame({
            'Length_km': l_grid.ravel(),
            'Rate_Hz': r_grid.ravel(),
            'D_limit_ps_nm_km': D_limit.ravel(),
        })
        save_to_excel_sheet(table, save_path, sheet_name)

    return result


def plot_dlimit_waterfall(
        result,
        title=None,
        x_axis='length',
        rate_unit='Gbaud',
        save_path=None,
        show=True,
        dpi=300):
    """
    Waterfall of dispersion-tolerance curves from a
    sweep_dlimit_vs_length_and_rate() result.

    Mirrors plot_ber_vs_rate_waterfall(): instead of log10(BER) on the
    y-axis, this plots D_limit = 1e5 / (B_Gbps^2 * L_km) -- the max
    tolerable dispersion parameter under the GVD pulse-broadening rule of
    thumb -- giving the classic "dispersion-limited reach" family of curves.

    Parameters
    ----------
    result     : dict -- Output of sweep_dlimit_vs_length_and_rate()
    x_axis     : str  -- 'length' (default): x = fiber length [km], one curve
                                per swept rate, color-coded low-to-high rate.
                         'rate'  : x = symbol/bit rate, one curve per swept
                                length, color-coded low-to-high length.
    rate_unit  : str  -- 'Gbaud' (Rs/1e9) or 'Gbps' (Rs*log2(M)/1e9), used
                         whenever rate appears (as the x-axis or in labels).
    """
    length_axis = result['length']
    Rs_arr = result['rate']
    M = result.get('M', 2)
    bits_per_symb = np.log2(M)
    D_limit = result['D_limit']

    if rate_unit.lower() == 'gbps':
        rate_scale = bits_per_symb / 1e9
        rate_label = 'Bit Rate (Gbps)'
    else:
        rate_scale = 1 / 1e9
        rate_label = 'Symbol Rate (Gbaud)'
    rate_axis = Rs_arr * rate_scale

    plt.figure(figsize=(8, 6))
    cmap = plt.cm.viridis

    if x_axis.lower() == 'length':
        # x = fiber length, one curve per rate
        n = len(rate_axis)
        for i in range(n):
            color = cmap(i / max(n - 1, 1))
            plt.plot(length_axis, D_limit[:, i],
                     'o-', color=color, markersize=3,
                     label=f'{rate_axis[i]:.1f} {rate_unit}')
        xlabel = 'Fiber Length [km]'
        default_title = 'Dispersion Tolerance vs Length at Multiple Rates'
    else:
        # x = rate, one curve per length
        n = len(length_axis)
        for i, L in enumerate(length_axis):
            color = cmap(i / max(n - 1, 1))
            plt.plot(rate_axis, D_limit[i, :],
                     'o-', color=color, markersize=3, label=f'L = {L:.1f} km')
        xlabel = rate_label
        default_title = 'Dispersion Tolerance vs Rate at Multiple Lengths'

    plt.xlabel(xlabel)
    plt.ylabel(r'$D_{limit}$ [ps/nm/km]')
    plt.title(title if title is not None else default_title)
    plt.grid(True)
    plt.legend(fontsize=8, ncol=2, loc='best')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def sweep_lmax_vs_rate_and_power(
        Rb_range_Gbps, Pi_range_dBm, M=2, SpS=16,
        fiber_alpha=0.2, fiber_D=18, Fc=193.1e12,
        nBits=200000, target_BER=1e-3,
        L_min=0.5, L_max_search=100.0, L_tol=0.25, max_iter=25,
        verbose=True, save_path=None, sheet_name='Lmax-Rb-Pi', **kwargs):
    """
    Determine the maximum transmission distance Lmax(Rb, Pi) for which the
    system still meets the target quality level (BER <= target_BER), swept
    jointly over:

        - Bit rate Rb    [Gb/s]  (row axis, Rb_range_Gbps)
        - Launch power Pi [dBm]  (column axis, Pi_range_dBm)

    at a fixed fiber dispersion parameter D (default D = 18 ps/nm/km, the
    realistic-case value).

    Approach: BER increases monotonically with fiber length L (attenuation
    reduces received power -> lower SNR, while dispersion-induced ISI also
    grows with L), so for every (Rb, Pi) pair a bisection search on L is
    used to locate the point where BER(L) crosses target_BER, instead of a
    dense (and much more expensive) sweep over L.

    Parameters
    ----------
    Rb_range_Gbps : array-like -- Bit rate values to sweep [Gb/s], e.g. 10 -> 100 Gb/s
    Pi_range_dBm  : array-like -- Launch power (MZM input) values to sweep [dBm],
                                   e.g. -14 -> 0 dBm
    M             : int        -- Modulation order (2 = OOK, 4 = PAM4).
                                   Rs = Rb / log2(M) is derived from Rb.
    SpS           : int        -- Samples per symbol
    fiber_alpha   : float      -- Fiber attenuation [dB/km]
    fiber_D       : float      -- Fiber dispersion parameter [ps/nm/km]
                                   (default 18 -- realistic case)
    Fc            : float      -- Central optical frequency [Hz]
    nBits         : int        -- Number of bits simulated per run_link() call
                                   (should be large enough for a reliable
                                   BER ~ 1e-3 estimate)
    target_BER    : float      -- Target BER threshold (default 1e-3)
    L_min         : float      -- Lower search bound for L [km]
    L_max_search  : float      -- Upper search bound for L [km]. If BER is
                                   still <= target_BER right at L_max_search,
                                   Lmax is treated as censored (right-bounded)
                                   at this value -- see is_censored.
    L_tol         : float      -- Length resolution at which bisection stops [km]
    max_iter      : int        -- Maximum number of bisection iterations
    verbose       : bool       -- Show tqdm progress bar
    save_path     : str|None   -- Path to save Excel file (long-format table)
    sheet_name    : str        -- Excel sheet name
    **kwargs                   -- Additional arguments forwarded to run_link()
                                   (e.g. rx_bandwidth=, rx_ideal=, ...)

    Returns
    -------
    dict:
        'Rb_Gbps'     : ndarray (nRb,)      -- Swept bit-rate axis [Gb/s]
        'Pi_dBm'      : ndarray (nPi,)      -- Swept launch-power axis [dBm]
        'Lmax_km'     : ndarray (nRb, nPi)  -- Maximum transmission distance [km]
        'BER_at_Lmax' : ndarray (nRb, nPi)  -- Simulated BER at Lmax
        'is_censored' : ndarray (nRb, nPi) bool
                                             -- True if Lmax is right-bounded by
                                                L_max_search (target BER still
                                                met at the search's upper bound)
                                                rather than a true crossing point
        'M'           : int
        'fiber_D'     : float
        'target_BER'  : float
    """
    from tqdm import tqdm
    import pandas as pd

    Rb_range_Gbps = np.asarray(Rb_range_Gbps, dtype=float)
    Pi_range_dBm = np.asarray(Pi_range_dBm, dtype=float)
    nRb, nPi = len(Rb_range_Gbps), len(Pi_range_dBm)

    bits_per_symb = np.log2(M)

    Lmax = np.full((nRb, nPi), np.nan)
    BER_at_Lmax = np.full((nRb, nPi), np.nan)
    is_censored = np.zeros((nRb, nPi), dtype=bool)

    total = nRb * nPi
    pbar = tqdm(total=total, desc='Sweep: Lmax(Rb, Pi)') if verbose else None

    for ib, Rb_Gbps in enumerate(Rb_range_Gbps):
        Rs = Rb_Gbps * 1e9 / bits_per_symb  # symbol rate [Hz] corresponding to Rb

        for ip, Pi_dBm in enumerate(Pi_range_dBm):
            # Seed fixed per (Rb, Pi) grid cell so the bisection over L reuses
            # the same bit/noise realization -> a cleaner monotonic BER(L).
            seed = 12335 + ib * nPi + ip

            def ber_at_length(L, _Rs=Rs, _Pi=Pi_dBm, _seed=seed):
                res = run_link(
                    Pi_dBm=_Pi, M=M, SpS=SpS, Rs=_Rs,
                    fiber_L=L, fiber_alpha=fiber_alpha, fiber_D=fiber_D,
                    Fc=Fc, nBits=nBits, seed=_seed, **kwargs
                )
                return res['BER']

            # --- Lower bound: if BER already exceeds the target at L_min,
            # the system cannot meet the target BER at any distance -> Lmax = 0
            ber_lo = ber_at_length(L_min)
            if ber_lo > target_BER:
                Lmax[ib, ip] = 0.0
                BER_at_Lmax[ib, ip] = ber_lo
                if pbar is not None:
                    pbar.update(1)
                continue

            # --- Upper bound: if BER still meets the target at L_max_search,
            # Lmax >= L_max_search (censored / right-bounded result)
            lo, hi = L_min, L_max_search
            ber_hi = ber_at_length(hi)
            if ber_hi <= target_BER:
                Lmax[ib, ip] = hi
                BER_at_Lmax[ib, ip] = ber_hi
                is_censored[ib, ip] = True
                if pbar is not None:
                    pbar.update(1)
                continue

            # --- Bisection search for the point where BER(L) = target_BER ---
            for _ in range(max_iter):
                if hi - lo <= L_tol:
                    break
                mid = 0.5 * (lo + hi)
                ber_mid = ber_at_length(mid)
                if ber_mid <= target_BER:
                    lo = mid
                else:
                    hi = mid

            Lmax[ib, ip] = lo
            BER_at_Lmax[ib, ip] = ber_at_length(lo)

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    result = {
        'Rb_Gbps': Rb_range_Gbps,
        'Pi_dBm': Pi_range_dBm,
        'Lmax_km': Lmax,
        'BER_at_Lmax': BER_at_Lmax,
        'is_censored': is_censored,
        'M': M,
        'fiber_D': fiber_D,
        'target_BER': target_BER,
    }

    # Optional Excel export (long format: one row per (Rb, Pi) pair)
    if save_path is not None:
        rb_grid, pi_grid = np.meshgrid(Rb_range_Gbps, Pi_range_dBm, indexing='ij')
        table = pd.DataFrame({
            'Rb_Gbps': rb_grid.ravel(),
            'Pi_dBm': pi_grid.ravel(),
            'Lmax_km': Lmax.ravel(),
            'BER_at_Lmax': BER_at_Lmax.ravel(),
            'is_censored': is_censored.ravel(),
        })
        save_to_excel_sheet(table, save_path, sheet_name)

    return result


def plot_lmax_waterfall(
        result,
        title=None,
        x_axis='rate',
        save_path=None,
        show=True,
        dpi=300):
    """
    Plot a waterfall of Lmax curves from a sweep_lmax_vs_rate_and_power() result.

    Parameters
    ----------
    result  : dict -- Output of sweep_lmax_vs_rate_and_power()
    x_axis  : str  -- 'rate'  (default): x = bit rate Rb [Gb/s], one curve
                               per launch power Pi, color-coded low-to-high.
                       'power': x = launch power Pi [dBm], one curve per
                               bit rate Rb.
    """
    Rb = result['Rb_Gbps']
    Pi = result['Pi_dBm']
    Lmax = result['Lmax_km']
    target_BER = result.get('target_BER')
    D = result.get('fiber_D')

    plt.figure(figsize=(8, 6))
    cmap = plt.cm.viridis

    if x_axis.lower() == 'power':
        n = len(Rb)
        for i in range(n):
            color = cmap(i / max(n - 1, 1))
            plt.plot(Pi, Lmax[i, :], 'o-', color=color, markersize=3,
                     label=f'Rb = {Rb[i]:.0f} Gb/s')
        xlabel = 'Launch Power Pi [dBm]'
        default_title = 'Maximum Reach Lmax vs Launch Power'
    else:
        n = len(Pi)
        for i, p in enumerate(Pi):
            color = cmap(i / max(n - 1, 1))
            plt.plot(Rb, Lmax[:, i], 'o-', color=color, markersize=3,
                     label=f'Pi = {p:.1f} dBm')
        xlabel = 'Bit Rate Rb [Gb/s]'
        default_title = 'Maximum Reach Lmax vs Bit Rate'

    subtitle = ''
    if D is not None:
        subtitle += f' (D = {D:.0f} ps/nm/km'
    if target_BER is not None:
        subtitle += f', target BER = {target_BER:.0e}' if subtitle else \
            f' (target BER = {target_BER:.0e}'
    if subtitle:
        subtitle += ')'

    plt.xlabel(xlabel)
    plt.ylabel(r'$L_{max}$ [km]')
    plt.title((title if title is not None else default_title) + subtitle)
    plt.grid(True)
    plt.legend(fontsize=8, ncol=2, loc='best')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_lmax_contour(
        result,
        title=None,
        save_path=None,
        show=True,
        dpi=300):
    """
    Plot a contour/heatmap map of Lmax(Rb, Pi) from a
    sweep_lmax_vs_rate_and_power() result. Gives an overall view of the
    usable (Rb, Pi) region and the corresponding maximum reach across the
    full swept plane, instead of only 1-D slices as in plot_lmax_waterfall().
    """
    Rb = result['Rb_Gbps']
    Pi = result['Pi_dBm']
    Lmax = result['Lmax_km']
    D = result.get('fiber_D')
    target_BER = result.get('target_BER')

    plt.figure(figsize=(8, 6))
    cf = plt.contourf(Rb, Pi, Lmax.T, levels=30, cmap='viridis')
    cbar = plt.colorbar(cf)
    cbar.set_label(r'$L_{max}$ [km]')

    plt.xlabel('Bit Rate Rb [Gb/s]')
    plt.ylabel('Launch Power Pi [dBm]')

    default_title = 'Maximum Reach Map Lmax(Rb, Pi)'
    if D is not None:
        default_title += f'\nD = {D:.0f} ps/nm/km'
    if target_BER is not None:
        default_title += f', target BER = {target_BER:.0e}'
    plt.title(title if title is not None else default_title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

def plot_evm_vs_bandwidth_waterfall(result, title='EVM vs Bandwidth at Multiple Power Levels',
                                     normalize_bw=True, save_path=None, show=True, dpi=300):
    bw = result['bandwidth']; Rs = result.get('Rs', 10e9)
    bw_axis = bw / Rs if normalize_bw else bw / 1e9
    Prx = result['Prx_dBm']

    plt.figure(figsize=(8, 6))
    cmap = plt.cm.viridis; n = len(Prx)
    for i, p in enumerate(Prx):
        plt.plot(bw_axis, result['EVM'][i], 'o-', color=cmap(i / max(n-1,1)),
                 markersize=3, label=f'Prx = {p:.1f} dBm')
    plt.xlabel('Normalized Receiver Bandwidth (B/Rs)' if normalize_bw else 'Receiver Bandwidth (GHz)')
    plt.ylabel('EVM (%)'); plt.title(title); plt.grid(True)
    plt.legend(fontsize=8, ncol=2, loc='best'); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show() if show else plt.close()