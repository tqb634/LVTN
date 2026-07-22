"""
run_all_experiments.py
=======================
IM-DD Optical Access Link — Trade-off Study (OOK vs 4-PAM)

Standalone, reproducible version of `main_simulation.ipynb`.
All signal-processing logic lives in `imdd_lib.py` (must be in the same
folder, or importable on PYTHONPATH) — this script only sets parameters,
calls the simulation/sweep functions, and saves every figure to disk so
thesis results can be reproduced with a single command.

Usage
-----
    python run_all_experiments.py                # full run (matches notebook main_simulation.ipynb)
    python run_all_experiments.py --quick        # fast smoke-test (small nBits, coarse sweeps)
    python run_all_experiments.py --output-dir results --show

See `--help` for all options.

Structure (mirrors the notebook)
---------------------------------
    1. Setup
    2. Global simulation parameters
    3. Single-run sanity check + eye diagrams
    4. Sweep 1 — BER vs Received Power        (OOK vs PAM4)
    5. Sweep 2 — BER vs Receiver Bandwidth     (OOK vs PAM4)
    6. Sweep 3 — BER vs Fiber Length           (OOK vs PAM4)
    7. Sweep 4 — BER vs Fiber Dispersion       (OOK vs PAM4)
    8. Trade-off summary table (TODO)
"""

# =============================================================================
# 1. SETUP
# =============================================================================
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

# Use a non-interactive backend by default so the script also runs headless
# (e.g. on a server/CI). Re-selected below if --show is passed.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from imdd_lib import (
        run_link,
        sweep_ber_vs_power,
        sweep_ber_vs_bandwidth,
        sweep_ber_vs_fiber_length,
        sweep_ber_vs_dispersion,
        plot_eye_diagrams,
        plot_ber_vs_power,
        plot_ber_vs_bandwidth,
        plot_ber_vs_length,
        plot_ber_vs_dispersion,
        print_summary,
    )
except ImportError as e:
    sys.exit(
        "ERROR: could not import 'imdd_lib'.\n"
        "Make sure imdd_lib.py is in the same directory as this script "
        "(or on your PYTHONPATH), and that OptiCommPy is installed:\n"
        "    pip install -r requirements.txt\n"
        "    pip install OptiCommPy\n"
        f"Original error: {e}"
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Reproduce all IM-DD OOK vs PAM4 trade-off simulations and figures."
    )
    p.add_argument(
        "--output-dir", "-o", type=str, default="results/figures/",
        help="Directory where all figures (and this run's log) are saved",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Fast smoke-test mode: much smaller nBits and coarser sweeps. "
             "Use this to quickly check the pipeline runs end-to-end; "
             "BER estimates will be noisy/unreliable in this mode.",
    )
    p.add_argument(
        "--show", action="store_true",
        help="Also display figures interactively (in addition to saving them). "
             "Requires a display backend; leave off for headless/server runs.",
    )
    p.add_argument(
        "--skip-sanity", action="store_true",
        help="Skip section 3 (single-run sanity check + eye diagrams).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.show:
        matplotlib.use("TkAgg", force=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run_all_experiments] Saving all figures to: {out_dir.resolve()}")

    t_start = time.time()

    # =========================================================================
    # 2. GLOBAL SIMULATION PARAMETERS
    # =========================================================================
    # ── Simulation resolution ────────────────────────────────────────────────
    SpS = 16  # samples per symbol
    nBits = 5_000 if args.quick else 100_000  # bits per simulation run

    # ── Signal parameters ────────────────────────────────────────────────────
    Rs = 10e9  # symbol rate [Hz]  ->  10 Gbaud
    Rs_4PAM = Rs / 2

    # ── Fiber parameters (defaults — overridden per sweep) ───────────────────
    FIBER_L = 10       # fiber length [km]
    FIBER_ALPHA = 0.2  # attenuation [dB/km]
    FIBER_D = 16       # dispersion  [ps/nm/km]
    Fc = 193.1e12      # central optical frequency [Hz]

    # ── Receiver parameters (defaults) ───────────────────────────────────────
    RX_BW = Rs             # matched-filter bandwidth (B = Rs)
    RX_BW_4PAM = Rs / 2

    # ── BER target reference line ────────────────────────────────────────────
    BER_TARGET = 1e-3  # FEC threshold for access links

    print(f"Symbol rate OOK   : {Rs/1e9:.0f} Gbaud")
    print(f"Symbol rate 4-PAM : {Rs_4PAM/1e9:.0f} Gbaud")
    print(f"Bit rate OOK      : {Rs/1e9:.0f} Gb/s")
    print(f"Bit rate PAM4     : {2*Rs_4PAM/1e9:.0f} Gb/s")
    print(f"Fiber             : {FIBER_L} km, alpha={FIBER_ALPHA} dB/km, D={FIBER_D} ps/nm/km")
    if args.quick:
        print(f"[quick mode] nBits reduced to {nBits} — BER results will be noisy.")

    # =========================================================================
    # 3. SINGLE-RUN SANITY CHECK + EYE DIAGRAMS
    # =========================================================================
    if not args.skip_sanity:
        print("\n" + "=" * 70)
        print("3. Single-run sanity check + eye diagrams")
        print("=" * 70)

        Pi_test = -10  # [dBm] — power level with expected BER > 0

        # ── OOK single run ───────────────────────────────────────────────────
        res_ook = run_link(
            Pi_dBm=Pi_test,
            M=2,
            SpS=SpS,
            Rs=Rs,
            fiber_L=FIBER_L,
            fiber_alpha=FIBER_ALPHA,
            fiber_D=FIBER_D,
            Fc=Fc,
            rx_bandwidth=RX_BW,
            rx_ideal=False,
            nBits=nBits,
            seed=12335,
        )
        print("\n=== OOK single run ===")
        print_summary(res_ook)

        plot_eye_diagrams(
            res_ook,
            discard=50,
            save_path_tx=str(out_dir / "03_eye_tx_ook.png"),
            save_path_rx=str(out_dir / "03_eye_rx_ook.png"),
            show=args.show,
        )

        # ── PAM4 single run ──────────────────────────────────────────────────
        res_pam4 = run_link(
            Pi_dBm=Pi_test,
            M=4,
            SpS=SpS,
            Rs=Rs_4PAM,
            fiber_L=FIBER_L,
            fiber_alpha=FIBER_ALPHA,
            fiber_D=FIBER_D,
            Fc=Fc,
            rx_bandwidth=RX_BW_4PAM,
            rx_ideal=False,
            nBits=nBits,
            seed=12335,
        )
        print("\n=== PAM4 single run ===")
        print_summary(res_pam4)

        plot_eye_diagrams(
            res_pam4,
            discard=50,
            save_path_tx=str(out_dir / "03_eye_tx_pam4.png"),
            save_path_rx=str(out_dir / "03_eye_rx_pam4.png"),
            show=args.show,
        )

    # =========================================================================
    # 4. SWEEP 1 — BER vs RECEIVED POWER
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. Sweep 1 — BER vs Received Power")
    print("=" * 70)

    power_range = np.arange(-30, -10, 1) if not args.quick else np.arange(-30, -10, 4)  # [dBm]

    sweep1_ook = sweep_ber_vs_power(
        power_range=power_range,
        M=2,
        Rs=Rs,
        SpS=SpS,
        fiber_L=FIBER_L,
        fiber_alpha=FIBER_ALPHA,
        fiber_D=FIBER_D,
        Fc=Fc,
        rx_bandwidth=RX_BW,
        nBits=nBits,
    )

    sweep1_pam4 = sweep_ber_vs_power(
        power_range=power_range,
        M=4,
        Rs=Rs_4PAM,
        SpS=SpS,
        fiber_L=FIBER_L,
        fiber_alpha=FIBER_ALPHA,
        fiber_D=FIBER_D,
        Fc=Fc,
        rx_bandwidth=RX_BW_4PAM,
        nBits=nBits,
    )

    plot_ber_vs_power(
        results_list=[sweep1_ook, sweep1_pam4],
        labels=["OOK", "PAM4"],
        title=f"BER vs Received Power  |  L={FIBER_L} km, B={RX_BW/1e9:.1f} GHz",
        target_BER=BER_TARGET,
        save_path=str(out_dir / "04_ber_vs_power.png"),
        show=args.show,
    )

    # =========================================================================
    # 5. SWEEP 2 — BER vs RECEIVER BANDWIDTH
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. Sweep 2 — BER vs Receiver Bandwidth")
    print("=" * 70)

    n_bw_pts = 8 if args.quick else 30
    bw_range = np.linspace(0.3 * Rs, 2.0 * Rs, n_bw_pts)  # [Hz]
    Pi_bw = -16  # [dBm] — fixed operating power for this sweep

    sweep2_ook = sweep_ber_vs_bandwidth(
        bw_range=bw_range,
        Pi_dBm=Pi_bw,
        M=2,
        Rs=Rs,
        SpS=SpS,
        fiber_L=FIBER_L,
        fiber_alpha=FIBER_ALPHA,
        fiber_D=FIBER_D,
        Fc=Fc,
        nBits=nBits,
    )

    sweep2_pam4 = sweep_ber_vs_bandwidth(
        bw_range=bw_range,
        Pi_dBm=Pi_bw,
        M=4,
        Rs=Rs_4PAM,
        SpS=SpS,
        fiber_L=FIBER_L,
        fiber_alpha=FIBER_ALPHA,
        fiber_D=FIBER_D,
        Fc=Fc,
        nBits=nBits,
    )

    plot_ber_vs_bandwidth(
        results_list=[sweep2_ook, sweep2_pam4],
        labels=["OOK", "PAM4"],
        title=f"BER vs Receiver Bandwidth  |  Pi={Pi_bw} dBm, L={FIBER_L} km",
        save_path=str(out_dir / "05_ber_vs_bandwidth.png"),
        show=args.show,
    )

    # =========================================================================
    # 6. SWEEP 3 — BER vs FIBER LENGTH
    # =========================================================================
    print("\n" + "=" * 70)
    print("6. Sweep 3 — BER vs Fiber Length")
    print("=" * 70)

    length_range = np.arange(0, 85, 5) if not args.quick else np.arange(0, 85, 20)  # [km]
    Pi_len = -10  # [dBm] — fixed transmit power for this sweep

    sweep3_ook = sweep_ber_vs_fiber_length(
        length_range=length_range,
        Pi_dBm=Pi_len,
        M=2,
        Rs=Rs,
        SpS=SpS,
        fiber_alpha=FIBER_ALPHA,
        fiber_D=FIBER_D,
        Fc=Fc,
        rx_bandwidth=RX_BW,
        nBits=nBits,
    )

    sweep3_pam4 = sweep_ber_vs_fiber_length(
        length_range=length_range,
        Pi_dBm=Pi_len,
        M=4,
        Rs=Rs_4PAM,
        SpS=SpS,
        fiber_alpha=FIBER_ALPHA,
        fiber_D=FIBER_D,
        Fc=Fc,
        rx_bandwidth=RX_BW,
        nBits=nBits,
    )

    plot_ber_vs_length(
        results_list=[sweep3_ook, sweep3_pam4],
        labels=["OOK", "4PAM"],
        title=f"BER vs Fiber Length  |  Pi={Pi_len} dBm, B={RX_BW/1e9:.1f} GHz",
        save_path=str(out_dir / "06_ber_vs_length.png"),
        show=args.show,
    )

    # =========================================================================
    # 7. SWEEP 4 — BER vs FIBER DISPERSION
    # =========================================================================
    print("\n" + "=" * 70)
    print("7. Sweep 4 — BER vs Fiber Dispersion")
    print("=" * 70)

    dispersion_range = np.arange(0, 21, 1) if not args.quick else np.arange(0, 21, 5)  # [ps/nm/km]
    Pi_disp = -10  # [dBm]

    sweep4_ook = sweep_ber_vs_dispersion(
        dispersion_range=dispersion_range,
        Pi_dBm=Pi_disp,
        M=2,
        Rs=Rs,
        SpS=SpS,
        fiber_L=FIBER_L,
        fiber_alpha=FIBER_ALPHA,
        Fc=Fc,
        rx_bandwidth=RX_BW,
        nBits=nBits,
    )

    sweep4_pam4 = sweep_ber_vs_dispersion(
        dispersion_range=dispersion_range,
        Pi_dBm=Pi_disp,
        M=4,
        Rs=Rs_4PAM,
        SpS=SpS,
        fiber_L=FIBER_L,
        fiber_alpha=FIBER_ALPHA,
        Fc=Fc,
        rx_bandwidth=RX_BW,
        nBits=nBits,
    )

    plot_ber_vs_dispersion(
        results_list=[sweep4_ook, sweep4_pam4],
        labels=["OOK", "4PAM"],
        title=f"BER vs Dispersion  |  Pi={Pi_disp} dBm, L={FIBER_L} km",
        save_path=str(out_dir / "07_ber_vs_dispersion.png"),
        show=args.show,
    )

    # =========================================================================
    # 8. TRADE-OFF SUMMARY TABLE
    # =========================================================================
    # NOTE: kept exactly as TODO in the source notebook — the summary-table
    # logic was never implemented there (PAM4 metrics are placeholders / NaN).
    # Left commented out here on purpose so this script's output matches the
    # notebook 1:1. Uncomment and finish `find_sensitivity` / `find_max_reach`
    # / `find_opt_bandwidth` below once that logic is ready.
    #
    # import pandas as pd
    #
    # def find_sensitivity(sweep_result, ber_target=1e-3):
    #     """
    #     Find the minimum power [dBm] at which BER <= ber_target.
    #     Returns NaN if target is never reached within the sweep range.
    #     """
    #     power = sweep_result['power']
    #     ber   = sweep_result['BER']
    #     mask  = ber <= ber_target
    #     return float(power[mask][0]) if mask.any() else float('nan')
    #
    #
    # def find_max_reach(sweep_result, ber_target=1e-3):
    #     """
    #     Find the maximum fiber length [km] at which BER <= ber_target.
    #     Returns NaN if target is never met.
    #     """
    #     length = sweep_result['length']
    #     ber    = sweep_result['BER']
    #     mask   = ber <= ber_target
    #     return float(length[mask][-1]) if mask.any() else float('nan')
    #
    #
    # def find_opt_bandwidth(sweep_result):
    #     """
    #     Find the receiver bandwidth [Hz] that minimizes BER.
    #     """
    #     bw  = sweep_result['bandwidth']
    #     ber = sweep_result['BER']
    #     return float(bw[np.argmin(ber)])
    #
    #
    # # ── Build summary ─────────────────────────────────────────────────────────
    # sensitivity_ook  = find_sensitivity(sweep1_ook,  BER_TARGET)
    # max_reach_ook    = find_max_reach(sweep3_ook,     BER_TARGET)
    # opt_bw_ook       = find_opt_bandwidth(sweep2_ook)
    #
    # # Replace with real values once PAM4 is implemented
    # sensitivity_pam4 = float('nan')
    # max_reach_pam4   = float('nan')
    # opt_bw_pam4      = float('nan')
    #
    # summary = pd.DataFrame({
    #     'Metric'                     : [
    #         f'Sensitivity @ BER={BER_TARGET} [dBm]',
    #         'Maximum reach [km]',
    #         'Optimum Rx bandwidth [GHz]',
    #         'Bit rate [Gb/s]',
    #         'Spectral efficiency [b/s/Hz]',
    #     ],
    #     'OOK'  : [
    #         f'{sensitivity_ook:.1f}',
    #         f'{max_reach_ook:.0f}',
    #         f'{opt_bw_ook/1e9:.2f}',
    #         f'{Rs/1e9:.0f}',
    #         '1.0',
    #     ],
    #     'PAM4' : [
    #         f'{sensitivity_pam4}',   # fill after PAM4 sweep
    #         f'{max_reach_pam4}',
    #         f'{opt_bw_pam4}',
    #         f'{2*Rs/1e9:.0f}',
    #         '2.0',
    #     ],
    # })
    #
    # summary.set_index('Metric', inplace=True)
    # print(summary)
    #
    # # Optional: export summary to CSV for inclusion in thesis
    # summary.to_csv(out_dir / 'tradeoff_summary.csv')
    # print(f"Summary saved to {out_dir / 'tradeoff_summary.csv'}")

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"Done. Total run time: {elapsed/60:.1f} min")
    print(f"All figures saved under: {out_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()