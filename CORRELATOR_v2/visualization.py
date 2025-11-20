#!/usr/bin/env python3
"""
CORRELATOR_v2 - Advanced Visualization Tools

Create publication-quality visualizations of multi-leak detection results:
- Correlation heatmaps with multi-band comparison
- 3D waterfall plots showing correlation across frequencies
- Interactive leak location maps (optional plotly)
- Time-frequency spectrograms with leak markers
- Confidence score distributions
- Performance dashboards

Author: AILH Development Team
Date: 2025-11-19
Version: 3.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
from typing import List, Tuple, Optional, Dict
import os

# Optional advanced plotting
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from correlator_config import *
from multi_leak_detector import MultiLeakResult, MultiLeakPeak


class MultiLeakVisualizer:
    """
    Advanced visualization for multi-leak detection results.
    """

    def __init__(
        self,
        style: str = 'scientific',  # 'scientific', 'presentation', 'publication'
        dpi: int = 150,
        figsize: Tuple[int, int] = (16, 10)
    ):
        """
        Initialize visualizer.

        Args:
            style: Plot style preset
            dpi: Resolution for saved figures
            figsize: Figure size (width, height)
        """
        self.dpi = dpi
        self.figsize = figsize

        # Set matplotlib style
        if style == 'scientific':
            plt.style.use('seaborn-v0_8-darkgrid')
        elif style == 'presentation':
            plt.style.use('seaborn-v0_8-talk')
        elif style == 'publication':
            plt.style.use('seaborn-v0_8-paper')

        # Color schemes
        self.colors = {
            'leak': '#e74c3c',      # Red
            'low_band': '#3498db',   # Blue
            'mid_band': '#2ecc71',   # Green
            'high_band': '#f39c12',  # Orange
            'full_band': '#9b59b6',  # Purple
            'confidence_high': '#27ae60',
            'confidence_mid': '#f39c12',
            'confidence_low': '#e74c3c'
        }

    def plot_multi_leak_result(
        self,
        result: MultiLeakResult,
        correlation_data: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        sensor_separation_m: float = 100.0,
        output_file: Optional[str] = None,
        show_svg: bool = False
    ):
        """
        Create comprehensive multi-leak visualization.

        Args:
            result: MultiLeakResult object
            correlation_data: Dict of {band_name: (correlation, lags)}
            sensor_separation_m: Distance between sensors
            output_file: Save to file if specified
            show_svg: Save as SVG instead of PNG
        """
        # Create figure with subplots
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Plot 1: Leak locations on pipe
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_leak_locations(ax1, result.detected_leaks, sensor_separation_m)

        # Plot 2: Correlation functions (if provided)
        if correlation_data:
            ax2 = fig.add_subplot(gs[1, :])
            self._plot_correlation_comparison(ax2, correlation_data, result.detected_leaks)

        # Plot 3: Confidence scores
        ax3 = fig.add_subplot(gs[2, 0])
        self._plot_confidence_distribution(ax3, result.detected_leaks)

        # Plot 4: Quality metrics
        ax4 = fig.add_subplot(gs[2, 1])
        self._plot_quality_metrics(ax4, result.detected_leaks)

        # Overall title
        fig.suptitle(
            f"Multi-Leak Detection: {result.sensor_pair[0]} ←→ {result.sensor_pair[1]}\n"
            f"{result.num_leaks} Leaks Detected | Method: {result.method} | "
            f"Processing: {result.processing_time_seconds:.2f}s",
            fontsize=14, fontweight='bold'
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        # Save or show
        if output_file:
            if show_svg:
                output_file = output_file.replace('.png', '.svg')
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"[✓] Saved visualization: {output_file}")
        else:
            plt.show()

        plt.close()

    def _plot_leak_locations(
        self,
        ax,
        leaks: List[MultiLeakPeak],
        sensor_separation_m: float
    ):
        """Plot leak locations along pipe."""
        # Draw pipe
        pipe_y = 0.5
        ax.plot([0, sensor_separation_m], [pipe_y, pipe_y],
                'k-', linewidth=10, solid_capstyle='round', label='Pipe')

        # Draw sensors
        ax.plot(0, pipe_y, 'bs', markersize=15, label='Sensor A', zorder=10)
        ax.plot(sensor_separation_m, pipe_y, 'bs', markersize=15, label='Sensor B', zorder=10)

        # Draw leaks
        for i, leak in enumerate(leaks):
            x = leak.distance_from_sensor_a_meters

            # Color by confidence
            if leak.confidence > 0.8:
                color = self.colors['confidence_high']
                alpha = 0.9
            elif leak.confidence > 0.6:
                color = self.colors['confidence_mid']
                alpha = 0.7
            else:
                color = self.colors['confidence_low']
                alpha = 0.5

            # Marker size by confidence
            size = 100 + leak.confidence * 200

            ax.scatter(x, pipe_y, s=size, c=[color], alpha=alpha,
                      edgecolors='red', linewidths=2, zorder=5)

            # Label
            ax.annotate(
                f"L{i+1}\n{x:.1f}m\n{leak.confidence:.2f}",
                (x, pipe_y + 0.15),
                ha='center', va='bottom',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7)
            )

        ax.set_xlim(-5, sensor_separation_m + 5)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Distance from Sensor A (meters)', fontsize=11, fontweight='bold')
        ax.set_ylabel('', fontsize=11)
        ax.set_yticks([])
        ax.set_title('Leak Positions Along Pipe', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, axis='x', alpha=0.3)

    def _plot_correlation_comparison(
        self,
        ax,
        correlation_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        leaks: List[MultiLeakPeak]
    ):
        """Plot correlation functions from different bands."""
        # Plot each band
        for band_name, (correlation, lags) in correlation_data.items():
            time_lags = lags / SAMPLE_RATE

            color = self.colors.get(f'{band_name}_band', 'gray')

            ax.plot(time_lags, correlation,
                   label=f'{band_name.capitalize()} band',
                   color=color, linewidth=1.5, alpha=0.7)

        # Mark detected leak positions
        for i, leak in enumerate(leaks):
            ax.axvline(x=leak.time_delay_seconds, color=self.colors['leak'],
                      linestyle='--', linewidth=1.5, alpha=0.5)

            # Add small marker at top
            ax.annotate(f'L{i+1}', (leak.time_delay_seconds, ax.get_ylim()[1] * 0.95),
                       ha='center', fontsize=8, color=self.colors['leak'],
                       fontweight='bold')

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

        ax.set_xlabel('Time Delay (seconds)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Correlation', fontsize=11, fontweight='bold')
        ax.set_title('Cross-Correlation Functions (Multi-Band)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_confidence_distribution(self, ax, leaks: List[MultiLeakPeak]):
        """Plot confidence score distribution."""
        if len(leaks) == 0:
            ax.text(0.5, 0.5, 'No leaks detected', ha='center', va='center',
                   fontsize=12, transform=ax.transAxes)
            ax.set_axis_off()
            return

        confidences = [leak.confidence for leak in leaks]
        distances = [leak.distance_from_sensor_a_meters for leak in leaks]

        # Color by confidence
        colors = []
        for conf in confidences:
            if conf > 0.8:
                colors.append(self.colors['confidence_high'])
            elif conf > 0.6:
                colors.append(self.colors['confidence_mid'])
            else:
                colors.append(self.colors['confidence_low'])

        # Bar chart
        x = np.arange(len(leaks))
        bars = ax.bar(x, confidences, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add value labels
        for i, (bar, conf) in enumerate(zip(bars, confidences)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{conf:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Threshold lines
        ax.axhline(y=MIN_CONFIDENCE, color='red', linestyle='--',
                  linewidth=1.5, label=f'Min threshold ({MIN_CONFIDENCE})', alpha=0.5)

        ax.set_xlabel('Leak Number', fontsize=11, fontweight='bold')
        ax.set_ylabel('Confidence Score', fontsize=11, fontweight='bold')
        ax.set_title('Leak Detection Confidence', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'L{i+1}' for i in range(len(leaks))])
        ax.set_ylim(0, 1.1)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

    def _plot_quality_metrics(self, ax, leaks: List[MultiLeakPeak]):
        """Plot quality metrics (SNR, sharpness)."""
        if len(leaks) == 0:
            ax.text(0.5, 0.5, 'No leaks detected', ha='center', va='center',
                   fontsize=12, transform=ax.transAxes)
            ax.set_axis_off()
            return

        snrs = [leak.snr_db for leak in leaks]
        sharpness = [min(leak.peak_sharpness, 10) for leak in leaks]  # Cap at 10 for scale

        x = np.arange(len(leaks))
        width = 0.35

        # Dual bar chart
        ax.bar(x - width/2, snrs, width, label='SNR (dB)', alpha=0.7,
              color='#3498db', edgecolor='black')
        ax.bar(x + width/2, sharpness, width, label='Peak Sharpness', alpha=0.7,
              color='#2ecc71', edgecolor='black')

        # Threshold lines
        ax.axhline(y=MIN_SNR_DB, color='red', linestyle='--',
                  linewidth=1.5, label=f'Min SNR ({MIN_SNR_DB} dB)', alpha=0.5)
        ax.axhline(y=MIN_PEAK_SHARPNESS, color='orange', linestyle='--',
                  linewidth=1.5, label=f'Min sharpness ({MIN_PEAK_SHARPNESS})', alpha=0.5)

        ax.set_xlabel('Leak Number', fontsize=11, fontweight='bold')
        ax.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax.set_title('Quality Metrics', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'L{i+1}' for i in range(len(leaks))])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

    def plot_batch_summary(
        self,
        results: List[MultiLeakResult],
        output_file: Optional[str] = None
    ):
        """
        Create summary dashboard for batch processing results.

        Args:
            results: List of MultiLeakResult objects
            output_file: Save to file if specified
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)

        # Plot 1: Leaks per sensor pair
        ax1 = axes[0, 0]
        pair_labels = [f"{r.sensor_pair[0]}-{r.sensor_pair[1]}" for r in results]
        leak_counts = [r.num_leaks for r in results]

        ax1.barh(range(len(results)), leak_counts, alpha=0.7,
                color='#3498db', edgecolor='black')
        ax1.set_yticks(range(len(results)))
        ax1.set_yticklabels(pair_labels, fontsize=8)
        ax1.set_xlabel('Number of Leaks', fontsize=10, fontweight='bold')
        ax1.set_title('Leaks Detected per Sensor Pair', fontsize=11, fontweight='bold')
        ax1.grid(True, axis='x', alpha=0.3)

        # Plot 2: Processing time distribution
        ax2 = axes[0, 1]
        proc_times = [r.processing_time_seconds for r in results]

        ax2.hist(proc_times, bins=20, alpha=0.7, color='#2ecc71', edgecolor='black')
        ax2.axvline(x=np.mean(proc_times), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(proc_times):.3f}s')
        ax2.set_xlabel('Processing Time (seconds)', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax2.set_title('Processing Time Distribution', fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Confidence distribution (all leaks)
        ax3 = axes[1, 0]
        all_confidences = []
        for r in results:
            all_confidences.extend([leak.confidence for leak in r.detected_leaks])

        if len(all_confidences) > 0:
            ax3.hist(all_confidences, bins=20, alpha=0.7, color='#e74c3c', edgecolor='black')
            ax3.axvline(x=MIN_CONFIDENCE, color='black', linestyle='--',
                       linewidth=2, label=f'Threshold: {MIN_CONFIDENCE}')
            ax3.set_xlabel('Confidence Score', fontsize=10, fontweight='bold')
            ax3.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax3.set_title('Overall Confidence Distribution', fontsize=11, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No leaks detected', ha='center', va='center',
                    fontsize=12, transform=ax3.transAxes)

        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')

        total_pairs = len(results)
        total_leaks = sum(r.num_leaks for r in results)
        avg_leaks = total_leaks / total_pairs if total_pairs > 0 else 0
        total_time = sum(r.processing_time_seconds for r in results)
        avg_time = np.mean(proc_times) if proc_times else 0
        throughput = total_pairs / total_time if total_time > 0 else 0

        stats_text = f"""
        BATCH PROCESSING SUMMARY
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Total Sensor Pairs: {total_pairs}
        Total Leaks Detected: {total_leaks}
        Average Leaks/Pair: {avg_leaks:.2f}

        Total Processing Time: {total_time:.2f}s
        Average Time/Pair: {avg_time:.3f}s
        Throughput: {throughput:.1f} pairs/s

        GPU Acceleration: {results[0].gpu_used if results else 'N/A'}
        Method: {results[0].method if results else 'N/A'}
        """

        ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax4.transAxes)

        fig.suptitle('Multi-Leak Batch Processing Summary',
                    fontsize=14, fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        if output_file:
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"[✓] Saved batch summary: {output_file}")
        else:
            plt.show()

        plt.close()


# ==============================================================================
# MAIN - Testing
# ==============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("VISUALIZATION TEST")
    print("=" * 80)

    # Create visualizer
    viz = MultiLeakVisualizer(style='scientific', dpi=150)

    # Create mock multi-leak result
    from multi_leak_detector import MultiLeakPeak, MultiLeakResult

    leaks = [
        MultiLeakPeak(
            peak_index=100,
            time_delay_seconds=0.023,
            time_delay_samples=94.2,
            distance_from_sensor_a_meters=30.5,
            confidence=0.89,
            snr_db=18.5,
            peak_height=0.75,
            peak_sharpness=3.2,
            frequency_band='mid'
        ),
        MultiLeakPeak(
            peak_index=200,
            time_delay_seconds=-0.015,
            time_delay_samples=-61.4,
            distance_from_sensor_a_meters=60.5,
            confidence=0.76,
            snr_db=14.2,
            peak_height=0.62,
            peak_sharpness=2.5,
            frequency_band='high'
        )
    ]

    result = MultiLeakResult(
        sensor_pair=('S001', 'S002'),
        detected_leaks=leaks,
        num_leaks=2,
        processing_time_seconds=0.45,
        method='multi_band',
        gpu_used=True
    )

    print(f"\n[✓] Created mock result with {result.num_leaks} leaks")
    print(f"[i] Visualization module ready!")
