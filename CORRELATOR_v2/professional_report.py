#!/usr/bin/env python3
"""
CORRELATOR_v2 - Professional Engineering Report Generator

Generate comprehensive PDF reports with:
- Executive summary
- Detailed leak locations with maps
- All visualizations and plots
- Engineering data tables
- QA/validation metrics
- Sensor configuration details
- Processing statistics

Output format: Professional PDF suitable for field crews and engineering review.

Dependencies: matplotlib, reportlab (or matplotlib PDF backend)

Author: AILH Development Team
Date: 2025-11-19
Version: 3.1.0
"""

import os
import sys
from datetime import datetime
from typing import List, Optional, Dict
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.table import Table

from correlator_config import *
from multi_leak_detector import MultiLeakResult, MultiLeakPeak
from multi_sensor_triangulation import TriangulatedLeak
from sensor_registry import SensorRegistry
from visualization import MultiLeakVisualizer


class ProfessionalReportGenerator:
    """
    Generate professional PDF engineering reports for leak detection results.

    Report sections:
    1. Cover page with project info
    2. Executive summary
    3. Leak location maps
    4. Detailed leak data tables
    5. Quality assurance metrics
    6. Sensor configuration
    7. Processing statistics
    8. Appendices (raw data, validation)
    """

    def __init__(
        self,
        project_name: str = "Leak Detection Survey",
        site_name: str = "Unknown Site",
        report_author: str = "AILH Correlator v3.0",
        company_name: str = "Water Utility",
        logo_path: Optional[str] = None
    ):
        """
        Initialize report generator.

        Args:
            project_name: Project/survey name
            site_name: Site location name
            report_author: Author/operator name
            company_name: Company/organization name
            logo_path: Path to company logo (optional)
        """
        self.project_name = project_name
        self.site_name = site_name
        self.report_author = report_author
        self.company_name = company_name
        self.logo_path = logo_path

        self.report_date = datetime.now()

    def generate_report(
        self,
        results: List[MultiLeakResult],
        registry: SensorRegistry,
        triangulated_leaks: Optional[List[TriangulatedLeak]] = None,
        output_file: str = "leak_detection_report.pdf",
        include_raw_data: bool = True,
        include_validation: bool = True
    ):
        """
        Generate complete PDF report.

        Args:
            results: List of MultiLeakResult objects
            registry: Sensor registry
            triangulated_leaks: Optional triangulated results from multi-sensor
            output_file: Output PDF filename
            include_raw_data: Include raw correlation data
            include_validation: Include validation metrics
        """
        print(f"\n[i] Generating professional engineering report...")
        print(f"    Output: {output_file}")

        with PdfPages(output_file) as pdf:
            # Page 1: Cover page
            self._add_cover_page(pdf)

            # Page 2: Executive summary
            self._add_executive_summary(pdf, results, triangulated_leaks)

            # Page 3-N: Leak location maps and details
            self._add_leak_details(pdf, results, registry, triangulated_leaks)

            # Data tables
            self._add_data_tables(pdf, results, triangulated_leaks)

            # Quality assurance
            if include_validation:
                self._add_qa_section(pdf, results)

            # Sensor configuration
            self._add_sensor_config(pdf, registry)

            # Processing statistics
            self._add_processing_stats(pdf, results)

            # Appendices
            if include_raw_data:
                self._add_appendices(pdf, results)

            # Metadata
            d = pdf.infodict()
            d['Title'] = f'{self.project_name} - Leak Detection Report'
            d['Author'] = self.report_author
            d['Subject'] = 'Acoustic Leak Detection and Localization'
            d['Keywords'] = 'leak detection, acoustic correlation, water utility'
            d['CreationDate'] = self.report_date

        print(f"[✓] Report generated: {output_file}")
        print(f"    Total pages: {pdf.get_pagecount()}")

    def _add_cover_page(self, pdf: PdfPages):
        """Add professional cover page."""
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')

        # Company logo (if provided)
        if self.logo_path and os.path.exists(self.logo_path):
            # Load and display logo
            pass  # TODO: Implement logo loading

        # Title
        title_y = 0.85
        ax.text(0.5, title_y, self.project_name,
               ha='center', va='center', fontsize=28, fontweight='bold',
               transform=ax.transAxes)

        # Subtitle
        ax.text(0.5, title_y - 0.08, 'Acoustic Leak Detection Report',
               ha='center', va='center', fontsize=18,
               transform=ax.transAxes, style='italic')

        # Site info box
        info_y = 0.6
        info_text = f"""
        Site: {self.site_name}
        Company: {self.company_name}

        Report Date: {self.report_date.strftime('%B %d, %Y')}
        Report Time: {self.report_date.strftime('%H:%M:%S')}

        Generated by: {self.report_author}
        """

        ax.text(0.5, info_y, info_text,
               ha='center', va='center', fontsize=12,
               transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.3))

        # Footer
        ax.text(0.5, 0.05,
               'CONFIDENTIAL - For Authorized Personnel Only',
               ha='center', va='center', fontsize=10,
               transform=ax.transAxes, style='italic', color='red')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _add_executive_summary(
        self,
        pdf: PdfPages,
        results: List[MultiLeakResult],
        triangulated_leaks: Optional[List[TriangulatedLeak]]
    ):
        """Add executive summary page."""
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')

        # Title
        ax.text(0.5, 0.95, 'Executive Summary',
               ha='center', va='top', fontsize=20, fontweight='bold',
               transform=ax.transAxes)

        # Statistics
        total_pairs = len(results)
        total_leaks = sum(r.num_leaks for r in results)
        avg_leaks = total_leaks / total_pairs if total_pairs > 0 else 0
        total_time = sum(r.processing_time_seconds for r in results)

        # High confidence leaks (>0.8)
        high_conf_leaks = sum(
            sum(1 for leak in r.detected_leaks if leak.confidence > 0.8)
            for r in results
        )

        summary_text = f"""
        SURVEY SUMMARY
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Sensor Pairs Analyzed: {total_pairs}
        Total Leaks Detected: {total_leaks}
        Average Leaks per Pair: {avg_leaks:.2f}

        High Confidence Leaks (>0.8): {high_conf_leaks}
        Medium Confidence (0.6-0.8): {total_leaks - high_conf_leaks}

        Total Processing Time: {total_time:.1f}s
        Average Time per Pair: {total_time/total_pairs:.2f}s
        """

        if triangulated_leaks:
            summary_text += f"""
        MULTI-SENSOR TRIANGULATION
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Triangulated Leak Locations: {len(triangulated_leaks)}
        Average Uncertainty: ±{np.mean([l.uncertainty_meters for l in triangulated_leaks]):.1f}m
        """

        ax.text(0.1, 0.85, summary_text,
               ha='left', va='top', fontsize=11, family='monospace',
               transform=ax.transAxes)

        # Key findings
        findings_y = 0.4

        ax.text(0.1, findings_y, 'KEY FINDINGS',
               ha='left', va='top', fontsize=14, fontweight='bold',
               transform=ax.transAxes)

        findings_text = self._generate_key_findings(results, triangulated_leaks)

        ax.text(0.1, findings_y - 0.05, findings_text,
               ha='left', va='top', fontsize=11,
               transform=ax.transAxes)

        # Recommendations
        rec_y = 0.15

        ax.text(0.1, rec_y, 'RECOMMENDATIONS',
               ha='left', va='top', fontsize=14, fontweight='bold',
               transform=ax.transAxes)

        recommendations = self._generate_recommendations(results)

        ax.text(0.1, rec_y - 0.05, recommendations,
               ha='left', va='top', fontsize=11,
               transform=ax.transAxes)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _generate_key_findings(
        self,
        results: List[MultiLeakResult],
        triangulated_leaks: Optional[List[TriangulatedLeak]]
    ) -> str:
        """Generate key findings text."""
        findings = []

        # Check for multiple leaks
        multi_leak_pairs = [r for r in results if r.num_leaks > 1]
        if len(multi_leak_pairs) > 0:
            findings.append(
                f"• {len(multi_leak_pairs)} sensor pairs detected multiple leaks\n"
                f"  (potential complex leak scenario)"
            )

        # Check for high confidence detections
        high_conf = sum(
            sum(1 for leak in r.detected_leaks if leak.confidence > 0.9)
            for r in results
        )

        if high_conf > 0:
            findings.append(f"• {high_conf} leaks with very high confidence (>0.9)")

        # Check for triangulated results
        if triangulated_leaks:
            precise_leaks = [l for l in triangulated_leaks if l.uncertainty_meters < 10]
            if len(precise_leaks) > 0:
                findings.append(
                    f"• {len(precise_leaks)} leak locations with high precision\n"
                    f"  (<10m uncertainty from multi-sensor triangulation)"
                )

        if len(findings) == 0:
            findings.append("• All detections within normal parameters")

        return "\n".join(findings)

    def _generate_recommendations(self, results: List[MultiLeakResult]) -> str:
        """Generate recommendations text."""
        recommendations = []

        # Priority leaks (high confidence)
        priority_leaks = []
        for r in results:
            for leak in r.detected_leaks:
                if leak.confidence > 0.85:
                    priority_leaks.append((r.sensor_pair, leak))

        if len(priority_leaks) > 0:
            recommendations.append(
                f"• PRIORITY: Investigate {len(priority_leaks)} high-confidence\n"
                f"  leak locations immediately"
            )

        # Low SNR warnings
        low_snr_count = sum(
            sum(1 for leak in r.detected_leaks if leak.snr_db < 10)
            for r in results
        )

        if low_snr_count > 0:
            recommendations.append(
                f"• {low_snr_count} detections with low SNR (<10 dB)\n"
                f"  - Recommend field verification\n"
                f"  - Consider signal stacking if available"
            )

        # Multi-leak scenarios
        multi_leak_pairs = [r for r in results if r.num_leaks > 2]
        if len(multi_leak_pairs) > 0:
            recommendations.append(
                f"• {len(multi_leak_pairs)} locations with 3+ leaks\n"
                f"  - Priority repair area\n"
                f"  - Possible pipe section replacement needed"
            )

        if len(recommendations) == 0:
            recommendations.append("• Continue normal monitoring schedule")

        return "\n".join(recommendations)

    def _add_leak_details(
        self,
        pdf: PdfPages,
        results: List[MultiLeakResult],
        registry: SensorRegistry,
        triangulated_leaks: Optional[List[TriangulatedLeak]]
    ):
        """Add detailed leak location pages with maps."""
        viz = MultiLeakVisualizer(style='scientific', dpi=150)

        # Group results by sensor pair
        for i, result in enumerate(results):
            if result.num_leaks == 0:
                continue

            # Create comprehensive figure for this sensor pair
            fig = plt.figure(figsize=(8.5, 11))
            gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)

            # Title
            fig.suptitle(
                f"Leak Detection Report - Pair {i+1}/{len(results)}\n"
                f"Sensors: {result.sensor_pair[0]} ←→ {result.sensor_pair[1]}",
                fontsize=14, fontweight='bold'
            )

            # Get sensor pair config
            pair_config = registry.get_sensor_pair(
                result.sensor_pair[0], result.sensor_pair[1]
            )

            if pair_config:
                # Leak location map
                ax1 = fig.add_subplot(gs[0, :])
                viz._plot_leak_locations(
                    ax1, result.detected_leaks, pair_config.distance_meters
                )

                # Data table
                ax2 = fig.add_subplot(gs[1, :])
                self._add_leak_table(ax2, result.detected_leaks)

                # Confidence plot
                ax3 = fig.add_subplot(gs[2, 0])
                viz._plot_confidence_distribution(ax3, result.detected_leaks)

                # Quality metrics
                ax4 = fig.add_subplot(gs[2, 1])
                viz._plot_quality_metrics(ax4, result.detected_leaks)

                # Sensor info
                ax5 = fig.add_subplot(gs[3, :])
                self._add_sensor_info_box(ax5, result.sensor_pair, pair_config, registry)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    def _add_leak_table(self, ax, leaks: List[MultiLeakPeak]):
        """Add leak data table."""
        ax.axis('tight')
        ax.axis('off')

        # Table data
        headers = ['Leak', 'Distance (m)', 'Delay (s)', 'Confidence', 'SNR (dB)', 'Band']
        rows = []

        for i, leak in enumerate(leaks):
            rows.append([
                f'L{i+1}',
                f'{leak.distance_from_sensor_a_meters:.1f}',
                f'{leak.time_delay_seconds:.6f}',
                f'{leak.confidence:.3f}',
                f'{leak.snr_db:.1f}',
                leak.frequency_band or 'N/A'
            ])

        table = ax.table(cellText=rows, colLabels=headers,
                        cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Color code by confidence
        for i, leak in enumerate(leaks):
            if leak.confidence > 0.8:
                color = '#d4edda'  # Green
            elif leak.confidence > 0.6:
                color = '#fff3cd'  # Yellow
            else:
                color = '#f8d7da'  # Red

            for j in range(len(headers)):
                table[(i+1, j)].set_facecolor(color)

        # Header style
        for j in range(len(headers)):
            table[(0, j)].set_facecolor('#007bff')
            table[(0, j)].set_text_props(weight='bold', color='white')

        ax.set_title('Detected Leaks - Detailed Data', fontsize=12, fontweight='bold', pad=20)

    def _add_sensor_info_box(
        self,
        ax,
        sensor_pair: Tuple[str, str],
        pair_config,
        registry: SensorRegistry
    ):
        """Add sensor configuration info box."""
        ax.axis('off')

        sensor_a_info = registry.get_sensor(sensor_pair[0])
        sensor_b_info = registry.get_sensor(sensor_pair[1])

        info_text = f"""
        SENSOR CONFIGURATION
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Sensor A: {sensor_pair[0]} ({sensor_a_info.name if sensor_a_info else 'Unknown'})
          Position: ({sensor_a_info.position.latitude:.6f}, {sensor_a_info.position.longitude:.6f})
          Logger ID: {sensor_a_info.logger_id if sensor_a_info else 'N/A'}

        Sensor B: {sensor_pair[1]} ({sensor_b_info.name if sensor_b_info else 'Unknown'})
          Position: ({sensor_b_info.position.latitude:.6f}, {sensor_b_info.position.longitude:.6f})
          Logger ID: {sensor_b_info.logger_id if sensor_b_info else 'N/A'}

        Pipe Configuration:
          Separation: {pair_config.distance_meters:.1f} meters
          Material: {pair_config.pipe_segment.material}
          Diameter: {pair_config.pipe_segment.diameter_mm} mm
          Wave Speed: {pair_config.wave_speed_mps} m/s
        """

        ax.text(0.1, 0.5, info_text,
               ha='left', va='center', fontsize=9, family='monospace',
               transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))

    def _add_data_tables(
        self,
        pdf: PdfPages,
        results: List[MultiLeakResult],
        triangulated_leaks: Optional[List[TriangulatedLeak]]
    ):
        """Add comprehensive data tables."""
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')

        ax.text(0.5, 0.98, 'Leak Detection Data Summary',
               ha='center', va='top', fontsize=16, fontweight='bold',
               transform=ax.transAxes)

        # Summary table for all leaks
        y_pos = 0.90

        all_leaks = []
        for r in results:
            for leak in r.detected_leaks:
                all_leaks.append({
                    'pair': f"{r.sensor_pair[0]}-{r.sensor_pair[1]}",
                    'distance': leak.distance_from_sensor_a_meters,
                    'confidence': leak.confidence,
                    'snr': leak.snr_db,
                    'band': leak.frequency_band or 'N/A'
                })

        # Sort by confidence (descending)
        all_leaks_sorted = sorted(all_leaks, key=lambda x: x['confidence'], reverse=True)

        # Create table (show top 30, or all if fewer)
        headers = ['#', 'Sensor Pair', 'Dist (m)', 'Confidence', 'SNR (dB)', 'Band']

        # Split into pages if needed (15 rows per page)
        rows_per_page = 15
        n_pages = (len(all_leaks_sorted) + rows_per_page - 1) // rows_per_page

        for page in range(min(n_pages, 3)):  # Max 3 pages of tables
            start_idx = page * rows_per_page
            end_idx = min(start_idx + rows_per_page, len(all_leaks_sorted))

            page_leaks = all_leaks_sorted[start_idx:end_idx]

            table_data = []
            for i, leak in enumerate(page_leaks):
                table_data.append([
                    f'{start_idx + i + 1}',
                    leak['pair'],
                    f"{leak['distance']:.1f}",
                    f"{leak['confidence']:.3f}",
                    f"{leak['snr']:.1f}",
                    leak['band']
                ])

            # Create subplot area for table
            table_ax = plt.subplot2grid((20, 1), (2 + page * 6, 0), rowspan=5, colspan=1)
            table_ax.axis('tight')
            table_ax.axis('off')

            table = table_ax.table(cellText=table_data, colLabels=headers,
                                  cellLoc='center', loc='center',
                                  bbox=[0, 0, 1, 1])

            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)

            # Color code by confidence
            for i, leak in enumerate(page_leaks):
                if leak['confidence'] > 0.8:
                    color = '#d4edda'  # Green
                elif leak['confidence'] > 0.6:
                    color = '#fff3cd'  # Yellow
                else:
                    color = '#f8d7da'  # Red

                for j in range(len(headers)):
                    table[(i+1, j)].set_facecolor(color)

            # Header style
            for j in range(len(headers)):
                table[(0, j)].set_facecolor('#007bff')
                table[(0, j)].set_text_props(weight='bold', color='white')

        # Add summary statistics
        stats_y = 0.05
        stats_text = f"Total Leaks: {len(all_leaks)} | High Confidence (>0.8): {sum(1 for l in all_leaks if l['confidence'] > 0.8)} | Showing top {min(len(all_leaks), rows_per_page * 3)}"
        ax.text(0.5, stats_y, stats_text,
               ha='center', va='bottom', fontsize=9, style='italic',
               transform=ax.transAxes)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _add_qa_section(self, pdf: PdfPages, results: List[MultiLeakResult]):
        """Add quality assurance and validation section."""
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')

        ax.text(0.5, 0.95, 'Quality Assurance & Validation',
               ha='center', va='top', fontsize=16, fontweight='bold',
               transform=ax.transAxes)

        # QA metrics
        qa_text = self._generate_qa_metrics(results)

        ax.text(0.1, 0.85, qa_text,
               ha='left', va='top', fontsize=10, family='monospace',
               transform=ax.transAxes)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _generate_qa_metrics(self, results: List[MultiLeakResult]) -> str:
        """Generate QA metrics text."""
        all_confidences = []
        all_snrs = []

        for r in results:
            for leak in r.detected_leaks:
                all_confidences.append(leak.confidence)
                all_snrs.append(leak.snr_db)

        if len(all_confidences) == 0:
            return "No leaks detected for QA analysis"

        qa_text = f"""
        QUALITY METRICS
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Confidence Scores:
          Mean: {np.mean(all_confidences):.3f}
          Std Dev: {np.std(all_confidences):.3f}
          Min: {np.min(all_confidences):.3f}
          Max: {np.max(all_confidences):.3f}

        SNR (Signal-to-Noise Ratio):
          Mean: {np.mean(all_snrs):.1f} dB
          Std Dev: {np.std(all_snrs):.1f} dB
          Min: {np.min(all_snrs):.1f} dB
          Max: {np.max(all_snrs):.1f} dB

        Quality Thresholds:
          Minimum Confidence: {MIN_CONFIDENCE}
          Minimum SNR: {MIN_SNR_DB} dB
          Minimum Sharpness: {MIN_PEAK_SHARPNESS}

        Pass/Fail:
          Above confidence threshold: {sum(1 for c in all_confidences if c >= MIN_CONFIDENCE)} / {len(all_confidences)}
          Above SNR threshold: {sum(1 for s in all_snrs if s >= MIN_SNR_DB)} / {len(all_snrs)}
        """

        return qa_text

    def _add_sensor_config(self, pdf: PdfPages, registry: SensorRegistry):
        """Add sensor configuration page."""
        fig = plt.figure(figsize=(8.5, 11))
        gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.3, height_ratios=[1, 2, 2])

        # Title
        fig.suptitle('Sensor Network Configuration',
                    fontsize=16, fontweight='bold')

        # Sensor map (top)
        ax_map = fig.add_subplot(gs[0])
        self._plot_sensor_network_map(ax_map, registry)

        # Sensor pairs table (middle)
        ax_pairs = fig.add_subplot(gs[1])
        self._plot_sensor_pairs_table(ax_pairs, registry)

        # Sensor details table (bottom)
        ax_sensors = fig.add_subplot(gs[2])
        self._plot_sensors_table(ax_sensors, registry)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _plot_sensor_network_map(self, ax, registry: SensorRegistry):
        """Plot sensor network map with positions."""
        ax.set_title('Sensor Network Map', fontsize=12, fontweight='bold')

        # Plot sensor positions
        for sensor in registry.sensors:
            ax.plot(sensor.position.longitude, sensor.position.latitude,
                   'ro', markersize=10, label=sensor.sensor_id if len(registry.sensors) < 10 else None)
            ax.text(sensor.position.longitude, sensor.position.latitude,
                   f'  {sensor.sensor_id}', fontsize=8, va='center')

        # Plot connections
        for pair in registry.sensor_pairs:
            sensor_a = registry.get_sensor(pair.sensor_a)
            sensor_b = registry.get_sensor(pair.sensor_b)

            if sensor_a and sensor_b:
                ax.plot([sensor_a.position.longitude, sensor_b.position.longitude],
                       [sensor_a.position.latitude, sensor_b.position.latitude],
                       'b-', alpha=0.5, linewidth=1)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    def _plot_sensor_pairs_table(self, ax, registry: SensorRegistry):
        """Plot sensor pairs configuration table."""
        ax.axis('tight')
        ax.axis('off')
        ax.set_title('Sensor Pair Configuration', fontsize=12, fontweight='bold', pad=15)

        headers = ['Pair', 'Sensor A', 'Sensor B', 'Distance (m)', 'Material', 'Wave Speed (m/s)']
        rows = []

        for i, pair in enumerate(registry.sensor_pairs):
            rows.append([
                f'P{i+1}',
                pair.sensor_a,
                pair.sensor_b,
                f'{pair.distance_meters:.1f}',
                pair.pipe_segment.material,
                f'{pair.wave_speed_mps}'
            ])

        table = ax.table(cellText=rows, colLabels=headers,
                        cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Header style
        for j in range(len(headers)):
            table[(0, j)].set_facecolor('#007bff')
            table[(0, j)].set_text_props(weight='bold', color='white')

    def _plot_sensors_table(self, ax, registry: SensorRegistry):
        """Plot sensors details table."""
        ax.axis('tight')
        ax.axis('off')
        ax.set_title('Sensor Details', fontsize=12, fontweight='bold', pad=15)

        headers = ['ID', 'Name', 'Latitude', 'Longitude', 'Logger ID', 'Gain (dB)']
        rows = []

        for sensor in registry.sensors:
            rows.append([
                sensor.sensor_id,
                sensor.name or 'N/A',
                f'{sensor.position.latitude:.6f}',
                f'{sensor.position.longitude:.6f}',
                sensor.logger_id or 'N/A',
                f'{sensor.gain_db}' if sensor.gain_db else 'N/A'
            ])

        table = ax.table(cellText=rows, colLabels=headers,
                        cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Header style
        for j in range(len(headers)):
            table[(0, j)].set_facecolor('#007bff')
            table[(0, j)].set_text_props(weight='bold', color='white')

    def _add_processing_stats(self, pdf: PdfPages, results: List[MultiLeakResult]):
        """Add processing statistics page."""
        fig = plt.figure(figsize=(8.5, 11))
        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)

        fig.suptitle('Processing Statistics & Performance',
                    fontsize=16, fontweight='bold')

        # Calculate statistics
        total_pairs = len(results)
        total_leaks = sum(r.num_leaks for r in results)
        processing_times = [r.processing_time_seconds for r in results]
        gpu_used = any(r.gpu_used for r in results)

        # Processing time histogram
        ax1 = fig.add_subplot(gs[0, :])
        ax1.hist(processing_times, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Processing Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Processing Time Distribution')
        ax1.axvline(np.mean(processing_times), color='red', linestyle='--',
                   label=f'Mean: {np.mean(processing_times):.3f}s')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Leaks per pair histogram
        ax2 = fig.add_subplot(gs[1, 0])
        leaks_per_pair = [r.num_leaks for r in results]
        ax2.hist(leaks_per_pair, bins=range(0, max(leaks_per_pair) + 2),
                color='orange', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Number of Leaks')
        ax2.set_ylabel('Number of Pairs')
        ax2.set_title('Leaks per Sensor Pair')
        ax2.grid(True, alpha=0.3)

        # Method distribution
        ax3 = fig.add_subplot(gs[1, 1])
        methods = [r.method for r in results]
        method_counts = {}
        for m in methods:
            method_counts[m] = method_counts.get(m, 0) + 1

        ax3.bar(method_counts.keys(), method_counts.values(),
               color='green', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Correlation Method')
        ax3.set_ylabel('Count')
        ax3.set_title('Correlation Methods Used')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')

        # Statistics text box
        ax4 = fig.add_subplot(gs[2:, :])
        ax4.axis('off')

        stats_text = f"""
        PROCESSING SUMMARY
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Total Sensor Pairs Processed: {total_pairs}
        Total Leaks Detected: {total_leaks}
        Average Leaks per Pair: {total_leaks/total_pairs:.2f}

        Processing Time Statistics:
          Total Time: {sum(processing_times):.2f} seconds
          Average Time per Pair: {np.mean(processing_times):.3f} seconds
          Std Dev: {np.std(processing_times):.3f} seconds
          Min Time: {min(processing_times):.3f} seconds
          Max Time: {max(processing_times):.3f} seconds
          Throughput: {total_pairs/sum(processing_times):.1f} pairs/second

        Hardware Configuration:
          GPU Acceleration: {'Enabled' if gpu_used else 'Disabled'}
          Precision: {results[0].quality_metrics.get('precision', 'N/A') if results else 'N/A'}
          CUDA Streams: {results[0].quality_metrics.get('n_cuda_streams', 'N/A') if results else 'N/A'}

        Quality Metrics (Average):
          Confidence: {np.mean([np.mean([l.confidence for l in r.detected_leaks]) for r in results if r.num_leaks > 0]):.3f}
          SNR: {np.mean([np.mean([l.snr_db for l in r.detected_leaks]) for r in results if r.num_leaks > 0]):.1f} dB
          Peak Sharpness: {np.mean([np.mean([l.peak_sharpness for l in r.detected_leaks]) for r in results if r.num_leaks > 0]):.2f}

        Processing Date: {self.report_date.strftime('%Y-%m-%d %H:%M:%S')}
        Software Version: CORRELATOR_v3.0
        """

        ax4.text(0.1, 0.9, stats_text,
                ha='left', va='top', fontsize=10, family='monospace',
                transform=ax4.transAxes,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.3))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _add_appendices(self, pdf: PdfPages, results: List[MultiLeakResult]):
        """Add appendices with raw data."""
        # Appendix A: Raw Detection Data
        self._add_appendix_raw_data(pdf, results)

        # Appendix B: Frequency Band Analysis
        self._add_appendix_frequency_analysis(pdf, results)

    def _add_appendix_raw_data(self, pdf: PdfPages, results: List[MultiLeakResult]):
        """Appendix A: Raw detection data."""
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')

        ax.text(0.5, 0.98, 'Appendix A: Raw Detection Data',
               ha='center', va='top', fontsize=16, fontweight='bold',
               transform=ax.transAxes)

        # Create detailed data listing
        y_pos = 0.92
        line_height = 0.03

        for i, result in enumerate(results[:20]):  # Limit to first 20 pairs
            if result.num_leaks == 0:
                continue

            pair_text = f"\nSensor Pair {i+1}: {result.sensor_pair[0]} ←→ {result.sensor_pair[1]}"
            ax.text(0.05, y_pos, pair_text,
                   ha='left', va='top', fontsize=10, fontweight='bold',
                   transform=ax.transAxes)
            y_pos -= line_height

            for j, leak in enumerate(result.detected_leaks):
                leak_text = f"  Leak {j+1}: {leak.distance_from_sensor_a_meters:.2f}m | " \
                           f"τ={leak.time_delay_seconds:.6f}s | " \
                           f"Conf={leak.confidence:.3f} | " \
                           f"SNR={leak.snr_db:.1f}dB | " \
                           f"Band={leak.frequency_band or 'N/A'}"

                ax.text(0.05, y_pos, leak_text,
                       ha='left', va='top', fontsize=8, family='monospace',
                       transform=ax.transAxes)
                y_pos -= line_height * 0.8

                if y_pos < 0.05:
                    break

            if y_pos < 0.05:
                break

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _add_appendix_frequency_analysis(self, pdf: PdfPages, results: List[MultiLeakResult]):
        """Appendix B: Frequency band analysis."""
        fig = plt.figure(figsize=(8.5, 11))
        gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.3)

        fig.suptitle('Appendix B: Frequency Band Analysis',
                    fontsize=16, fontweight='bold')

        # Collect frequency band statistics
        band_counts = {}
        band_confidences = {}

        for result in results:
            for leak in result.detected_leaks:
                band = leak.frequency_band or 'unknown'
                band_counts[band] = band_counts.get(band, 0) + 1

                if band not in band_confidences:
                    band_confidences[band] = []
                band_confidences[band].append(leak.confidence)

        # Band distribution
        ax1 = fig.add_subplot(gs[0])
        if band_counts:
            bands = list(band_counts.keys())
            counts = list(band_counts.values())

            ax1.bar(bands, counts, color='steelblue', alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Frequency Band')
            ax1.set_ylabel('Number of Detections')
            ax1.set_title('Detection Count by Frequency Band')
            ax1.grid(True, alpha=0.3, axis='y')

        # Band confidence comparison
        ax2 = fig.add_subplot(gs[1])
        if band_confidences:
            bands = list(band_confidences.keys())
            conf_data = [band_confidences[b] for b in bands]

            bp = ax2.boxplot(conf_data, labels=bands, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightgreen')

            ax2.set_xlabel('Frequency Band')
            ax2.set_ylabel('Confidence Score')
            ax2.set_title('Confidence Distribution by Frequency Band')
            ax2.grid(True, alpha=0.3, axis='y')

        # Summary statistics table
        ax3 = fig.add_subplot(gs[2])
        ax3.axis('tight')
        ax3.axis('off')

        if band_confidences:
            headers = ['Band', 'Count', 'Mean Conf', 'Std Dev', 'Min', 'Max']
            rows = []

            for band in sorted(band_counts.keys()):
                confs = band_confidences[band]
                rows.append([
                    band,
                    f'{band_counts[band]}',
                    f'{np.mean(confs):.3f}',
                    f'{np.std(confs):.3f}',
                    f'{np.min(confs):.3f}',
                    f'{np.max(confs):.3f}'
                ])

            table = ax3.table(cellText=rows, colLabels=headers,
                            cellLoc='center', loc='center',
                            bbox=[0, 0, 1, 1])

            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)

            # Header style
            for j in range(len(headers)):
                table[(0, j)].set_facecolor('#007bff')
                table[(0, j)].set_text_props(weight='bold', color='white')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()


# ==============================================================================
# MAIN - Testing
# ==============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("PROFESSIONAL REPORT GENERATOR TEST")
    print("=" * 80)

    generator = ProfessionalReportGenerator(
        project_name="Test Leak Survey 2025",
        site_name="Downtown Water District",
        report_author="Test Engineer",
        company_name="Test Water Utility"
    )

    print("\n[✓] Report generator initialized")
    print(f"[i] Ready to generate professional PDF reports!")
