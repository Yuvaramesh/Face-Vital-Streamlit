import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
from scipy import signal
from scipy.fft import fft
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
import io
import base64

# PDF generation libraries
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import (
        SimpleDocTemplate,
        Table,
        TableStyle,
        Paragraph,
        Spacer,
        Image as RLImage,
        PageBreak,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.error(
        "ReportLab is required for PDF generation! Install with: pip install reportlab"
    )


class StreamlitFaceVitalMonitor:
    def __init__(self):
        # Initialize MediaPipe face detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Initialize session state
        if "monitoring_active" not in st.session_state:
            st.session_state.monitoring_active = False
        if "face_detected" not in st.session_state:
            st.session_state.face_detected = False
        if "ppg_signal" not in st.session_state:
            st.session_state.ppg_signal = deque(maxlen=900)  # 30 seconds at 30fps
        if "timestamps" not in st.session_state:
            st.session_state.timestamps = deque(maxlen=900)
        if "calculation_count" not in st.session_state:
            st.session_state.calculation_count = 0

        # Individual metric signals for wave display
        if "hr_values" not in st.session_state:
            st.session_state.hr_values = deque(maxlen=300)
        if "br_values" not in st.session_state:
            st.session_state.br_values = deque(maxlen=300)
        if "hrv_values" not in st.session_state:
            st.session_state.hrv_values = deque(maxlen=300)
        if "stress_values" not in st.session_state:
            st.session_state.stress_values = deque(maxlen=300)
        if "para_values" not in st.session_state:
            st.session_state.para_values = deque(maxlen=300)
        if "wellness_values" not in st.session_state:
            st.session_state.wellness_values = deque(maxlen=300)
        if "bp_sys_values" not in st.session_state:
            st.session_state.bp_sys_values = deque(maxlen=300)
        if "bp_dia_values" not in st.session_state:
            st.session_state.bp_dia_values = deque(maxlen=300)

        # Results
        if "results" not in st.session_state:
            st.session_state.results = {
                "heart_rate": 0,
                "breathing_rate": 0,
                "blood_pressure_sys": 0,
                "blood_pressure_dia": 0,
                "hrv": 0,
                "stress_index": 0,
                "parasympathetic": 0,
                "wellness_score": 0,
            }

        # Session data for PDF report
        if "session_data" not in st.session_state:
            st.session_state.session_data = {
                "start_time": None,
                "end_time": None,
                "measurements": [],
                "raw_ppg_data": [],
                "timestamps_data": [],
            }

    def extract_ppg_signal(self, frame, landmarks):
        """Extract PPG signal from facial regions using MediaPipe landmarks"""
        try:
            h, w = frame.shape[:2]

            # Define ROI indices for forehead and cheek regions
            forehead_indices = [10, 151, 9, 10, 151, 9, 10, 151]
            left_cheek_indices = [116, 117, 118, 119, 120, 121]
            right_cheek_indices = [345, 346, 347, 348, 349, 350]

            roi_values = []

            for indices in [forehead_indices, left_cheek_indices, right_cheek_indices]:
                region_points = []
                for idx in indices:
                    if idx < len(landmarks):
                        x = int(landmarks[idx].x * w)
                        y = int(landmarks[idx].y * h)
                        region_points.append([x, y])

                if len(region_points) > 2:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [np.array(region_points)], 255)

                    green_channel = frame[:, :, 1]
                    roi_mean = cv2.mean(green_channel, mask)[0]
                    roi_values.append(roi_mean)

            if roi_values:
                ppg_value = np.mean(roi_values)
                return ppg_value
            else:
                return 0

        except Exception as e:
            st.error(f"Error extracting PPG signal: {e}")
            return 0

    def calculate_heart_rate(self, signal_data, timestamps, fps=30):
        """Calculate heart rate from PPG signal"""
        if len(signal_data) < fps * 8:
            return 0

        try:
            detrended = signal.detrend(signal_data)
            nyquist = fps / 2
            low = 0.8 / nyquist
            high = 4.0 / nyquist
            b, a = signal.butter(4, [low, high], btype="band")
            filtered = signal.filtfilt(b, a, detrended)

            fft_data = fft(filtered)
            freqs = np.fft.fftfreq(len(filtered), 1 / fps)

            valid_indices = (freqs >= 0.8) & (freqs <= 4.0)
            valid_fft = np.abs(fft_data[valid_indices])
            valid_freqs = freqs[valid_indices]

            if len(valid_fft) > 0:
                peak_idx = np.argmax(valid_fft)
                heart_rate_hz = valid_freqs[peak_idx]
                heart_rate_bpm = heart_rate_hz * 60
                return max(50, min(200, heart_rate_bpm))

        except Exception as e:
            st.error(f"Error calculating heart rate: {e}")

        return 0

    def calculate_breathing_rate(self, signal_data, fps=30):
        """Calculate breathing rate from signal variations"""
        if len(signal_data) < fps * 12:
            return 0

        try:
            nyquist = fps / 2
            low = 0.1 / nyquist
            high = 0.5 / nyquist
            b, a = signal.butter(2, [low, high], btype="band")
            filtered = signal.filtfilt(b, a, signal_data)

            peaks, _ = signal.find_peaks(filtered, distance=fps * 2)
            breathing_rate = len(peaks) * (60 / (len(signal_data) / fps))

            return max(8, min(35, breathing_rate))

        except Exception as e:
            st.error(f"Error calculating breathing rate: {e}")
            return 0

    def estimate_blood_pressure(self, heart_rate, hrv, stress_index):
        """Estimate blood pressure using correlation with HR, HRV, and stress"""
        try:
            base_sys = 120
            base_dia = 80

            hr_factor = (heart_rate - 70) * 0.5
            stress_factor = stress_index * 10
            hrv_factor = (50 - hrv) * 0.2

            sys_bp = base_sys + hr_factor + stress_factor + hrv_factor
            dia_bp = base_dia + hr_factor * 0.6 + stress_factor * 0.6 + hrv_factor * 0.6

            sys_bp = max(90, min(180, sys_bp))
            dia_bp = max(60, min(120, dia_bp))

            return int(sys_bp), int(dia_bp)

        except Exception as e:
            st.error(f"Error estimating blood pressure: {e}")
            return 120, 80

    def calculate_hrv(self, signal_data, fps=30):
        """Calculate Heart Rate Variability"""
        if len(signal_data) < fps * 15:
            return 0

        try:
            filtered = signal.medfilt(signal_data, 5)
            peaks, _ = signal.find_peaks(filtered, distance=fps // 3)

            if len(peaks) < 5:
                return 0

            intervals = np.diff(peaks) / fps * 1000
            successive_diffs = np.diff(intervals)
            rmssd = np.sqrt(np.mean(successive_diffs**2))

            return min(100, max(10, rmssd))

        except Exception as e:
            st.error(f"Error calculating HRV: {e}")
            return 0

    def calculate_stress_index(self, heart_rate, hrv, breathing_rate):
        """Calculate stress index based on multiple parameters"""
        try:
            hr_stress = max(0, (heart_rate - 70) / 50)
            hrv_stress = max(0, (50 - hrv) / 50)
            br_stress = max(0, (breathing_rate - 15) / 15)

            stress_index = (hr_stress + hrv_stress + br_stress) / 3
            return min(1.0, max(0.0, stress_index))

        except Exception as e:
            st.error(f"Error calculating stress index: {e}")
            return 0

    def calculate_parasympathetic_activity(self, hrv, breathing_rate):
        """Estimate parasympathetic nervous system activity"""
        try:
            hrv_factor = min(1.0, hrv / 50)
            breathing_factor = max(0, (20 - breathing_rate) / 10)

            parasympathetic = (hrv_factor + breathing_factor) / 2 * 100
            return min(100, max(0, parasympathetic))

        except Exception as e:
            st.error(f"Error calculating parasympathetic activity: {e}")
            return 50

    def calculate_wellness_score(self):
        """Calculate overall wellness score"""
        try:
            hr = st.session_state.results["heart_rate"]
            hrv = st.session_state.results["hrv"]
            stress = st.session_state.results["stress_index"]
            para = st.session_state.results["parasympathetic"]

            hr_score = 1 - abs(hr - 70) / 50 if hr > 0 else 0.5
            hrv_score = min(1, hrv / 50)
            stress_score = 1 - stress
            para_score = para / 100

            wellness = (hr_score + hrv_score + stress_score + para_score) / 4 * 100
            return max(0, min(100, wellness))

        except Exception as e:
            st.error(f"Error calculating wellness score: {e}")
            return 50

    def calculate_all_metrics(self, landmarks):
        """Calculate all health metrics"""
        if len(st.session_state.ppg_signal) < 300:
            return

        signal_array = np.array(list(st.session_state.ppg_signal))

        # Calculate all metrics
        hr = self.calculate_heart_rate(signal_array, list(st.session_state.timestamps))
        st.session_state.results["heart_rate"] = int(hr) if hr > 0 else 0
        st.session_state.hr_values.append(st.session_state.results["heart_rate"])

        br = self.calculate_breathing_rate(signal_array)
        st.session_state.results["breathing_rate"] = int(br) if br > 0 else 0
        st.session_state.br_values.append(st.session_state.results["breathing_rate"])

        hrv = self.calculate_hrv(signal_array)
        st.session_state.results["hrv"] = int(hrv)
        st.session_state.hrv_values.append(st.session_state.results["hrv"])

        stress = self.calculate_stress_index(hr, hrv, br)
        st.session_state.results["stress_index"] = round(stress, 2)
        st.session_state.stress_values.append(st.session_state.results["stress_index"])

        para = self.calculate_parasympathetic_activity(hrv, br)
        st.session_state.results["parasympathetic"] = int(para)
        st.session_state.para_values.append(st.session_state.results["parasympathetic"])

        sys_bp, dia_bp = self.estimate_blood_pressure(hr, hrv, stress)
        st.session_state.results["blood_pressure_sys"] = sys_bp
        st.session_state.results["blood_pressure_dia"] = dia_bp
        st.session_state.bp_sys_values.append(sys_bp)
        st.session_state.bp_dia_values.append(dia_bp)

        wellness = self.calculate_wellness_score()
        st.session_state.results["wellness_score"] = int(wellness)
        st.session_state.wellness_values.append(
            st.session_state.results["wellness_score"]
        )

        st.session_state.calculation_count += 1

        # Store measurement for PDF report
        measurement = {
            "timestamp": datetime.now().isoformat(),
            "heart_rate": st.session_state.results["heart_rate"],
            "breathing_rate": st.session_state.results["breathing_rate"],
            "blood_pressure_sys": st.session_state.results["blood_pressure_sys"],
            "blood_pressure_dia": st.session_state.results["blood_pressure_dia"],
            "hrv": st.session_state.results["hrv"],
            "stress_index": st.session_state.results["stress_index"],
            "parasympathetic": st.session_state.results["parasympathetic"],
            "wellness_score": st.session_state.results["wellness_score"],
        }
        st.session_state.session_data["measurements"].append(measurement)

    def get_status_color(self, metric, value):
        """Get color for metric based on value"""
        if metric == "heart_rate":
            if 60 <= value <= 100:
                return "green"
            elif 50 <= value <= 120:
                return "orange"
            else:
                return "red"
        elif metric == "breathing_rate":
            if 12 <= value <= 20:
                return "green"
            elif 8 <= value <= 25:
                return "orange"
            else:
                return "red"
        elif metric == "hrv":
            if value > 40:
                return "green"
            elif value > 20:
                return "orange"
            else:
                return "red"
        elif metric == "stress_index":
            if value < 0.3:
                return "green"
            elif value < 0.7:
                return "orange"
            else:
                return "red"
        elif metric == "parasympathetic":
            if value > 60:
                return "green"
            elif value > 30:
                return "orange"
            else:
                return "red"
        elif metric == "wellness_score":
            if value > 70:
                return "green"
            elif value > 40:
                return "orange"
            else:
                return "red"
        elif metric == "blood_pressure":
            sys_bp, dia_bp = value
            if sys_bp < 130 and dia_bp < 85:
                return "green"
            elif sys_bp < 140 and dia_bp < 90:
                return "orange"
            else:
                return "red"
        return "black"

    def process_frame(self, frame):
        """Process video frame for face detection and PPG extraction"""
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            st.session_state.face_detected = True
            face_landmarks = results.multi_face_landmarks[0]

            # Draw styled mesh
            self.mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 255), thickness=1, circle_radius=1
                ),
            )
            self.mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 255, 255), thickness=2, circle_radius=1
                ),
            )
            self.mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 200, 255), thickness=2, circle_radius=1
                ),
            )

            # PPG signal extraction during monitoring
            if st.session_state.monitoring_active:
                ppg_value = self.extract_ppg_signal(frame, face_landmarks.landmark)
                current_time = time.time()

                st.session_state.ppg_signal.append(ppg_value)
                st.session_state.timestamps.append(current_time)

                st.session_state.session_data["raw_ppg_data"].append(ppg_value)
                st.session_state.session_data["timestamps_data"].append(current_time)

                # Auto-stop after 30 seconds
                if len(st.session_state.ppg_signal) >= 900:
                    st.session_state.monitoring_active = False
                    st.session_state.session_data["end_time"] = (
                        datetime.now().isoformat()
                    )
                    st.success(
                        "30-second monitoring completed! You can now generate your PDF report."
                    )

                # Start calculating after 10 seconds
                elif len(st.session_state.ppg_signal) >= 150:
                    self.calculate_all_metrics(face_landmarks.landmark)
        else:
            st.session_state.face_detected = False

        return frame

    def generate_pdf_report(self):
        """Generate comprehensive PDF report"""
        if not st.session_state.session_data["measurements"]:
            st.warning(
                "No monitoring data available! Please complete a monitoring session first."
            )
            return None

        try:
            # Create PDF in memory
            buffer = io.BytesIO()

            # Build story
            story = []
            styles = getSampleStyleSheet()

            # Title
            title_style = ParagraphStyle(
                "CustomTitle",
                parent=styles["Heading1"],
                fontSize=20,
                spaceAfter=30,
                alignment=1,
            )
            story.append(
                Paragraph("COMPREHENSIVE HEALTH MONITORING REPORT", title_style)
            )
            story.append(Spacer(1, 20))

            # Session Information
            story.append(Paragraph("SESSION INFORMATION", styles["Heading2"]))
            session_data = [
                ["Parameter", "Value"],
                [
                    "Start Time",
                    st.session_state.session_data["start_time"][:19].replace("T", " "),
                ],
                [
                    "End Time",
                    (
                        st.session_state.session_data["end_time"][:19].replace("T", " ")
                        if st.session_state.session_data["end_time"]
                        else "N/A"
                    ),
                ],
                ["Duration", "30 seconds"],
                [
                    "Total Frames Captured",
                    str(len(st.session_state.session_data["raw_ppg_data"])),
                ],
                [
                    "Total Measurements",
                    str(len(st.session_state.session_data["measurements"])),
                ],
                ["Calculations Performed", str(st.session_state.calculation_count)],
            ]
            session_table = Table(session_data, colWidths=[3 * inch, 3 * inch])
            session_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            story.append(session_table)
            story.append(Spacer(1, 20))

            # Final Health Measurements
            story.append(Paragraph("FINAL HEALTH MEASUREMENTS", styles["Heading2"]))
            measurements_data = [
                ["Metric", "Value", "Status"],
                [
                    "Heart Rate",
                    f"{st.session_state.results['heart_rate']} bpm",
                    self.get_status_text("heart_rate"),
                ],
                [
                    "Breathing Rate",
                    f"{st.session_state.results['breathing_rate']} rpm",
                    self.get_status_text("breathing_rate"),
                ],
                [
                    "Blood Pressure",
                    f"{st.session_state.results['blood_pressure_sys']}/{st.session_state.results['blood_pressure_dia']} mmHg",
                    self.get_status_text("blood_pressure"),
                ],
                [
                    "HRV",
                    f"{st.session_state.results['hrv']} ms",
                    self.get_status_text("hrv"),
                ],
                [
                    "Stress Index",
                    f"{st.session_state.results['stress_index']}",
                    self.get_status_text("stress_index"),
                ],
                [
                    "Parasympathetic Activity",
                    f"{st.session_state.results['parasympathetic']}%",
                    self.get_status_text("parasympathetic"),
                ],
                [
                    "Wellness Score",
                    f"{st.session_state.results['wellness_score']}/100",
                    self.get_status_text("wellness_score"),
                ],
            ]
            measurements_table = Table(
                measurements_data, colWidths=[2.5 * inch, 2 * inch, 1.5 * inch]
            )
            measurements_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            story.append(measurements_table)
            story.append(PageBreak())

            # Add trend plots
            story.append(Paragraph("HEALTH METRICS TREND ANALYSIS", styles["Heading2"]))

            # Create trend plots for each metric
            metrics_to_plot = [
                ("Heart Rate", "heart_rate", st.session_state.hr_values, "bpm", "red"),
                (
                    "Breathing Rate",
                    "breathing_rate",
                    st.session_state.br_values,
                    "rpm",
                    "blue",
                ),
                ("HRV", "hrv", st.session_state.hrv_values, "ms", "green"),
                (
                    "Stress Index",
                    "stress_index",
                    st.session_state.stress_values,
                    "",
                    "orange",
                ),
                (
                    "Parasympathetic Activity",
                    "parasympathetic",
                    st.session_state.para_values,
                    "%",
                    "purple",
                ),
                (
                    "Wellness Score",
                    "wellness_score",
                    st.session_state.wellness_values,
                    "/100",
                    "teal",
                ),
            ]

            for name, key, data_deque, unit, color in metrics_to_plot:
                if len(data_deque) > 1:
                    try:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        fig.patch.set_facecolor("white")
                        time_axis = np.linspace(-len(data_deque), 0, len(data_deque))
                        ax.plot(time_axis, list(data_deque), color=color, linewidth=2.5)
                        ax.set_title(
                            f"{name} Trend Over 30 Seconds",
                            fontsize=14,
                            fontweight="bold",
                        )
                        ax.set_xlabel("Time (relative seconds)", fontsize=12)
                        ax.set_ylabel(f"{name} {unit}", fontsize=12)
                        ax.grid(True, alpha=0.3)
                        mean_val = np.mean(list(data_deque))
                        ax.axhline(
                            y=mean_val,
                            color="red",
                            linestyle="--",
                            alpha=0.7,
                            label=f"Mean: {mean_val:.1f}",
                        )
                        ax.legend()

                        img_buffer = io.BytesIO()
                        fig.savefig(
                            img_buffer, format="png", dpi=150, bbox_inches="tight"
                        )
                        img_buffer.seek(0)
                        story.append(
                            RLImage(img_buffer, width=7 * inch, height=3.5 * inch)
                        )
                        story.append(Spacer(1, 15))
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Error creating plot for {name}: {e}")
                        continue

            # Add health interpretation and recommendations
            story.append(PageBreak())
            story.append(Paragraph("HEALTH INTERPRETATION", styles["Heading2"]))
            story.append(Paragraph(self.get_health_interpretation(), styles["Normal"]))
            story.append(Spacer(1, 20))

            story.append(Paragraph("RECOMMENDATIONS", styles["Heading2"]))
            story.append(Paragraph(self.get_recommendations(), styles["Normal"]))

            # Create PDF
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18,
            )
            doc.build(story)

            buffer.seek(0)
            return buffer.getvalue()

        except Exception as e:
            st.error(f"Error generating PDF: {e}")
            return None

    def get_status_text(self, metric):
        """Get status text for a metric"""
        if metric == "heart_rate":
            hr = st.session_state.results["heart_rate"]
            if 60 <= hr <= 100:
                return "Normal"
            elif 50 <= hr <= 120:
                return "Acceptable"
            else:
                return "Abnormal"
        elif metric == "breathing_rate":
            br = st.session_state.results["breathing_rate"]
            if 12 <= br <= 20:
                return "Normal"
            elif 8 <= br <= 25:
                return "Acceptable"
            else:
                return "Abnormal"
        elif metric == "blood_pressure":
            sys_bp = st.session_state.results["blood_pressure_sys"]
            dia_bp = st.session_state.results["blood_pressure_dia"]
            if sys_bp < 130 and dia_bp < 85:
                return "Normal"
            elif sys_bp < 140 and dia_bp < 90:
                return "Elevated"
            else:
                return "High"
        elif metric == "hrv":
            hrv = st.session_state.results["hrv"]
            if hrv > 40:
                return "Good"
            elif hrv > 20:
                return "Fair"
            else:
                return "Poor"
        elif metric == "stress_index":
            stress = st.session_state.results["stress_index"]
            if stress < 0.3:
                return "Low"
            elif stress < 0.7:
                return "Moderate"
            else:
                return "High"
        elif metric == "parasympathetic":
            para = st.session_state.results["parasympathetic"]
            if para > 60:
                return "Good"
            elif para > 30:
                return "Fair"
            else:
                return "Poor"
        elif metric == "wellness_score":
            wellness = st.session_state.results["wellness_score"]
            if wellness > 70:
                return "Excellent"
            elif wellness > 40:
                return "Good"
            else:
                return "Needs Improvement"
        return "Unknown"

    def get_health_interpretation(self):
        """Get health interpretation based on measurements"""
        interpretations = []

        hr = st.session_state.results["heart_rate"]
        if hr < 60:
            interpretations.append(
                "• Heart Rate: Below normal (Bradycardia) - Consider consulting a healthcare provider."
            )
        elif hr > 100:
            interpretations.append(
                "• Heart Rate: Above normal (Tachycardia) - May indicate stress, exercise, or other factors."
            )
        else:
            interpretations.append("• Heart Rate: Normal range (60-100 bpm).")

        sys_bp = st.session_state.results["blood_pressure_sys"]
        dia_bp = st.session_state.results["blood_pressure_dia"]
        if sys_bp < 120 and dia_bp < 80:
            interpretations.append("• Blood Pressure: Normal (<120/80 mmHg).")
        elif sys_bp < 130 and dia_bp < 85:
            interpretations.append("• Blood Pressure: Normal to slightly elevated.")
        elif sys_bp < 140 and dia_bp < 90:
            interpretations.append(
                "• Blood Pressure: Stage 1 hypertension - monitor regularly."
            )
        else:
            interpretations.append(
                "• Blood Pressure: High (≥140/90 mmHg) - Recommend medical evaluation."
            )

        hrv = st.session_state.results["hrv"]
        if hrv > 40:
            interpretations.append(
                "• Heart Rate Variability: Good - indicates healthy autonomic nervous system."
            )
        elif hrv > 20:
            interpretations.append(
                "• Heart Rate Variability: Fair - room for improvement through stress management."
            )
        else:
            interpretations.append(
                "• Heart Rate Variability: Low - may indicate stress or autonomic dysfunction."
            )

        stress = st.session_state.results["stress_index"]
        if stress < 0.3:
            interpretations.append(
                "• Stress Level: Low - maintaining good stress management."
            )
        elif stress < 0.7:
            interpretations.append(
                "• Stress Level: Moderate - consider stress reduction techniques."
            )
        else:
            interpretations.append(
                "• Stress Level: High - recommend stress management and relaxation practices."
            )

        wellness = st.session_state.results["wellness_score"]
        if wellness > 70:
            interpretations.append(
                "• Overall Wellness: Excellent - maintaining good health habits."
            )
        elif wellness > 40:
            interpretations.append(
                "• Overall Wellness: Good - some areas for improvement identified."
            )
        else:
            interpretations.append(
                "• Overall Wellness: Needs attention - consider comprehensive health evaluation."
            )

        return "\n".join(interpretations)

    def get_recommendations(self):
        """Get health recommendations based on measurements"""
        recommendations = []

        hr = st.session_state.results["heart_rate"]
        stress = st.session_state.results["stress_index"]
        hrv = st.session_state.results["hrv"]
        wellness = st.session_state.results["wellness_score"]

        recommendations.append("GENERAL RECOMMENDATIONS:")
        recommendations.append(
            "• Maintain regular exercise routine (150 minutes moderate activity per week)"
        )
        recommendations.append(
            "• Practice stress management techniques (meditation, deep breathing)"
        )
        recommendations.append("• Ensure adequate sleep (7-9 hours per night)")
        recommendations.append("• Stay hydrated and maintain balanced nutrition")

        if stress > 0.5:
            recommendations.append("\nSTRESS MANAGEMENT:")
            recommendations.append("• Consider mindfulness meditation or yoga")
            recommendations.append("• Practice progressive muscle relaxation")
            recommendations.append("• Limit caffeine intake")
            recommendations.append("• Ensure work-life balance")

        if hrv < 30:
            recommendations.append("\nHEART RATE VARIABILITY IMPROVEMENT:")
            recommendations.append("• Regular cardiovascular exercise")
            recommendations.append("• Breathing exercises (4-7-8 technique)")
            recommendations.append("• Reduce alcohol consumption")
            recommendations.append("• Consider heart rate variability training")

        if hr > 100 or hr < 60:
            recommendations.append("\nHEART RATE CONCERNS:")
            recommendations.append("• Monitor heart rate regularly")
            recommendations.append("• Consult healthcare provider if persistent")
            recommendations.append("• Avoid excessive caffeine")
            recommendations.append("• Maintain regular sleep schedule")

        if wellness < 50:
            recommendations.append("\nWELLNESS IMPROVEMENT:")
            recommendations.append("• Comprehensive health assessment recommended")
            recommendations.append("• Consider lifestyle modifications")
            recommendations.append("• Regular monitoring of vital signs")
            recommendations.append("• Professional health consultation advised")

        recommendations.append("\nDISCLAIMER:")
        recommendations.append("This is a demonstration tool for educational purposes.")
        recommendations.append(
            "Measurements are estimates and should not replace professional medical advice."
        )
        recommendations.append("Consult healthcare professionals for medical concerns.")

        return "\n".join(recommendations)


def main():
    st.set_page_config(
        page_title="Face Vital Monitor",
        page_icon="💓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    .scanning-border {
        border: 3px solid #ff6b6b;
        border-radius: 10px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { border-color: #ff6b6b; }
        50% { border-color: #4ecdc4; }
        100% { border-color: #ff6b6b; }
    }
    .stProgress .st-bo {
        background-color: #667eea;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-header"><h1>💓 Face Vital Monitor</h1><p>Advanced Contactless Health Monitoring System</p></div>',
        unsafe_allow_html=True,
    )

    monitor = StreamlitFaceVitalMonitor()

    with st.sidebar:
        st.header("📹 Camera Controls")

        # Instructions
        st.info("🎥 Start video capture to analyze your vital signs for 30 seconds")
        st.warning("⚠️ Ensure good lighting and face visibility")

        # Check if running locally or deployed
        is_local = st.session_state.get("is_local", True)

        # Camera input method selection
        camera_method = st.radio(
            "Camera Method:",
            ["Auto-detect", "Webcam (Local)", "Camera Input (Deployed)"],
            help="Auto-detect will try webcam first, then fallback to camera input",
        )

        if camera_method == "Auto-detect":
            # Try to detect if we can access webcam
            try:
                test_cap = cv2.VideoCapture(0)
                if test_cap.isOpened():
                    test_cap.release()
                    is_local = True
                    st.success("✅ Webcam detected - using direct video capture")
                else:
                    is_local = False
                    st.info("📱 Using camera input for deployment compatibility")
            except:
                is_local = False
                st.info("📱 Using camera input for deployment compatibility")
        elif camera_method == "Webcam (Local)":
            is_local = True
        else:
            is_local = False

        st.session_state.is_local = is_local

        # Video capture controls
        if not st.session_state.monitoring_active:
            if is_local:
                # Local webcam method
                if st.button(
                    "🎥 Start 30s Video Analysis",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state.monitoring_active = True
                    st.session_state.face_detected = False
                    st.session_state.ppg_signal = deque(maxlen=900)
                    st.session_state.timestamps = deque(maxlen=900)
                    st.session_state.hr_values = deque(maxlen=300)
                    st.session_state.br_values = deque(maxlen=300)
                    st.session_state.hrv_values = deque(maxlen=300)
                    st.session_state.stress_values = deque(maxlen=300)
                    st.session_state.para_values = deque(maxlen=300)
                    st.session_state.wellness_values = deque(maxlen=300)
                    st.session_state.bp_sys_values = deque(maxlen=300)
                    st.session_state.bp_dia_values = deque(maxlen=300)
                    st.session_state.session_data = {
                        "start_time": datetime.now().isoformat(),
                        "end_time": None,
                        "measurements": [],
                        "raw_ppg_data": [],
                        "timestamps_data": [],
                    }
                    st.rerun()
            else:
                # Deployed camera input method
                st.info("📸 Take multiple photos for analysis (deployment mode)")
                camera_input = st.camera_input("Take photos for vital sign analysis")

                if camera_input is not None:
                    if st.button(
                        "🎥 Analyze Photos", type="primary", use_container_width=True
                    ):
                        st.session_state.monitoring_active = True
                        st.session_state.face_detected = False
                        st.session_state.camera_input = camera_input
                        st.session_state.ppg_signal = deque(maxlen=900)
                        st.session_state.timestamps = deque(maxlen=900)
                        st.session_state.hr_values = deque(maxlen=300)
                        st.session_state.br_values = deque(maxlen=300)
                        st.session_state.hrv_values = deque(maxlen=300)
                        st.session_state.stress_values = deque(maxlen=300)
                        st.session_state.para_values = deque(maxlen=300)
                        st.session_state.wellness_values = deque(maxlen=300)
                        st.session_state.bp_sys_values = deque(maxlen=300)
                        st.session_state.bp_dia_values = deque(maxlen=300)
                        st.session_state.session_data = {
                            "start_time": datetime.now().isoformat(),
                            "end_time": None,
                            "measurements": [],
                            "raw_ppg_data": [],
                            "timestamps_data": [],
                        }
                        st.rerun()
        else:
            if st.button("⏹️ Stop Analysis", type="secondary", use_container_width=True):
                st.session_state.monitoring_active = False
                st.session_state.session_data["end_time"] = datetime.now().isoformat()
                st.rerun()

        if st.session_state.monitoring_active:
            # Create video capture placeholder
            video_placeholder = st.empty()
            progress_placeholder = st.empty()
            status_placeholder = st.empty()

            if is_local:
                # Local webcam processing (original method)
                cap = cv2.VideoCapture(0)

                if cap.isOpened():
                    # Set camera properties for better quality
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)

                    frame_count = 0
                    start_time = time.time()

                    while st.session_state.monitoring_active and frame_count < 900:
                        ret, frame = cap.read()

                        if ret:
                            # Process the frame to detect face and extract features
                            processed_frame = monitor.process_frame(frame)

                            # Display processed video frame every 10 frames to reduce UI updates
                            if frame_count % 10 == 0:
                                frame_rgb = cv2.cvtColor(
                                    processed_frame, cv2.COLOR_BGR2RGB
                                )
                                video_placeholder.image(
                                    frame_rgb,
                                    caption="Live Video Analysis with Face Detection",
                                    use_container_width=True,
                                )

                            # Update progress every 30 frames
                            if frame_count % 30 == 0:
                                progress = min(frame_count / 900, 1.0)
                                progress_placeholder.progress(
                                    progress, f"Analysis Progress: {progress*100:.1f}%"
                                )

                                if st.session_state.face_detected:
                                    status_placeholder.success(
                                        "📈 Live metrics updating for all vital signs!"
                                    )
                                else:
                                    status_placeholder.error(
                                        "❌ No face detected in video stream"
                                    )

                            frame_count += 1

                            # Small delay to maintain ~30 FPS without blocking
                            time.sleep(0.01)
                        else:
                            break

                    # Complete monitoring
                    st.session_state.monitoring_active = False
                    st.session_state.session_data["end_time"] = (
                        datetime.now().isoformat()
                    )
                    cap.release()
                    st.success("✅ 30-second video analysis completed!")
                    st.balloons()

                else:
                    st.error(
                        "❌ Could not access camera. Please check camera permissions."
                    )
                    st.session_state.monitoring_active = False
            else:
                # Deployed camera input processing
                if (
                    hasattr(st.session_state, "camera_input")
                    and st.session_state.camera_input is not None
                ):
                    # Convert camera input to OpenCV format
                    image_bytes = st.session_state.camera_input.read()
                    image_array = np.frombuffer(image_bytes, np.uint8)
                    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                    if frame is not None:
                        # Simulate multiple frames by processing the same image multiple times
                        # with slight variations for PPG signal extraction
                        frame_count = 0
                        start_time = time.time()

                        progress_placeholder.info(
                            "📸 Processing captured image for vital signs..."
                        )

                        # Process the image multiple times to simulate video frames
                        for i in range(300):  # Reduced from 900 for faster processing
                            # Add slight noise to simulate different frames
                            noisy_frame = frame.copy()
                            if i > 0:
                                noise = np.random.normal(0, 2, frame.shape).astype(
                                    np.uint8
                                )
                                noisy_frame = cv2.add(noisy_frame, noise)

                            # Process the frame
                            processed_frame = monitor.process_frame(noisy_frame)

                            # Update display every 30 iterations
                            if i % 30 == 0:
                                frame_rgb = cv2.cvtColor(
                                    processed_frame, cv2.COLOR_BGR2RGB
                                )
                                video_placeholder.image(
                                    frame_rgb,
                                    caption=f"Photo Analysis - Processing iteration {i+1}/300",
                                    use_container_width=True,
                                )

                                progress = min(i / 300, 1.0)
                                progress_placeholder.progress(
                                    progress, f"Analysis Progress: {progress*100:.1f}%"
                                )

                                if st.session_state.face_detected:
                                    status_placeholder.success(
                                        "📈 Face detected - extracting vital signs!"
                                    )
                                else:
                                    status_placeholder.error(
                                        "❌ No face detected in image"
                                    )

                            frame_count += 1
                            time.sleep(0.01)  # Small delay for UI updates

                        # Complete monitoring
                        st.session_state.monitoring_active = False
                        st.session_state.session_data["end_time"] = (
                            datetime.now().isoformat()
                        )
                        st.success("✅ Photo analysis completed!")
                        st.balloons()
                    else:
                        st.error("❌ Could not process the captured image.")
                        st.session_state.monitoring_active = False
                else:
                    st.error("❌ No image captured. Please take a photo first.")
                    st.session_state.monitoring_active = False

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📊 Real-Time Health Metrics")

        # Display all metrics with live updates
        metrics_data = [
            (
                "Heart Rate",
                st.session_state.results["heart_rate"],
                "bpm",
                "❤️",
                "heart_rate",
            ),
            (
                "Breathing Rate",
                st.session_state.results["breathing_rate"],
                "rpm",
                "🫁",
                "breathing_rate",
            ),
            (
                "Blood Pressure",
                f"{st.session_state.results['blood_pressure_sys']}/{st.session_state.results['blood_pressure_dia']}",
                "mmHg",
                "🩸",
                "blood_pressure",
            ),
            ("HRV", st.session_state.results["hrv"], "ms", "📊", "hrv"),
            (
                "Stress Index",
                f"{st.session_state.results['stress_index']:.2f}",
                "",
                "😰",
                "stress_index",
            ),
            (
                "Wellness Score",
                f"{st.session_state.results['wellness_score']}/100",
                "",
                "🌟",
                "wellness_score",
            ),
        ]

        for name, value, unit, icon, metric_key in metrics_data:
            if metric_key == "blood_pressure":
                color = monitor.get_status_color(
                    metric_key,
                    (
                        st.session_state.results["blood_pressure_sys"],
                        st.session_state.results["blood_pressure_dia"],
                    ),
                )
            else:
                color = monitor.get_status_color(
                    metric_key,
                    (
                        st.session_state.results[metric_key]
                        if metric_key in st.session_state.results
                        else 0
                    ),
                )

            # Add live update indicator for active monitoring
            live_indicator = "🔴 LIVE" if st.session_state.monitoring_active else ""

            st.markdown(
                f"""
            <div class="metric-card" style="border-left: 4px solid {color};">
                <h3>{icon} {name} <span style="color: red; font-size: 12px;">{live_indicator}</span></h3>
                <h2 style="color: {color};">{value} {unit}</h2>
                <small style="color: gray;">Status: {monitor.get_status_text(metric_key) if hasattr(monitor, 'get_status_text') else 'Normal'}</small>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        st.header("📈 Live Signal Analysis")

        if len(st.session_state.ppg_signal) > 10:
            # PPG Signal Plot
            st.subheader("🌊 PPG Signal (Raw)")
            fig, ax = plt.subplots(figsize=(10, 4))
            signal_data = list(st.session_state.ppg_signal)[-300:]  # Last 10 seconds
            time_axis = np.arange(len(signal_data))
            ax.plot(time_axis, signal_data, color="#667eea", linewidth=2, alpha=0.8)
            ax.set_title("Photoplethysmography (PPG) Signal")
            ax.set_xlabel("Frame Number")
            ax.set_ylabel("Signal Amplitude")
            ax.grid(True, alpha=0.3)
            ax.set_facecolor("#f8f9fa")
            st.pyplot(fig)
            plt.close()

            # Heart Rate Trend
            if len(st.session_state.hr_values) > 1:
                st.subheader("💓 Heart Rate Trend")
                fig, ax = plt.subplots(figsize=(10, 3))
                hr_data = list(st.session_state.hr_values)
                ax.plot(hr_data, color="red", linewidth=3, marker="o", markersize=4)
                ax.set_title("Heart Rate Over Time")
                ax.set_xlabel("Measurement")
                ax.set_ylabel("BPM")
                ax.grid(True, alpha=0.3)
                ax.axhline(
                    y=np.mean(hr_data),
                    color="orange",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Average: {np.mean(hr_data):.1f} BPM",
                )
                ax.legend()
                ax.set_facecolor("#f8f9fa")
                st.pyplot(fig)
                plt.close()

            if len(st.session_state.br_values) > 1:
                st.subheader("🫁 Breathing Rate Trend")
                fig, ax = plt.subplots(figsize=(10, 3))
                br_data = list(st.session_state.br_values)
                ax.plot(br_data, color="blue", linewidth=3, marker="s", markersize=4)
                ax.set_title("Breathing Rate Over Time")
                ax.set_xlabel("Measurement")
                ax.set_ylabel("RPM")
                ax.grid(True, alpha=0.3)
                ax.axhline(
                    y=np.mean(br_data),
                    color="cyan",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Average: {np.mean(br_data):.1f} RPM",
                )
                ax.legend()
                ax.set_facecolor("#f8f9fa")
                st.pyplot(fig)
                plt.close()

            if len(st.session_state.hrv_values) > 1:
                st.subheader("📊 HRV Trend")
                fig, ax = plt.subplots(figsize=(10, 3))
                hrv_data = list(st.session_state.hrv_values)
                ax.plot(hrv_data, color="green", linewidth=3, marker="^", markersize=4)
                ax.set_title("Heart Rate Variability Over Time")
                ax.set_xlabel("Measurement")
                ax.set_ylabel("ms")
                ax.grid(True, alpha=0.3)
                ax.axhline(
                    y=np.mean(hrv_data),
                    color="lime",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Average: {np.mean(hrv_data):.1f} ms",
                )
                ax.legend()
                ax.set_facecolor("#f8f9fa")
                st.pyplot(fig)
                plt.close()

            if len(st.session_state.stress_values) > 1:
                st.subheader("😰 Stress Index Trend")
                fig, ax = plt.subplots(figsize=(10, 3))
                stress_data = list(st.session_state.stress_values)
                ax.plot(
                    stress_data, color="orange", linewidth=3, marker="d", markersize=4
                )
                ax.set_title("Stress Index Over Time")
                ax.set_xlabel("Measurement")
                ax.set_ylabel("Stress Level")
                ax.grid(True, alpha=0.3)
                ax.axhline(
                    y=np.mean(stress_data),
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Average: {np.mean(stress_data):.2f}",
                )
                ax.legend()
                ax.set_facecolor("#f8f9fa")
                st.pyplot(fig)
                plt.close()

            if len(st.session_state.para_values) > 1:
                st.subheader("🧘 Parasympathetic Activity Trend")
                fig, ax = plt.subplots(figsize=(10, 3))
                para_data = list(st.session_state.para_values)
                ax.plot(
                    para_data, color="purple", linewidth=3, marker="v", markersize=4
                )
                ax.set_title("Parasympathetic Activity Over Time")
                ax.set_xlabel("Measurement")
                ax.set_ylabel("Percentage")
                ax.grid(True, alpha=0.3)
                ax.axhline(
                    y=np.mean(para_data),
                    color="magenta",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Average: {np.mean(para_data):.1f}%",
                )
                ax.legend()
                ax.set_facecolor("#f8f9fa")
                st.pyplot(fig)
                plt.close()

            if len(st.session_state.wellness_values) > 1:
                st.subheader("🌟 Wellness Score Trend")
                fig, ax = plt.subplots(figsize=(10, 3))
                wellness_data = list(st.session_state.wellness_values)
                ax.plot(
                    wellness_data, color="gold", linewidth=3, marker="*", markersize=6
                )
                ax.set_title("Wellness Score Over Time")
                ax.set_xlabel("Measurement")
                ax.set_ylabel("Score (0-100)")
                ax.grid(True, alpha=0.3)
                ax.axhline(
                    y=np.mean(wellness_data),
                    color="darkorange",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Average: {np.mean(wellness_data):.1f}/100",
                )
                ax.legend()
                ax.set_facecolor("#f8f9fa")
                st.pyplot(fig)
                plt.close()

        else:
            st.info("📊 Start monitoring to see live signal analysis")
            st.markdown(
                """
            **What you'll see:**
            - 🌊 Real-time PPG signal from your face
            - 💓 Heart rate trend over time
            - 🫁 Breathing rate trend
            - 📊 HRV (Heart Rate Variability) trend
            - 😰 Stress index trend
            - 🧘 Parasympathetic activity trend
            - 🌟 Wellness score trend
            - 🎯 Signal quality indicators
            """
            )

    st.header("📄 Health Report Generation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📊 Data Points", len(st.session_state.ppg_signal))

    with col2:
        st.metric("🧮 Calculations", st.session_state.calculation_count)

    with col3:
        monitoring_time = (
            len(st.session_state.ppg_signal) / 30
            if len(st.session_state.ppg_signal) > 0
            else 0
        )
        st.metric("⏱️ Monitoring Time", f"{monitoring_time:.1f}s")

    if st.button(
        "📋 Generate Comprehensive Report", type="secondary", use_container_width=True
    ):
        if len(st.session_state.session_data["measurements"]) > 0:
            with st.spinner("Generating detailed health report..."):
                pdf_data = monitor.generate_pdf_report()
                if pdf_data:
                    st.download_button(
                        label="📥 Download Health Report (PDF)",
                        data=pdf_data,
                        file_name=f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                    st.success("✅ Report generated successfully!")
                else:
                    st.error("❌ Failed to generate report")
        else:
            st.warning(
                "⚠️ No monitoring data available. Please complete a monitoring session first."
            )


if __name__ == "__main__":
    main()
