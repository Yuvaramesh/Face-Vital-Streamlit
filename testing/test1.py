import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from collections import deque
import base64
from io import BytesIO
from PIL import Image
import threading
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av  # Added missing av import for WebRTC video frame processing
import mediapipe as mp
from scipy import signal
from scipy.fft import fft
import matplotlib.pyplot as plt
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
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
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
        if "frame_count" not in st.session_state:
            st.session_state.frame_count = 0

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

                    # Use green channel for better PPG signal
                    if len(frame.shape) == 3:
                        green_channel = frame[:, :, 1]
                    else:
                        green_channel = frame

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
        if len(st.session_state.ppg_signal) < 240:  # Wait for 8 seconds of data
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
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            st.session_state.face_detected = True
            face_landmarks = results.multi_face_landmarks[0]

            # Draw face mesh with enhanced styling
            self.mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 255), thickness=1, circle_radius=1
                ),
            )

            # Draw contours
            self.mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 255, 255), thickness=2, circle_radius=1
                ),
            )

            # Draw irises
            self.mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=3, circle_radius=2
                ),
            )

            # Add status text overlay
            cv2.putText(
                frame,
                "FACE DETECTED - MONITORING ACTIVE",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Add frame count
            cv2.putText(
                frame,
                f"Frame: {st.session_state.frame_count}/900",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # PPG signal extraction during monitoring
            if st.session_state.monitoring_active:
                ppg_value = self.extract_ppg_signal(frame_rgb, face_landmarks.landmark)
                current_time = time.time()

                st.session_state.ppg_signal.append(ppg_value)
                st.session_state.timestamps.append(current_time)
                st.session_state.frame_count += 1

                st.session_state.session_data["raw_ppg_data"].append(ppg_value)
                st.session_state.session_data["timestamps_data"].append(current_time)

                # Add PPG signal value overlay
                cv2.putText(
                    frame,
                    f"PPG: {ppg_value:.2f}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )

                # Calculate metrics every 30 frames (once per second at 30fps)
                if (
                    st.session_state.frame_count % 30 == 0
                    and len(st.session_state.ppg_signal) >= 240
                ):
                    try:
                        self.calculate_all_metrics(face_landmarks.landmark)
                        print(
                            f"Metrics calculated at frame {st.session_state.frame_count}"
                        )
                    except Exception as e:
                        print(f"Error calculating metrics: {e}")

                # Add current metrics overlay
                if st.session_state.results["heart_rate"] > 0:
                    cv2.putText(
                        frame,
                        f"HR: {st.session_state.results['heart_rate']} bpm",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )
                    cv2.putText(
                        frame,
                        f"BR: {st.session_state.results['breathing_rate']} rpm",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 255),
                        1,
                    )
                    cv2.putText(
                        frame,
                        f"HRV: {st.session_state.results['hrv']} ms",
                        (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

                # Auto-stop after 30 seconds (900 frames at 30fps)
                if st.session_state.frame_count >= 900:
                    st.session_state.monitoring_active = False
                    st.session_state.session_data["end_time"] = (
                        datetime.now().isoformat()
                    )
                    print("Monitoring completed - 900 frames reached")

        else:
            st.session_state.face_detected = False
            # Add "no face detected" overlay
            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        return frame

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

    # [Include all the PDF generation methods from the original code here]
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


def create_deployment_video_monitor():
    """Create video monitoring for deployment using streamlit-webrtc"""

    # Initialize monitor if not exists
    if "monitor" not in st.session_state:
        st.session_state.monitor = StreamlitFaceVitalMonitor()

    def video_frame_callback(frame):
        """Process each video frame"""
        if not st.session_state.monitoring_active:
            return frame

        img = frame.to_ndarray(format="bgr24")

        # Process frame with the monitor - this will update session state
        processed_frame = st.session_state.monitor.process_frame(img)

        # Force metric calculation if we have enough data
        if (
            st.session_state.frame_count % 30 == 0
            and len(st.session_state.ppg_signal) >= 240
            and st.session_state.face_detected
        ):
            try:
                # Get the latest face landmarks from the processed frame
                frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = st.session_state.monitor.face_mesh.process(frame_rgb)

                if results.multi_face_landmarks:
                    st.session_state.monitor.calculate_all_metrics(
                        results.multi_face_landmarks[0].landmark
                    )
            except Exception as e:
                print(f"Error in metric calculation: {e}")

        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

    # Configure WebRTC
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Create WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="face-vital-monitor",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {
                "width": {"min": 640, "ideal": 640},
                "height": {"min": 480, "ideal": 480},
                "frameRate": {"min": 15, "ideal": 30},
            },
            "audio": False,
        },
        async_processing=True,
    )

    return webrtc_ctx


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
        box_shadow: 0 2px 4px rgba(0,0,0,0.1);
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

    # Initialize monitor
    if "monitor" not in st.session_state:
        st.session_state.monitor = StreamlitFaceVitalMonitor()

    monitor = st.session_state.monitor

    with st.sidebar:
        st.header("📹 Camera Controls")

        # Instructions
        st.info("🎥 Start video capture to analyze your vital signs for 30 seconds")
        st.warning("⚠️ Ensure good lighting and face visibility")

        # Camera input method selection
        camera_method = st.radio(
            "Camera Method:",
            ["Auto-detect", "Webcam (Local)", "WebRTC (Deployed)"],
            help="Auto-detect will try webcam first, then fallback to WebRTC",
        )

        # Determine camera method
        is_local = True
        if camera_method == "Auto-detect":
            try:
                test_cap = cv2.VideoCapture(0)
                if test_cap.isOpened():
                    test_cap.release()
                    is_local = True
                    st.success("✅ Webcam detected - using direct video capture")
                else:
                    is_local = False
                    st.info("📱 Using WebRTC for deployment compatibility")
            except:
                is_local = False
                st.info("📱 Using WebRTC for deployment compatibility")
        elif camera_method == "Webcam (Local)":
            is_local = True
        else:
            is_local = False

        st.session_state.is_local = is_local

        # Reset/Start button
        if not st.session_state.monitoring_active:
            if st.button(
                "🎥 Start 30s Video Analysis", type="primary", use_container_width=True
            ):
                # Reset all session state variables
                st.session_state.monitoring_active = True
                st.session_state.face_detected = False
                st.session_state.ppg_signal = deque(maxlen=900)
                st.session_state.timestamps = deque(maxlen=900)
                st.session_state.frame_count = 0
                st.session_state.calculation_count = 0

                # Reset metric values
                st.session_state.hr_values = deque(maxlen=300)
                st.session_state.br_values = deque(maxlen=300)
                st.session_state.hrv_values = deque(maxlen=300)
                st.session_state.stress_values = deque(maxlen=300)
                st.session_state.para_values = deque(maxlen=300)
                st.session_state.wellness_values = deque(maxlen=300)
                st.session_state.bp_sys_values = deque(maxlen=300)
                st.session_state.bp_dia_values = deque(maxlen=300)

                # Reset results
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

                # Reset session data
                st.session_state.session_data = {
                    "start_time": datetime.now().isoformat(),
                    "end_time": None,
                    "measurements": [],
                    "raw_ppg_data": [],
                    "timestamps_data": [],
                }

                st.rerun()

        else:
            # Show stop button during monitoring
            if st.button(
                "🛑 Stop Monitoring", type="secondary", use_container_width=True
            ):
                st.session_state.monitoring_active = False
                st.session_state.session_data["end_time"] = datetime.now().isoformat()
                st.info("Monitoring stopped manually")
                st.rerun()

        # Show monitoring status
        if st.session_state.monitoring_active:
            progress = min(st.session_state.frame_count / 900, 1.0)
            st.progress(
                progress, f"Progress: {st.session_state.frame_count}/900 frames"
            )

            if st.session_state.face_detected:
                st.success("Face detected - collecting data")
            else:
                st.error("No face detected")

            st.metric("PPG Samples", len(st.session_state.ppg_signal))
            st.metric("Calculations", st.session_state.calculation_count)

            # Add refresh button for WebRTC mode
            if not is_local:
                if st.button("Refresh Status", use_container_width=True):
                    st.rerun()

            # Show current metrics if available
            if st.session_state.results["heart_rate"] > 0:
                st.markdown("### Current Metrics")
                st.write(f"Heart Rate: {st.session_state.results['heart_rate']} bpm")
                st.write(
                    f"Breathing Rate: {st.session_state.results['breathing_rate']} rpm"
                )
                st.write(f"HRV: {st.session_state.results['hrv']} ms")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📊 Real-Time Health Metrics")

        # Display video feed and monitoring controls
        if st.session_state.monitoring_active:
            if is_local:
                # Local webcam method
                video_placeholder = st.empty()

                # Initialize webcam
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)

                    # Process frames
                    while (
                        st.session_state.monitoring_active
                        and st.session_state.frame_count < 900
                    ):
                        ret, frame = cap.read()
                        if ret:
                            processed_frame = monitor.process_frame(frame)

                            # Display every 5th frame to reduce load
                            if st.session_state.frame_count % 5 == 0:
                                frame_rgb = cv2.cvtColor(
                                    processed_frame, cv2.COLOR_BGR2RGB
                                )
                                video_placeholder.image(
                                    frame_rgb,
                                    caption=f"Live Analysis - Frame {st.session_state.frame_count}/900",
                                    use_container_width=True,
                                )

                            time.sleep(0.03)  # Maintain ~30 FPS
                        else:
                            st.error("Failed to read from camera")
                            break

                    cap.release()

                    # Complete monitoring
                    if st.session_state.frame_count >= 900:
                        st.session_state.monitoring_active = False
                        st.session_state.session_data["end_time"] = (
                            datetime.now().isoformat()
                        )
                        st.success("✅ 30-second analysis completed!")
                        st.balloons()
                        st.rerun()

                else:
                    st.error("❌ Could not access camera")
                    st.session_state.monitoring_active = False
            else:
                # WebRTC method for deployment
                st.info("Using WebRTC for video streaming")
                webrtc_ctx = create_deployment_video_monitor()

                if webrtc_ctx.state.playing:
                    st.success(
                        "Video streaming active - Keep your face visible for 30 seconds"
                    )

                    # Show current status
                    if st.session_state.frame_count > 0:
                        progress = min(st.session_state.frame_count / 900, 1.0)
                        st.progress(
                            progress,
                            f"Analysis Progress: {st.session_state.frame_count}/900 frames",
                        )

                        if st.session_state.face_detected:
                            st.success("Face detected - collecting PPG data")
                        else:
                            st.warning("No face detected - position yourself in view")

                        # Show live metrics if available
                        if st.session_state.results["heart_rate"] > 0:
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric(
                                    "Heart Rate",
                                    f"{st.session_state.results['heart_rate']} bpm",
                                )
                            with col_b:
                                st.metric(
                                    "PPG Samples", len(st.session_state.ppg_signal)
                                )
                            with col_c:
                                st.metric(
                                    "Calculations", st.session_state.calculation_count
                                )

                    # Check if monitoring is complete
                    if (
                        st.session_state.frame_count >= 900
                        and st.session_state.monitoring_active
                    ):
                        st.session_state.monitoring_active = False
                        st.session_state.session_data["end_time"] = (
                            datetime.now().isoformat()
                        )
                        st.success("30-second analysis completed!")
                        st.balloons()
                        st.rerun()

                else:
                    st.warning("Please allow camera access to start monitoring")
                    st.info(
                        "Click the camera icon and grant permission to access your camera"
                    )

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

            live_indicator = "🔴 LIVE" if st.session_state.monitoring_active else ""

            st.markdown(
                f"""
            <div class="metric-card" style="border-left: 4px solid {color};">
                <h3>{icon} {name} <span style="color: red; font-size: 12px;">{live_indicator}</span></h3>
                <h2 style="color: {color};">{value} {unit}</h2>
                <small style="color: gray;">Status: {monitor.get_status_text(metric_key)}</small>
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

            # Create metric trend plots
            if len(st.session_state.hr_values) > 1:
                st.subheader("💓 Heart Rate Trend")
                fig, ax = plt.subplots(figsize=(10, 3))
                hr_data = list(st.session_state.hr_values)
                ax.plot(hr_data, color="red", linewidth=3, marker="o", markersize=4)
                ax.set_title("Heart Rate Over Time")
                ax.set_xlabel("Measurement")
                ax.set_ylabel("BPM")
                ax.grid(True, alpha=0.3)
                if len(hr_data) > 0:
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

            # Breathing Rate Trend
            if len(st.session_state.br_values) > 1:
                st.subheader("🫁 Breathing Rate Trend")
                fig, ax = plt.subplots(figsize=(10, 3))
                br_data = list(st.session_state.br_values)
                ax.plot(br_data, color="blue", linewidth=3, marker="s", markersize=4)
                ax.set_title("Breathing Rate Over Time")
                ax.set_xlabel("Measurement")
                ax.set_ylabel("RPM")
                ax.grid(True, alpha=0.3)
                if len(br_data) > 0:
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

            # HRV Trend
            if len(st.session_state.hrv_values) > 1:
                st.subheader("📊 HRV Trend")
                fig, ax = plt.subplots(figsize=(10, 3))
                hrv_data = list(st.session_state.hrv_values)
                ax.plot(hrv_data, color="green", linewidth=3, marker="^", markersize=4)
                ax.set_title("Heart Rate Variability Over Time")
                ax.set_xlabel("Measurement")
                ax.set_ylabel("ms")
                ax.grid(True, alpha=0.3)
                if len(hrv_data) > 0:
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

        else:
            st.info("📊 Start monitoring to see live signal analysis")
            st.markdown(
                """
            **What you'll see:**
            - 🌊 Real-time PPG signal from your face
            - 💓 Heart rate trend over time
            - 🫁 Breathing rate analysis
            - 📊 HRV measurements
            - 😰 Stress level indicators
            - 🧘 Wellness metrics
            """
            )

    # PDF Report Generation
    if not st.session_state.monitoring_active and st.session_state.session_data.get(
        "measurements"
    ):
        st.markdown("---")
        st.markdown("## 📋 Health Report Generation")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "📊 Data Points",
                len(st.session_state.session_data.get("raw_ppg_data", [])),
            )
        with col2:
            st.metric(
                "🧮 Calculations",
                len(st.session_state.session_data.get("measurements", [])),
            )
        with col3:
            st.metric("⏱️ Frames Captured", st.session_state.frame_count)

        if st.button(
            "📄 Generate Comprehensive Report", type="primary", use_container_width=True
        ):
            if PDF_AVAILABLE:
                pdf_buffer = monitor.generate_pdf_report()
                if pdf_buffer:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
            else:
                st.error(
                    "PDF generation not available. Please install reportlab: pip install reportlab"
                )

        # Display final results summary
        st.markdown("---")
        st.markdown("## 📊 Final Health Analysis Results")

        if st.session_state.results:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown("### ❤️ Heart Rate")
                st.markdown(f"**{st.session_state.results.get('heart_rate', 0)} bpm**")
                hr_status = (
                    "Normal"
                    if 60 <= st.session_state.results.get("heart_rate", 0) <= 100
                    else "Needs Attention"
                )
                st.markdown(f"*Status: {hr_status}*")

                st.markdown("### 🫁 Breathing Rate")
                st.markdown(
                    f"**{st.session_state.results.get('breathing_rate', 0)} rpm**"
                )
                br_status = (
                    "Normal"
                    if 12 <= st.session_state.results.get("breathing_rate", 0) <= 20
                    else "Needs Attention"
                )
                st.markdown(f"*Status: {br_status}*")

            with col2:
                st.markdown("### 📊 HRV")
                st.markdown(f"**{st.session_state.results.get('hrv', 0)} ms**")
                hrv_status = (
                    "Good" if st.session_state.results.get("hrv", 0) > 50 else "Fair"
                )
                st.markdown(f"*Status: {hrv_status}*")

                st.markdown("### 😰 Stress Index")
                st.markdown(
                    f"**{st.session_state.results.get('stress_index', 0):.2f}**"
                )
                stress_status = (
                    "Low"
                    if st.session_state.results.get("stress_index", 0) < 0.5
                    else "Moderate"
                )
                st.markdown(f"*Status: {stress_status}*")

            with col3:
                st.markdown("### 🧘 Parasympathetic")
                st.markdown(
                    f"**{st.session_state.results.get('parasympathetic', 0)}%**"
                )
                para_status = (
                    "Good"
                    if st.session_state.results.get("parasympathetic", 0) > 50
                    else "Fair"
                )
                st.markdown(f"*Status: {para_status}*")

                st.markdown("### 💪 Wellness Score")
                st.markdown(
                    f"**{st.session_state.results.get('wellness_score', 0)}/100**"
                )
                wellness_status = (
                    "Excellent"
                    if st.session_state.results.get("wellness_score", 0) > 70
                    else "Good"
                )
                st.markdown(f"*Status: {wellness_status}*")

            with col4:
                st.markdown("### 🩺 Blood Pressure")
                bp_sys = st.session_state.results.get("blood_pressure_sys", 120)
                bp_dia = st.session_state.results.get("blood_pressure_dia", 80)
                st.markdown(f"**{bp_sys}/{bp_dia} mmHg**")
                bp_status = "Normal" if bp_sys < 130 and bp_dia < 85 else "Elevated"
                st.markdown(f"*Status: {bp_status}*")


if __name__ == "__main__":
    main()


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
        box_shadow: 0 2px 4px rgba(0,0,0,0.1);
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

    # Initialize monitor
    if "monitor" not in st.session_state:
        st.session_state.monitor = StreamlitFaceVitalMonitor()

    monitor = st.session_state.monitor

    with st.sidebar:
        st.header("📹 Camera Controls")

        # Instructions
        st.info("🎥 Start video capture to analyze your vital signs for 30 seconds")
        st.warning("⚠️ Ensure good lighting and face visibility")

        # Camera input method selection
        camera_method = st.radio(
            "Camera Method:",
            ["Auto-detect", "Webcam (Local)", "WebRTC (Deployed)"],
            help="Auto-detect will try webcam first, then fallback to WebRTC",
        )

        # Determine camera method
        is_local = True
        if camera_method == "Auto-detect":
            try:
                test_cap = cv2.VideoCapture(0)
                if test_cap.isOpened():
                    test_cap.release()
                    is_local = True
                    st.success("✅ Webcam detected - using direct video capture")
                else:
                    is_local = False
                    st.info("📱 Using WebRTC for deployment compatibility")
            except:
                is_local = False
                st.info("📱 Using WebRTC for deployment compatibility")
        elif camera_method == "Webcam (Local)":
            is_local = True
        else:
            is_local = False

        st.session_state.is_local = is_local

        # Reset/Start button
        if not st.session_state.monitoring_active:
            if st.button(
                "🎥 Start 30s Video Analysis", type="primary", use_container_width=True
            ):
                # Reset all session state variables
                st.session_state.monitoring_active = True
                st.session_state.face_detected = False
                st.session_state.ppg_signal = deque(maxlen=900)
                st.session_state.timestamps = deque(maxlen=900)
                st.session_state.frame_count = 0
                st.session_state.calculation_count = 0

                # Reset metric values
                st.session_state.hr_values = deque(maxlen=300)
                st.session_state.br_values = deque(maxlen=300)
                st.session_state.hrv_values = deque(maxlen=300)
                st.session_state.stress_values = deque(maxlen=300)
                st.session_state.para_values = deque(maxlen=300)
                st.session_state.wellness_values = deque(maxlen=300)
                st.session_state.bp_sys_values = deque(maxlen=300)
                st.session_state.bp_dia_values = deque(maxlen=300)

                # Reset results
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

                # Reset session data
                st.session_state.session_data = {
                    "start_time": datetime.now().isoformat(),
                    "end_time": None,
                    "measurements": [],
                    "raw_ppg_data": [],
                    "timestamps_data": [],
                }

                st.rerun()

        else:
            # Show stop button during monitoring
            if st.button(
                "🛑 Stop Monitoring", type="secondary", use_container_width=True
            ):
                st.session_state.monitoring_active = False
                st.session_state.session_data["end_time"] = datetime.now().isoformat()
                st.info("Monitoring stopped manually")
                st.rerun()

        # Show monitoring status
        if st.session_state.monitoring_active:
            progress = min(st.session_state.frame_count / 900, 1.0)
            st.progress(
                progress, f"Progress: {st.session_state.frame_count}/900 frames"
            )

            if st.session_state.face_detected:
                st.success("✅ Face detected - collecting data")
            else:
                st.error("❌ No face detected")

            st.metric("PPG Samples", len(st.session_state.ppg_signal))
            st.metric("Calculations", st.session_state.calculation_count)

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📊 Real-Time Health Metrics")

        # Display video feed and monitoring controls
        if st.session_state.monitoring_active:
            if is_local:
                # Local webcam method
                video_placeholder = st.empty()

                # Initialize webcam
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)

                    # Process frames
                    while (
                        st.session_state.monitoring_active
                        and st.session_state.frame_count < 900
                    ):
                        ret, frame = cap.read()
                        if ret:
                            processed_frame = monitor.process_frame(frame)

                            # Display every 5th frame to reduce load
                            if st.session_state.frame_count % 5 == 0:
                                frame_rgb = cv2.cvtColor(
                                    processed_frame, cv2.COLOR_BGR2RGB
                                )
                                video_placeholder.image(
                                    frame_rgb,
                                    caption=f"Live Analysis - Frame {st.session_state.frame_count}/900",
                                    use_container_width=True,
                                )

                            time.sleep(0.03)  # Maintain ~30 FPS
                        else:
                            st.error("Failed to read from camera")
                            break

                    cap.release()

                    # Complete monitoring
                    if st.session_state.frame_count >= 900:
                        st.session_state.monitoring_active = False
                        st.session_state.session_data["end_time"] = (
                            datetime.now().isoformat()
                        )
                        st.success("✅ 30-second analysis completed!")
                        st.balloons()
                        st.rerun()

                else:
                    st.error("❌ Could not access camera")
                    st.session_state.monitoring_active = False
            else:
                # WebRTC method for deployment
                st.info("🌐 Using WebRTC for video streaming")
                webrtc_ctx = create_deployment_video_monitor()

                if webrtc_ctx.state.playing:
                    st.success("📹 Video streaming active")

                    # Auto-complete monitoring after 30 seconds
                    if st.session_state.frame_count >= 900:
                        st.session_state.monitoring_active = False
                        st.session_state.session_data["end_time"] = (
                            datetime.now().isoformat()
                        )
                        st.success("✅ 30-second analysis completed!")
                        st.balloons()
                        st.rerun()
                else:
                    st.warning("📹 Please allow camera access to start monitoring")

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

            live_indicator = "🔴 LIVE" if st.session_state.monitoring_active else ""

            st.markdown(
                f"""
            <div class="metric-card" style="border-left: 4px solid {color};">
                <h3>{icon} {name} <span style="color: red; font-size: 12px;">{live_indicator}</span></h3>
                <h2 style="color: {color};">{value} {unit}</h2>
                <small style="color: gray;">Status: {monitor.get_status_text(metric_key)}</small>
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

            # Create metric trend plots
            if len(st.session_state.hr_values) > 1:
                st.subheader("💓 Heart Rate Trend")
                fig, ax = plt.subplots(figsize=(10, 3))
                hr_data = list(st.session_state.hr_values)
                ax.plot(hr_data, color="red", linewidth=3, marker="o", markersize=4)
                ax.set_title("Heart Rate Over Time")
                ax.set_xlabel("Measurement")
                ax.set_ylabel("BPM")
                ax.grid(True, alpha=0.3)
                if len(hr_data) > 0:
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

        else:
            st.info("📊 Start monitoring to see live signal analysis")
            st.markdown(
                """
            **What you'll see:**
            - 🌊 Real-time PPG signal from your face
            - 💓 Heart rate trend over time
            - 🫁 Breathing rate analysis
            - 📊 HRV measurements
            - 😰 Stress level indicators
            - 🧘 Wellness metrics
            """
            )

    # PDF Report Generation
    if not st.session_state.monitoring_active and st.session_state.session_data.get(
        "measurements"
    ):
        st.markdown("---")
        st.markdown("## 📋 Health Report Generation")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "📊 Data Points",
                len(st.session_state.session_data.get("raw_ppg_data", [])),
            )
        with col2:
            st.metric(
                "🧮 Calculations",
                len(st.session_state.session_data.get("measurements", [])),
            )
        with col3:
            st.metric("⏱️ Frames Captured", st.session_state.frame_count)

        if st.button(
            "📄 Generate Comprehensive Report", type="primary", use_container_width=True
        ):
            if PDF_AVAILABLE:
                pdf_buffer = monitor.generate_pdf_report()
                if pdf_buffer:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
            else:
                st.error(
                    "PDF generation not available. Please install reportlab: pip install reportlab"
                )


if __name__ == "__main__":
    main()
