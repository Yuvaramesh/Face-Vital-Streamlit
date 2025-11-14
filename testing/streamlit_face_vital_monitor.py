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
import threading
import queue
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def get_webrtc_config():
    """Get enhanced WebRTC configuration with multiple STUN/TURN servers"""
    return {
        "iceServers": [
            # Google STUN servers
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
            # Additional public STUN servers for better connectivity
            {"urls": ["stun:stun.stunprotocol.org:3478"]},
            {"urls": ["stun:stun.voiparound.com"]},
            {"urls": ["stun:stun.voipbuster.com"]},
            {"urls": ["stun:stun.voipstunt.com"]},
            {"urls": ["stun:stun.voxgratia.org"]},
            # OpenRelay STUN servers
            {"urls": ["stun:openrelay.metered.ca:80"]},
            # Twilio STUN servers (public)
            {"urls": ["stun:global.stun.twilio.com:3478"]},
            # Free TURN servers (you may want to replace with your own)
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": ["turn:openrelay.metered.ca:443"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ],
        "iceCandidatePoolSize": 10,
        "bundlePolicy": "max-bundle",
        "rtcpMuxPolicy": "require",
    }


def get_media_constraints():
    """Get media constraints with fallback options for better compatibility"""
    return {
        "video": {
            "width": {"min": 320, "ideal": 640, "max": 1280},
            "height": {"min": 240, "ideal": 480, "max": 720},
            "frameRate": {"min": 15, "ideal": 30, "max": 30},
            "facingMode": "user",
            "aspectRatio": 1.333,
        },
        "audio": False,
    }


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

        if "video_processor" not in st.session_state:
            st.session_state.video_processor = None

        if "webrtc_connection_state" not in st.session_state:
            st.session_state.webrtc_connection_state = "disconnected"
        if "connection_attempts" not in st.session_state:
            st.session_state.connection_attempts = 0

    def extract_ppg_signal(self, frame, landmarks):
        """Extract PPG signal from facial landmarks"""
        try:
            h, w, _ = frame.shape

            # Define ROI points for forehead and cheeks
            forehead_points = [10, 151, 9, 8]
            left_cheek_points = [
                116,
                117,
                118,
                119,
                120,
                121,
                126,
                142,
                36,
                205,
                206,
                207,
                213,
                192,
                147,
                187,
                207,
                213,
                192,
                147,
            ]
            right_cheek_points = [
                345,
                346,
                347,
                348,
                349,
                350,
                451,
                452,
                453,
                464,
                435,
                410,
                454,
                323,
                361,
                340,
                346,
                347,
                348,
                349,
            ]

            roi_values = []

            # Extract values from all ROI points
            for points in [forehead_points, left_cheek_points, right_cheek_points]:
                for point_idx in points:
                    if point_idx < len(landmarks.landmark):
                        x = int(landmarks.landmark[point_idx].x * w)
                        y = int(landmarks.landmark[point_idx].y * h)

                        if 0 <= x < w and 0 <= y < h:
                            # Extract green channel value (most sensitive to blood volume changes)
                            roi_values.append(frame[y, x, 1])

            if roi_values:
                ppg_value = np.mean(roi_values)
                return ppg_value
            else:
                return 0

        except Exception as e:
            logger.error(f"Error extracting PPG signal: {e}")
            return 0

    def calculate_heart_rate(self, signal_data, timestamps, fps=30):
        """Calculate heart rate from PPG signal"""
        try:
            if len(signal_data) < fps * 10:  # Need at least 10 seconds
                return 0

            # Convert to numpy array and apply filtering
            signal_array = np.array(signal_data)

            # Apply bandpass filter for heart rate (0.8-3.5 Hz = 48-210 BPM)
            nyquist = fps / 2
            low = 0.8 / nyquist
            high = 3.5 / nyquist
            b, a = signal.butter(4, [low, high], btype="band")
            filtered_signal = signal.filtfilt(b, a, signal_array)

            # Apply FFT
            fft_result = np.abs(fft(filtered_signal))
            freqs = np.fft.fftfreq(len(filtered_signal), 1 / fps)

            # Find peak in valid frequency range
            valid_idx = (freqs >= 0.8) & (freqs <= 3.5)
            if np.any(valid_idx):
                valid_freqs = freqs[valid_idx]
                valid_fft = fft_result[valid_idx]
                peak_idx = np.argmax(valid_fft)
                heart_rate_hz = valid_freqs[peak_idx]
                heart_rate_bpm = heart_rate_hz * 60
                return max(50, min(200, heart_rate_bpm))

        except Exception as e:
            logger.error(f"Error calculating heart rate: {e}")

        return 0

    def calculate_breathing_rate(self, signal_data, fps=30):
        """Calculate breathing rate from PPG signal variations"""
        try:
            if len(signal_data) < fps * 15:  # Need at least 15 seconds
                return 0

            signal_array = np.array(signal_data)

            # Apply low-pass filter for breathing rate (0.1-0.6 Hz = 6-36 BPM)
            nyquist = fps / 2
            high = 0.6 / nyquist
            b, a = signal.butter(4, high, btype="low")
            filtered_signal = signal.filtfilt(b, a, signal_array)

            # Calculate breathing rate using autocorrelation
            breathing_rate = self._estimate_breathing_from_envelope(
                filtered_signal, fps
            )
            return max(8, min(35, breathing_rate))

        except Exception as e:
            logger.error(f"Error calculating breathing rate: {e}")
            return 0

    def estimate_blood_pressure(self, heart_rate, hrv, stress_index):
        """Estimate blood pressure using correlation with HR, HRV, and stress"""
        try:
            base_sys = 120
            base_dia = 80

            # Adjust based on heart rate
            hr_factor = (heart_rate - 70) * 0.5

            # Adjust based on HRV (lower HRV = higher BP)
            hrv_factor = (50 - hrv) * 0.3

            # Adjust based on stress
            stress_factor = stress_index * 0.4

            sys_bp = base_sys + hr_factor + hrv_factor + stress_factor
            dia_bp = (
                base_dia
                + (hr_factor * 0.6)
                + (hrv_factor * 0.5)
                + (stress_factor * 0.3)
            )

            # Clamp to reasonable ranges
            sys_bp = max(90, min(180, sys_bp))
            dia_bp = max(60, min(120, dia_bp))

            return int(sys_bp), int(dia_bp)

        except Exception as e:
            logger.error(f"Error estimating blood pressure: {e}")
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


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.monitor = StreamlitFaceVitalMonitor()
        self.frame_count = 0
        self.last_calculation_time = time.time()

    def recv(self, frame):
        try:
            if st.session_state.webrtc_connection_state != "connected":
                st.session_state.webrtc_connection_state = "connected"
                st.session_state.connection_attempts = 0
                logger.info("WebRTC connection established successfully")

            img = frame.to_ndarray(format="bgr24")

            # Process frame for face detection and vital signs
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.monitor.face_mesh.process(rgb_img)

            current_time = time.time()

            if results.multi_face_landmarks:
                st.session_state.face_detected = True

                for face_landmarks in results.multi_face_landmarks:
                    # Draw face mesh
                    self.monitor.mp_drawing.draw_landmarks(
                        img,
                        face_landmarks,
                        self.monitor.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.monitor.mp_drawing.DrawingSpec(
                            color=(0, 255, 0), thickness=1, circle_radius=1
                        ),
                    )

                    # Extract PPG signal
                    if st.session_state.monitoring_active:
                        ppg_value = self.monitor.extract_ppg_signal(img, face_landmarks)
                        st.session_state.ppg_signal.append(ppg_value)
                        st.session_state.timestamps.append(current_time)

                        # Calculate vitals every 2 seconds
                        if current_time - self.last_calculation_time >= 2.0:
                            self._calculate_vitals()
                            self.last_calculation_time = current_time
            else:
                st.session_state.face_detected = False

            self.frame_count += 1
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        except Exception as e:
            logger.error(f"Error in video processing: {e}")
            st.session_state.webrtc_connection_state = "error"
            return frame

    def _calculate_vitals(self):
        """Calculate all vital signs"""
        try:
            if len(st.session_state.ppg_signal) > 300:  # At least 10 seconds of data
                signal_data = list(st.session_state.ppg_signal)
                timestamps = list(st.session_state.timestamps)

                # Calculate heart rate
                hr = self.monitor.calculate_heart_rate(signal_data, timestamps)
                st.session_state.results["heart_rate"] = hr
                st.session_state.hr_values.append(hr)

                # Calculate breathing rate
                br = self.monitor.calculate_breathing_rate(signal_data)
                st.session_state.results["breathing_rate"] = br
                st.session_state.br_values.append(br)

                # Calculate HRV and other metrics
                hrv = self.monitor.calculate_hrv(signal_data, timestamps)
                st.session_state.results["hrv"] = hrv
                st.session_state.hrv_values.append(hrv)

                stress = self.monitor.calculate_stress_index(hrv, hr)
                st.session_state.results["stress_index"] = stress
                st.session_state.stress_values.append(stress)

                para = self.monitor.calculate_parasympathetic_activity(hrv, hr)
                st.session_state.results["parasympathetic"] = para
                st.session_state.para_values.append(para)

                wellness = self.monitor.calculate_wellness_score(hr, hrv, stress, br)
                st.session_state.results["wellness_score"] = wellness
                st.session_state.wellness_values.append(wellness)

                # Estimate blood pressure
                sys_bp, dia_bp = self.monitor.estimate_blood_pressure(hr, hrv, stress)
                st.session_state.results["blood_pressure_sys"] = sys_bp
                st.session_state.results["blood_pressure_dia"] = dia_bp
                st.session_state.bp_sys_values.append(sys_bp)
                st.session_state.bp_dia_values.append(dia_bp)

                st.session_state.calculation_count += 1

        except Exception as e:
            logger.error(f"Error calculating vitals: {e}")


def main():
    st.set_page_config(
        page_title="Face Vital Monitor - Comprehensive Health Analysis",
        page_icon="🫀",
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
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .status-connected {
        color: #28a745;
        font-weight: bold;
    }
    .status-connecting {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .connection-help {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-header"><h1>🫀 Face Vital Monitor</h1><p>Advanced Contactless Health Monitoring System</p></div>',
        unsafe_allow_html=True,
    )

    monitor = StreamlitFaceVitalMonitor()

    # Sidebar controls
    with st.sidebar:
        st.header("📹 Camera Controls")

        connection_state = st.session_state.get(
            "webrtc_connection_state", "disconnected"
        )
        if connection_state == "connected":
            st.markdown(
                '<p class="status-connected">🟢 Camera Connected</p>',
                unsafe_allow_html=True,
            )
        elif connection_state == "connecting":
            st.markdown(
                '<p class="status-connecting">🟡 Connecting...</p>',
                unsafe_allow_html=True,
            )
        elif connection_state == "error":
            st.markdown(
                '<p class="status-error">🔴 Connection Error</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p class="status-error">🔴 Camera Disconnected</p>',
                unsafe_allow_html=True,
            )

        camera_enabled = st.checkbox("Enable Camera", value=True)

        if (
            connection_state in ["error", "disconnected"]
            or st.session_state.get("connection_attempts", 0) > 0
        ):
            with st.expander("🔧 Connection Troubleshooting", expanded=True):
                st.markdown(
                    """
                <div class="connection-help">
                <h4>If camera connection fails:</h4>
                <ol>
                <li><strong>Check browser permissions:</strong> Allow camera access when prompted</li>
                <li><strong>Try different browser:</strong> Chrome/Edge work best for WebRTC</li>
                <li><strong>Check network:</strong> Corporate firewalls may block WebRTC</li>
                <li><strong>Refresh page:</strong> Sometimes helps reset the connection</li>
                <li><strong>Close other apps:</strong> That might be using your camera</li>
                </ol>
                <p><strong>Technical:</strong> Using enhanced STUN/TURN servers for better connectivity</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔴 Start Scan", disabled=st.session_state.monitoring_active):
                st.session_state.monitoring_active = True
                st.session_state.session_data["start_time"] = datetime.now()
                st.rerun()

        with col2:
            if st.button(
                "⏹️ Stop Scan", disabled=not st.session_state.monitoring_active
            ):
                st.session_state.monitoring_active = False
                st.session_state.session_data["end_time"] = datetime.now()
                st.rerun()

        if st.session_state.monitoring_active:
            st.success("⏱️ Ready to start monitoring")
        else:
            st.info("📱 Click 'Start Scan' to begin")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📷 Camera Feed")

        if camera_enabled:
            if "video_processor" not in st.session_state:
                st.session_state.video_processor = VideoProcessor

            try:
                st.session_state.webrtc_connection_state = "connecting"
                st.session_state.connection_attempts += 1

                webrtc_ctx = webrtc_streamer(
                    key="face-vital-monitor",
                    mode=WebRtcMode.SENDRECV,
                    video_processor_factory=st.session_state.video_processor,
                    media_stream_constraints=get_media_constraints(),
                    rtc_configuration=get_webrtc_config(),
                    async_processing=True,
                    video_html_attrs={
                        "style": {
                            "width": "100%",
                            "margin": "0 auto",
                            "border-radius": "10px",
                        },
                        "controls": False,
                        "autoplay": True,
                        "muted": True,
                    },
                )

                if webrtc_ctx.state.playing:
                    st.session_state.webrtc_connection_state = "connected"
                elif webrtc_ctx.state.signalling:
                    st.session_state.webrtc_connection_state = "connecting"
                else:
                    st.session_state.webrtc_connection_state = "disconnected"

            except Exception as e:
                logger.error(f"WebRTC connection error: {e}")
                st.session_state.webrtc_connection_state = "error"
                st.error(f"Camera connection failed: {str(e)}")
                st.info("Please check the troubleshooting guide in the sidebar.")

            # Status indicator
            status = (
                "🟢 Face Detected"
                if st.session_state.face_detected
                else "🔴 No Face Detected"
            )
            st.markdown(f"**Status:** {status}")

            if st.session_state.get("connection_attempts", 0) > 3:
                st.warning(
                    "⚠️ Multiple connection attempts detected. Try refreshing the page or check your network connection."
                )

        else:
            st.info("📷 Enable camera to start monitoring")

    with col2:
        st.subheader("📊 Health Metrics")

        # Display current results
        results = st.session_state.results

        # Heart Rate
        st.metric(
            label="💓 Heart Rate", value=f"{results['heart_rate']:.0f} bpm", delta=None
        )

        # Breathing Rate
        st.metric(
            label="🫁 Breathing Rate",
            value=f"{results['breathing_rate']:.0f} rpm",
            delta=None,
        )

        # Blood Pressure
        st.metric(
            label="🩸 Blood Pressure",
            value=f"{results['blood_pressure_sys']:.0f}/{results['blood_pressure_dia']:.0f} mmHg",
            delta=None,
        )

        # HRV
        st.metric(label="📈 HRV", value=f"{results['hrv']:.0f} ms", delta=None)

    # Data Visualization Section
    if len(st.session_state.hr_values) > 1:
        st.header("📈 Data Visualization")

        tab1, tab2, tab3 = st.tabs(
            ["📊 Metric Trends", "🌊 Raw Signals", "📋 Session Data"]
        )

        with tab1:
            # Individual metric plots
            metrics_to_plot = [
                ("Heart Rate", st.session_state.hr_values, "bpm", "red"),
                ("Breathing Rate", st.session_state.br_values, "rpm", "blue"),
                ("HRV", st.session_state.hrv_values, "ms", "green"),
                ("Stress Index", st.session_state.stress_values, "", "orange"),
                ("Parasympathetic", st.session_state.para_values, "%", "purple"),
                ("Wellness Score", st.session_state.wellness_values, "/100", "teal"),
            ]

            for i in range(0, len(metrics_to_plot), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(metrics_to_plot):
                        name, data_deque, unit, color = metrics_to_plot[i + j]

                        if len(data_deque) > 1:
                            with col:
                                fig, ax = plt.subplots(figsize=(6, 4))
                                time_axis = np.linspace(
                                    -len(data_deque), 0, len(data_deque)
                                )
                                ax.plot(
                                    time_axis,
                                    list(data_deque),
                                    color=color,
                                    linewidth=2,
                                )
                                ax.set_title(f"{name} Trend")
                                ax.set_xlabel("Time (s)")
                                ax.set_ylabel(f"{name} {unit}")
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                                plt.close(fig)

            # Blood pressure plot
            if len(st.session_state.bp_sys_values) > 1:
                st.subheader("Blood Pressure Trend")
                fig, ax = plt.subplots(figsize=(12, 4))
                time_axis = np.linspace(
                    -len(st.session_state.bp_sys_values),
                    0,
                    len(st.session_state.bp_sys_values),
                )
                ax.plot(
                    time_axis,
                    list(st.session_state.bp_sys_values),
                    "darkred",
                    linewidth=2,
                    label="Systolic",
                )
                ax.plot(
                    time_axis,
                    list(st.session_state.bp_dia_values),
                    "maroon",
                    linewidth=2,
                    label="Diastolic",
                )
                ax.set_title("Blood Pressure Trend")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Blood Pressure (mmHg)")
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

        with tab2:
            if len(st.session_state.ppg_signal) > 10:
                st.subheader("Raw PPG Signal")
                fig, ax = plt.subplots(figsize=(12, 6))
                time_axis = np.linspace(
                    -len(st.session_state.ppg_signal) / 30,
                    0,
                    len(st.session_state.ppg_signal),
                )
                ax.plot(time_axis, list(st.session_state.ppg_signal), "b-", linewidth=1)
                ax.set_title("Raw PPG Signal")
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("PPG Amplitude")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)

        with tab3:
            if st.session_state.session_data["measurements"]:
                st.subheader("Session Measurements")
                measurements_df = []
                for measurement in st.session_state.session_data["measurements"]:
                    measurements_df.append(
                        {
                            "Timestamp": measurement["timestamp"][:19].replace(
                                "T", " "
                            ),
                            "Heart Rate (bpm)": measurement["heart_rate"],
                            "Breathing Rate (rpm)": measurement["breathing_rate"],
                            "Systolic BP (mmHg)": measurement["blood_pressure_sys"],
                            "Diastolic BP (mmHg)": measurement["blood_pressure_dia"],
                            "HRV (ms)": measurement["hrv"],
                            "Stress Index": measurement["stress_index"],
                            "Parasympathetic (%)": measurement["parasympathetic"],
                            "Wellness Score": measurement["wellness_score"],
                        }
                    )

                import pandas as pd

                df = pd.DataFrame(measurements_df)
                st.dataframe(df, use_container_width=True)

    # Auto-refresh for real-time updates when monitoring is active
    if st.session_state.monitoring_active:
        time.sleep(0.1)
        st.rerun()


if __name__ == "__main__":
    main()
