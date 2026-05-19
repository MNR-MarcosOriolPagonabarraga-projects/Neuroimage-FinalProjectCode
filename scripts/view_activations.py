import sys
import os
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QTreeView, QPushButton, 
                             QFileDialog, QLabel, QSplitter, QSlider, QComboBox)
from PyQt6.QtGui import QFileSystemModel
from PyQt6.QtCore import QTimer, Qt

import pyvista as pv
from pyvistaqt import QtInteractor
from nilearn import datasets, surface

# Core project imports
from src.load_data import PatientLoader
from src.process import NeuroFeatureExtractor

class BrainVisualizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neurotech Lab - 3D Brain Activation Viewer")
        self.resize(1300, 850)

        self.loader = PatientLoader()
        self.extractor = NeuroFeatureExtractor(target_resolution=3, n_rois=200)
        self.selected_patient_path = None

        self.current_time_series = None
        self.current_frame = 0
        self.total_frames = 0
        self.base_speed_ms = 150 
        
        self._init_mesh_structures()
        self.init_ui()

    def _init_mesh_structures(self):
        print("Initializing cortical surface models (Both Hemispheres)...")
        fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
        
        # Load geometries for BOTH hemispheres
        self.lh_coords, self.lh_faces = surface.load_surf_mesh(fsaverage.pial_left)
        self.rh_coords, self.rh_faces = surface.load_surf_mesh(fsaverage.pial_right)
        
        # Map Schaefer atlas to Left Hemisphere
        self.lh_vertex_to_roi_map = surface.vol_to_surf(
            self.extractor.atlas.maps, 
            fsaverage.pial_left, 
            interpolation='nearest_most_frequent'
        )
        self.lh_vertex_to_roi_map = np.nan_to_num(self.lh_vertex_to_roi_map).astype(int)

        # Map Schaefer atlas to Right Hemisphere
        self.rh_vertex_to_roi_map = surface.vol_to_surf(
            self.extractor.atlas.maps, 
            fsaverage.pial_right, 
            interpolation='nearest_most_frequent'
        )
        self.rh_vertex_to_roi_map = np.nan_to_num(self.rh_vertex_to_roi_map).astype(int)

    def init_ui(self):
        main_splitter = QSplitter(self)
        self.setCentralWidget(main_splitter)

        # --- LEFT PANEL (File Management) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.btn_select_dir = QPushButton("Open Dataset Directory...")
        self.btn_select_dir.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_select_dir.clicked.connect(self.open_directory)
        left_layout.addWidget(self.btn_select_dir)

        self.dir_model = QFileSystemModel()
        self.dir_tree = QTreeView()
        self.dir_tree.setModel(self.dir_model)
        for i in range(1, 4):
            self.dir_tree.setColumnHidden(i, True)
        self.dir_tree.clicked.connect(self.on_tree_item_clicked)
        
        left_layout.addWidget(QLabel("Workspace Navigator (Click a 'sub-' folder):"))
        left_layout.addWidget(self.dir_tree)

        # Dedicated Run Button
        self.btn_run_pipeline = QPushButton("Run Preprocessing on Selected")
        self.btn_run_pipeline.setEnabled(False) 
        self.btn_run_pipeline.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        self.btn_run_pipeline.clicked.connect(self.run_pipeline)
        left_layout.addWidget(self.btn_run_pipeline)

        main_splitter.addWidget(left_panel)

        # --- RIGHT PANEL (Visualization & Media Controls) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.lbl_status = QLabel("System Status: Awaiting valid patient folder selection.")
        self.lbl_status.setStyleSheet("color: #555; font-style: italic; font-size: 12px;")
        right_layout.addWidget(self.lbl_status)

        # 3D Canvas
        self.plotter_widget = QtInteractor(self)
        right_layout.addWidget(self.plotter_widget, stretch=1)
        
        # Convert raw faces to VTK format for both hemispheres
        lh_vtk_faces = np.hstack([np.c_[np.full(len(self.lh_faces), 3), self.lh_faces]])
        rh_vtk_faces = np.hstack([np.c_[np.full(len(self.rh_faces), 3), self.rh_faces]])
        
        self.lh_polydata = pv.PolyData(self.lh_coords, lh_vtk_faces)
        self.rh_polydata = pv.PolyData(self.rh_coords, rh_vtk_faces)
        
        self.lh_polydata.point_data['activation'] = np.zeros(len(self.lh_coords))
        self.rh_polydata.point_data['activation'] = np.zeros(len(self.rh_coords))
        
        # Add Left Hemisphere (with the color bar)
        self.plotter_widget.add_mesh(
            self.lh_polydata, 
            cmap="coolwarm", 
            clim=[-2, 2], 
            show_scalar_bar=True,
            scalar_bar_args={'title': 'Z-Score BOLD Amplitude'}
        )
        
        # Add Right Hemisphere (hide the duplicate color bar)
        self.plotter_widget.add_mesh(
            self.rh_polydata, 
            cmap="coolwarm", 
            clim=[-2, 2], 
            show_scalar_bar=False
        )
        
        # Change view from lateral to an angled top-down isometric view to see both halves
        self.plotter_widget.view_isometric()
        self.plotter_widget.background_color = "white"

        # Playback Controls Layout
        control_layout = QHBoxLayout()
        
        self.btn_play_pause = QPushButton("▶ Play")
        self.btn_play_pause.setEnabled(False)
        self.btn_play_pause.clicked.connect(self.toggle_playback)
        control_layout.addWidget(self.btn_play_pause)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.valueChanged.connect(self.scrub_timeline)
        control_layout.addWidget(self.timeline_slider)

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x (Slow)", "1.0x (Normal)", "2.0x (Fast)", "Max Speed"])
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.setEnabled(False)
        self.speed_combo.currentIndexChanged.connect(self.update_playback_speed)
        control_layout.addWidget(self.speed_combo)

        right_layout.addLayout(control_layout)

        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.advance_frame)

        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([350, 950])

    # --- DIRECTORY & SELECTION LOGIC ---

    def open_directory(self):
        chosen_dir = QFileDialog.getExistingDirectory(self, "Select Root Dataset Folder")
        if chosen_dir:
            self.dir_model.setRootPath(chosen_dir)
            self.dir_tree.setRootIndex(self.dir_model.index(chosen_dir))

    def on_tree_item_clicked(self, index):
        selected_path = self.dir_model.filePath(index)
        folder_name = os.path.basename(os.path.normpath(selected_path))
        
        # ONLY enable the run button if the user clicked exactly on a folder starting with "sub-"
        if os.path.isdir(selected_path) and folder_name.startswith("sub-"):
            self.selected_patient_path = selected_path
            self.btn_run_pipeline.setEnabled(True)
            self.lbl_status.setText(f"Selected Patient: {folder_name}. Ready for preprocessing.")
        else:
            # Lock the run button if they click on an internal folder (like 'func' or 'anat') or a file
            self.selected_patient_path = None
            self.btn_run_pipeline.setEnabled(False)
            self.lbl_status.setText("Please click on a root patient directory (e.g., 'sub-1001') to proceed.")

    # --- EXECUTION & PLAYBACK LOGIC ---

    def run_pipeline(self):
        if not self.selected_patient_path:
            return

        self.playback_timer.stop()
        self.btn_play_pause.setText("▶ Play")
        patient_id = os.path.basename(os.path.normpath(self.selected_patient_path))
        
        self.lbl_status.setText(f"Status: Loading and extracting features for {patient_id}... Please wait.")
        QApplication.processEvents()

        try:
            patient_record = self.loader.load(self.selected_patient_path)
            self.current_time_series = self.extractor.extract_time_series(patient_record)
            
            self.total_frames = self.current_time_series.shape[0]
            
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setRange(0, self.total_frames - 1)
            self.timeline_slider.setValue(0)
            self.timeline_slider.blockSignals(False)
            
            self.btn_play_pause.setEnabled(True)
            self.timeline_slider.setEnabled(True)
            self.speed_combo.setEnabled(True)
            
            self.set_frame(0)
            
        except Exception as error:
            self.lbl_status.setText(f"Pipeline Error: Failed to process {patient_id}. Details: {str(error)}")
            self.current_time_series = None

    def toggle_playback(self):
        if self.playback_timer.isActive():
            self.playback_timer.stop()
            self.btn_play_pause.setText("▶ Play")
            self.lbl_status.setText(f"Paused at frame {self.current_frame}/{self.total_frames - 1}")
        else:
            # If we reached the end, loop back to the start
            if self.current_frame >= self.total_frames - 1:
                self.set_frame(0)
                
            # FIX: Explicitly fetch the interval and START the timer
            speeds = [self.base_speed_ms * 2, self.base_speed_ms, int(self.base_speed_ms / 2), 20]
            interval = speeds[self.speed_combo.currentIndex()]
            self.playback_timer.start(interval)
            
            self.btn_play_pause.setText("⏸ Pause")

    def update_playback_speed(self):
        speeds = [self.base_speed_ms * 2, self.base_speed_ms, int(self.base_speed_ms / 2), 20]
        interval = speeds[self.speed_combo.currentIndex()]
        # Only update the active timer if it is currently playing
        if self.playback_timer.isActive():
            self.playback_timer.start(interval)

    def advance_frame(self):
        next_frame = self.current_frame + 1
        if next_frame >= self.total_frames:
            self.toggle_playback() 
            return
            
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(next_frame)
        self.timeline_slider.blockSignals(False)
        
        self.set_frame(next_frame)

    def scrub_timeline(self, value):
        self.set_frame(value)

    def set_frame(self, frame_idx):
        if self.current_time_series is None:
            return

        self.current_frame = frame_idx
        active_frame_signals = self.current_time_series[self.current_frame, :]

        # --- Update Left Hemisphere ---
        lh_mesh_scalars = np.zeros(len(self.lh_coords))
        lh_valid_mask = (self.lh_vertex_to_roi_map > 0) & (self.lh_vertex_to_roi_map <= len(active_frame_signals))
        lh_mesh_scalars[lh_valid_mask] = active_frame_signals[self.lh_vertex_to_roi_map[lh_valid_mask] - 1]
        self.lh_polydata.point_data['activation'] = lh_mesh_scalars

        # --- Update Right Hemisphere ---
        rh_mesh_scalars = np.zeros(len(self.rh_coords))
        rh_valid_mask = (self.rh_vertex_to_roi_map > 0) & (self.rh_vertex_to_roi_map <= len(active_frame_signals))
        rh_mesh_scalars[rh_valid_mask] = active_frame_signals[self.rh_vertex_to_roi_map[rh_valid_mask] - 1]
        self.rh_polydata.point_data['activation'] = rh_mesh_scalars

        # Redraw the 3D scene
        self.plotter_widget.update()
        
        if self.playback_timer.isActive():
            self.lbl_status.setText(f"Playing... {self.current_frame}/{self.total_frames - 1}")
        else:
            self.lbl_status.setText(f"Scrubbing frame {self.current_frame}/{self.total_frames - 1}")

    def closeEvent(self, event):
        self.playback_timer.stop()
        self.plotter_widget.clear()
        self.plotter_widget.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BrainVisualizerApp()
    window.show()
    sys.exit(app.exec())