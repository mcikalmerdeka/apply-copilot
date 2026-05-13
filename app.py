"""
Main application entry point for ApplyCopilot.
Run this file to start the Gradio interface.
"""

import gradio as gr
from src.ui.gradio_interface import ApplyCopilotUI

ui = ApplyCopilotUI()
demo = ui.create_interface()

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), server_name="0.0.0.0", server_port=7860)