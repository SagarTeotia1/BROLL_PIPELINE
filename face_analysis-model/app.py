"""GUI entry point.

    python app.py                       # open the window
    python app.py --video clip.mp4      # open the window with a clip loaded
    python app.py --config my.yaml      # use a different configuration

Everything heavy (model loading, the first CUDA context) is deferred until the user
actually asks for analysis or registration, so the window appears immediately.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the project root is importable when launched from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from configs.config import AppConfig  # noqa: E402
from gui.main_window import MainWindow  # noqa: E402
from utils.gpu import get_device_info  # noqa: E402
from utils.logging_utils import get_logger, setup_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Face Recognition + Emotion Timeline (GUI)"
    )
    parser.add_argument("--config", type=str, default=None, help="path to a YAML config")
    parser.add_argument("--video", type=str, default=None, help="video to open on start")
    parser.add_argument(
        "--log-level", type=str, default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = AppConfig.load(args.config)
    if args.log_level:
        config.logging.level = args.log_level

    log_path = setup_logging(
        config.logging.level,
        config.paths.resolve(config.paths.log_dir),
        config.logging.to_file,
    )
    log = get_logger("app")
    log.info("Starting GUI | %s", get_device_info(config.gpu.device_id))
    if log_path:
        log.info("Log file: %s", log_path)

    QGuiApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("Face Emotion Timeline")

    window = MainWindow(config)
    window.show()
    if args.video:
        window.load_video(args.video)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
