import argparse
import json
import os
import sys

try:
	from app.gui import MainWindow
except Exception:
	# Fallback when executed directly as a script (no package context)
	from app.gui import MainWindow


def resolve_config_path(config_path: str) -> str:
	# If absolute and exists
	if os.path.isabs(config_path) and os.path.exists(config_path):
		return config_path
	# Try CWD
	cwd_path = os.path.abspath(config_path)
	if os.path.exists(cwd_path):
		return cwd_path
	# Try project root (one level above this file's directory)
	script_dir = os.path.dirname(os.path.abspath(__file__))
	project_root = os.path.dirname(script_dir)
	root_path = os.path.join(project_root, config_path)
	if os.path.exists(root_path):
		return root_path
	# Try alongside script
	script_path = os.path.join(script_dir, config_path)
	if os.path.exists(script_path):
		return script_path
	# Not found
	return cwd_path


def load_config(config_path: str) -> dict:
	path = resolve_config_path(config_path)
	if not os.path.exists(path):
		raise FileNotFoundError(f"Config file not found: {config_path} (resolved: {path})")
	with open(path, 'r', encoding='utf-8') as f:
		return json.load(f)


def main() -> None:
	parser = argparse.ArgumentParser(description="Jetson Smart Surveillance GUI")
	parser.add_argument('--config', type=str, default='config.json', help='Path to config.json')
	args = parser.parse_args()

	config = load_config(args.config)

	from PySide6.QtWidgets import QApplication  # delayed import to avoid GUI deps on import
	app = QApplication(sys.argv)
	window = MainWindow(config)
	window.show()
	sys.exit(app.exec())


if __name__ == '__main__':
	main()

