# RSSSelector / CANOE Interface

A **desktop application** for managing and processing CANOE energy model databases and configurations.  
The app provides a simple graphical interface (built with [Flet](https://flet.dev)) to select model inputs, manage SQL schemas, and process datasets for the CANOE energy systems model.

---

## Features

- **Cross-platform UI:** built with Flet (runs as a Python script or compiled Windows executable).
- **Automatic logging:** every session writes a timestamped log file for debugging and tracking.  
- **Platform-aware directories:** logs, config, and assets are stored in appropriate locations for Python and compiled builds.  
- **Database schema support:** automatically loads a bundled `schema.sql` file used for SQLite setup.  
- **Config persistence:** saves UI options and state to a JSON configuration file.  

---

## Project Structure

```
.
├── main.py                # Entry point for the Flet application
├── database_processing.py # Database processing utilities
├── directories.py         # Manages platform-specific paths
├── log_setup.py           # Global logging configuration
├── constants.py           # Shared enums and constants
├── requirements.txt       # Python dependencies
├── assets/
│   └── schema.sql         # SQL schema used by the database
├── logs/                  # Log output (created automatically)
├── input/
│   └── config.json        # Saved UI configuration
```

---

## Installation (Python environment)

### Create and activate a virtual environment

```bash
# Create an empty environment (no dependencies yet)
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the app in Python

Once dependencies are installed, launch the app with:

```bash
python main.py
```

Logs will be written to:
```
./logs/
```

Configuration files will be created in:
```
./input/config.json
```

---

## Using the App

When the window opens:
- **Select model directories or inputs** — depending on the UI options available in your version of the interface.
- **Select region-sector-scenario configurations** — determines the resolution of the output CANOE model.
- **Click submit to process** — The input dataset will be processed into the output database
- **View logs** in the terminal or in the `logs/` folder.

If you’re running a **compiled executable**, logs and config are stored in your user directories (via `platformdirs`):

```
C:\Users\<you>\AppData\Local\CANOE\RSSSelector\Logs
C:\Users\<you>\AppData\Local\CANOE\RSSSelector\Config
```

---

## Compiling to an Executable

You can build a standalone Windows executable using **PyInstaller**.

### Install PyInstaller (inside your venv)

```bash
pip install pyinstaller
```

### Build the executable

```bash
pyinstaller CANOE.spec
```

### Run the executable

Once built, your app will appear in the `dist/` folder:

```
dist/
└── main.exe
```

When you run the EXE:
- `platformdirs` ensures logs and configs are written under your user’s app data folder.  
- Bundled resources (like `schema.sql`) are available through `directories.get_resource_path()`.

---

## Logging

Every run of the app generates a timestamped rotating log file, such as:

```
logs/canoe_app_2025-11-10_15-30-42.log
```

Logs include timestamps, log levels, and message origins. When running as a compiled EXE, logs are written under:

```
%LOCALAPPDATA%\CANOE\RSSSelector\Logs
```

---

## Constants and Configuration

Common definitions for regions, sectors, and model parameters are located in [`constants.py`](constants.py).  
This includes:
- Enumerations for model regions, sectors, and variants.
- UI scenario defaults.
- Table lists and default parameter values for CANOE datasets.
