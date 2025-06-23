# Workspace Organization Guide

This workspace contains a large number of files which can cause VS Code to be slow when indexing. To improve performance, consider opening one of these focused sub-folders instead:

## Core Development Folders

### 1. Main Project (Recommended)
```
projectp/
```
The main project folder containing the core ML pipeline code.

### 2. Agent System
```
agent/
```
The intelligent agent system for monitoring and improvements.

### 3. Configuration & Scripts
```
configs/
scripts/
```
Configuration files and utility scripts.

### 4. Documentation
```
docs/
```
Project documentation and guides.

### 5. Tests
```
tests/
```
Test suites and validation scripts.

## How to Open a Sub-folder

1. Close this workspace
2. Use "File > Open Folder" 
3. Navigate to `/content/drive/MyDrive/Phiradon1688_co/projectp/` (or your preferred subfolder)
4. Open that specific folder as your workspace

This will significantly improve VS Code performance and reduce file enumeration time.

## Excluded Directories

The following directories are excluded from VS Code indexing to improve performance:
- Virtual environments (`venv/`, `.venv/`, `venv310/`)
- Cache files (`__pycache__/`, `.pytest_cache/`)
- Logs (`logs/`, `tmp_logs/`, `*.log`)
- ML outputs (`mlruns/`, `artifacts/`, `output/`)
- Backups (`backups/`, `.backups/`)
- Protection system cache (`protection_cache/`, `protection_logs/`)

If you need to access any of these files, you can do so through the terminal or by temporarily modifying the `.vscode/settings.json` file.
