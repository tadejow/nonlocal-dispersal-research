# Klausmeier Vegetation Pipeline (pipeline_klausmeier)

This pipeline simulates the extended non-local Klausmeier model for vegetation stripe formation in semi-arid environments. It uses a non-local integral operator to model plant dispersal and competition.

## Usage

Ensure you run scripts from this directory and that your configuration JSON files are correctly set in the config/ directory.

### Running a Simulation

`powershell
..\..\.venv\Scripts\python.exe simulate.py --config config/default.json
`
Output data, logs, and generated plots will be saved to the directory specified in your configuration file.
