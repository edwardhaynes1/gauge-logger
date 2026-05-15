#!/usr/bin/env python3
"""
Correct upstream pressure gauge readings for shunt resistance and 
atmospheric reference errors.

Calibration errors:
- Shunt resistance: assumed 100.0 Ω, actually 98.6 Ω
- Atmospheric reference: assumed 1.00 bar, actually 0.953 bar (20.03.2026)

Correction formula:
P_corrected = ((((P_old - 0.13) / 0.625) + 4) * 1.0142 - 4) * 0.625 + 0.036
"""

import os
import glob
import pandas as pd


def list_csv_files():
    """List all CSV files in current directory."""
    csv_files = glob.glob("*.csv")
    if not csv_files:
        print("No CSV files found in current directory.")
        return None
    
    print("\nAvailable CSV files:")
    for i, filename in enumerate(csv_files, 1):
        print(f"{i}. {filename}")
    
    return csv_files


def correct_pressure(p_old):
    """
    Apply calibration correction to pressure reading.
    
    Parameters:
    -----------
    p_old : float or array
        Original logged pressure in bar
    
    Returns:
    --------
    float or array
        Corrected pressure in bar
    """
    # Correction formula accounting for shunt (98.6 Ω) and atmospheric (0.953 bar) errors
    p_corrected = ((((p_old - 0.13) / 0.625) + 4) * 1.0142 - 4) * 0.625 + 0.036
    return p_corrected


def main():
    print("=" * 60)
    print("Gauge Log Pressure Correction Tool")
    print("=" * 60)
    
    # List available CSV files
    csv_files = list_csv_files()
    if not csv_files:
        return
    
    # Get user selection
    while True:
        try:
            selection = input(f"\nSelect file number (1-{len(csv_files)}) or 'q' to quit: ").strip()
            
            if selection.lower() == 'q':
                print("Exiting.")
                return
            
            file_index = int(selection) - 1
            if 0 <= file_index < len(csv_files):
                selected_file = csv_files[file_index]
                break
            else:
                print(f"Please enter a number between 1 and {len(csv_files)}.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")
    
    print(f"\nProcessing: {selected_file}")
    
    # Read CSV file
    try:
        df = pd.read_csv(selected_file)
        print(f"Loaded {len(df)} rows")
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Identify pressure column
    pressure_columns = [col for col in df.columns if 'pressure' in col.lower() or 'bar' in col.lower()]
    
    if not pressure_columns:
        print("\nCould not automatically identify pressure column.")
        print("Available columns:", df.columns.tolist())
        pressure_col = input("Enter pressure column name: ").strip()
    else:
        pressure_col = pressure_columns[0]
        print(f"Using pressure column: '{pressure_col}'")
    
    if pressure_col not in df.columns:
        print(f"Error: Column '{pressure_col}' not found in file.")
        return
    
    # Apply correction
    print("\nApplying calibration correction...")
    df_corrected = df.copy()
    df_corrected[pressure_col] = correct_pressure(df[pressure_col])
    
    # Show summary statistics
    print("\nPressure correction summary:")
    print(f"  Original:  min={df[pressure_col].min():.4f} bar, max={df[pressure_col].max():.4f} bar")
    print(f"  Corrected: min={df_corrected[pressure_col].min():.4f} bar, max={df_corrected[pressure_col].max():.4f} bar")
    print(f"  Mean shift: {(df_corrected[pressure_col] - df[pressure_col]).mean():.4f} bar")
    
    # Generate output filename
    base_name = os.path.splitext(selected_file)[0]
    output_file = f"{base_name}_CORRECTED.csv"
    
    # Check if output file already exists
    if os.path.exists(output_file):
        overwrite = input(f"\n{output_file} already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Correction cancelled.")
            return
    
    # Save corrected data
    try:
        df_corrected.to_csv(output_file, index=False)
        print(f"\n✓ Corrected data saved to: {output_file}")
        print(f"✓ Original file unchanged: {selected_file}")
    except Exception as e:
        print(f"Error saving file: {e}")
        return


if __name__ == "__main__":
    main()
