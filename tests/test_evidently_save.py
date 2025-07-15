# File: test_evidently_save.py
# A minimal script to test the core functionality of the Evidently library.
from sklearn import datasets
import sys

# --- Print Library Version for Debugging ---
try:
    import evidently
    print(f"Evidently library version found: {evidently.__version__}")
except ImportError:
    print("ERROR: Evidently library is not installed.")
    sys.exit(1)

# --- Correct, Modern Imports for Evidently ---
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# --- Create Simple Dummy Data ---
# This uses the classic Iris dataset, which is small and simple.
print("Loading dummy data...")
iris_data = datasets.load_iris(as_frame=True)
iris_frame = iris_data.frame
print("Dummy data loaded.")

# --- Create and Run the Report ---
print("Creating Data Drift Report object...")
data_drift_report = Report(metrics=[DataDriftPreset()])

print("Running calculations...")
data_drift_report.run(reference_data=iris_frame.iloc[:60], current_data=iris_frame.iloc[60:])
print("Report calculations complete.")

# --- Attempt to Save the Report ---
# This is the critical test.
try:
    report_filename = "test_report.html"
    print(f"Attempting to save report to '{report_filename}'...")
    data_drift_report.save_html(report_filename)
    print(f"SUCCESS: Report saved successfully to '{report_filename}'!")
except AttributeError:
    print("\n--- TEST FAILED ---")
    print("ERROR: The 'save_html' attribute does not exist on the Report object.")
    print("This confirms a library installation or version conflict issue.")
    print("-------------------\n")
except Exception as e:
    print("\n--- TEST FAILED ---")
    print(f"An unexpected error occurred during save: {e}")
    print("-------------------\n")
