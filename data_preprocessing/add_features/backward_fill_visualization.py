import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np


def create_simplified_example():
    """Create a simplified visualization with synthetic but realistic data patterns."""

    # Create realistic synthetic data for clear visualization
    base_date = datetime(2020, 1, 1)

    # Biochemical measurements (sparse, like real data) - REDUCED TO 2 MEASUREMENTS
    biochem_dates = [
        base_date + timedelta(days=20),  # Jan 21
        base_date + timedelta(days=60),  # Mar 1
    ]
    biochem_values = [7.2, 6.8]  # HbA1c values

    # Generate realistic glucose time series - REDUCED FREQUENCY AND TIME WINDOW
    glucose_dates = []
    glucose_values = []

    # Define the overall time range - REDUCED TO 3 MONTHS
    start_date = base_date
    end_date = base_date + timedelta(days=90)

    # Generate glucose measurements every 4 hours for even fewer points
    current_time = start_date
    base_glucose = 130  # Base glucose level
    daily_pattern_amplitude = 25  # Daily glucose variation
    noise_level = 12  # Random noise

    # Simulate realistic daily glucose patterns
    while current_time <= end_date:
        # Create daily pattern (higher in morning/evening, lower at night)
        hour_of_day = current_time.hour + current_time.minute / 60
        daily_factor = (
            np.sin((hour_of_day - 6) * np.pi / 12) * 0.5 + 0.5
        )  # Peak around noon

        # Add some weekly variation
        day_of_week = current_time.weekday()
        weekly_factor = 1 + 0.1 * np.sin(day_of_week * np.pi / 3.5)

        # Calculate glucose value with realistic patterns
        glucose_value = (
            base_glucose
            + daily_factor * daily_pattern_amplitude
            + np.random.normal(0, noise_level) * weekly_factor
        )

        # Ensure realistic glucose range (80-220 mg/dL)
        glucose_value = max(80, min(220, glucose_value))

        glucose_dates.append(current_time)
        glucose_values.append(glucose_value)

        # Move to next 4-hour interval
        current_time += timedelta(hours=4)

    # Create DataFrames
    df_biochem_viz = pd.DataFrame(
        {
            "Timestamp": biochem_dates,
            "HbA1c": biochem_values,
            "Patient_ID": ["P001"] * len(biochem_dates),
        }
    )

    df_glucose_viz = pd.DataFrame(
        {
            "Timestamp": glucose_dates,
            "Measurement": glucose_values,
            "Patient_ID": ["P001"] * len(glucose_dates),
        }
    )

    # Sort by timestamp
    df_glucose_viz = df_glucose_viz.sort_values("Timestamp").reset_index(drop=True)
    df_biochem_viz = df_biochem_viz.sort_values("Timestamp").reset_index(drop=True)

    # Create visualization - Single combined plot with larger size for thesis
    # fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Create secondary y-axis for HbA1c
    ax2 = ax.twinx()

    # Plot biochemical measurements with 30-day forward tolerance windows FIRST
    tolerance_windows = []
    for idx, row in df_biochem_viz.iterrows():
        # Draw the tolerance window
        start_window = row["Timestamp"]
        end_window = row["Timestamp"] + pd.Timedelta(days=30)
        tolerance_windows.append((start_window, end_window))

        # Fill the tolerance window with visible color
        ax.axvspan(
            start_window,
            end_window,
            alpha=0.5,
            color="papayawhip",
            zorder=1,
        )

        # Draw vertical line at biochemical measurement
        ax.axvline(
            x=row["Timestamp"],
            color="darkred",
            linestyle="--",
            alpha=1.0,
            linewidth=4,
            zorder=3,
        )

    # Plot HbA1c measurements on secondary axis - RED SQUARES WITHOUT ANNOTATIONS
    ax2.scatter(
        df_biochem_viz["Timestamp"],
        df_biochem_viz["HbA1c"],
        color="orangered",
        alpha=1.0,
        s=400,  # Large size
        marker="s",
        edgecolors="darkred",
        linewidth=3,
        zorder=4,
    )

    # Plot glucose time series with different colors inside/outside tolerance windows
    for i in range(len(df_glucose_viz)):
        timestamp = df_glucose_viz.iloc[i]["Timestamp"]
        measurement = df_glucose_viz.iloc[i]["Measurement"]

        # Check if this point is inside any tolerance window
        inside_window = False
        for start_window, end_window in tolerance_windows:
            if start_window <= timestamp <= end_window:
                inside_window = True
                break

        # Plot single point with appropriate color and alpha
        if inside_window:
            color = "darkblue"
            alpha = 1.0
            linewidth = 3
        else:
            color = "lightblue"
            alpha = 0.5
            linewidth = 2

        # Plot the point and connect to previous if exists
        if i > 0:
            prev_timestamp = df_glucose_viz.iloc[i - 1]["Timestamp"]
            prev_measurement = df_glucose_viz.iloc[i - 1]["Measurement"]

            # Check if previous point was inside window too
            prev_inside = False
            for start_window, end_window in tolerance_windows:
                if start_window <= prev_timestamp <= end_window:
                    prev_inside = True
                    break

            # Draw line segment with appropriate color
            if inside_window and prev_inside:
                line_color = "darkblue"
                line_alpha = 1.0
                line_width = 2.5
            elif not inside_window and not prev_inside:
                line_color = "steelblue"
                line_alpha = 0.8
                line_width = 2
            else:
                # Transition point - use darker color
                line_color = "darkblue"
                line_alpha = 0.7
                line_width = 2.5

            ax.plot(
                [prev_timestamp, timestamp],
                [prev_measurement, measurement],
                color=line_color,
                alpha=line_alpha,
                linewidth=line_width,
                zorder=2,
            )

        # Plot the marker
        ax.plot(
            timestamp,
            measurement,
            marker="o",
            markersize=3,
            color=color,
            alpha=alpha,
            zorder=2,
        )

    # Set labels and formatting - larger fonts
    ax.set_xlabel("Tempo", fontsize=18, fontweight="bold")
    ax.set_ylabel("Glucosio (mg/dL)", fontsize=18, fontweight="bold", color="darkblue")
    ax2.set_ylabel("HbA1c (%)", fontsize=18, fontweight="bold", color="darkred")

    # Set y-axis limits for better visibility
    ax.set_ylim(70, 200)
    ax2.set_ylim(6.5, 7.5)

    # Color the y-axis labels - larger fonts
    ax.tick_params(axis="y", labelsize=14, width=2)
    ax2.tick_params(axis="y", labelsize=14, width=2)
    ax.tick_params(axis="x", labelsize=14, width=2)

    # Grid
    ax.grid(True, alpha=0.3, linewidth=1.5)

    # Format x-axis - SHOW DATES
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=14)

    # Add border around the plot
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    for spine in ax2.spines.values():
        spine.set_linewidth(2)

    plt.tight_layout()

    # Save the plot with high quality for thesis
    output_path = "plots/backward_fill_thesis_visualization.png"
    plt.savefig(
        output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="black"
    )
    print(f"\n✅ Visualizzazione per tesi salvata come: {output_path}")

    # Calculate statistics
    merged_data = pd.merge_asof(
        df_glucose_viz,
        df_biochem_viz[["Timestamp", "HbA1c"]],
        on="Timestamp",
        direction="backward",
        tolerance=pd.Timedelta(days=30),
    )

    successful_matches = merged_data.dropna()
    failed_matches = merged_data[merged_data["HbA1c"].isna()]

    return (
        merged_data,
        len(df_glucose_viz),
        len(successful_matches),
        len(failed_matches),
    )


if __name__ == "__main__":
    print("🔄 CREAZIONE VISUALIZZAZIONE SEMPLIFICATA PER TESI")
    print("=" * 60)

    try:
        result = create_simplified_example()
        if result:
            merged_data, total_glucose, successful_merges, failed_merges = result

            print("\n✅ Visualizzazione per tesi completata!")
            print("Controlla il file 'backward_fill_thesis_visualization.png'")
            print(f"\nStatistiche:")
            print(f"- Misurazioni glucosio totali: {total_glucose:,} (ogni 4 ore)")
            print(f"- Periodo: 3 mesi")
            print(f"- Misurazioni HbA1c: 2")
            print(
                f"- Match riusciti: {successful_merges:,} ({successful_merges/total_glucose*100:.1f}%)"
            )
            # print(
            #     f"- Match falliti: {failed_matches:,} ({failed_matches/total_glucose*100:.1f}%)"
            # )
        else:
            print("❌ Errore nella creazione della visualizzazione")

    except Exception as e:
        print(f"❌ Errore durante l'esecuzione: {e}")
        import traceback

        traceback.print_exc()
