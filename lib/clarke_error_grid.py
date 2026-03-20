#!/usr/bin/env python3
"""Utilità per l'analisi Clarke Error Grid."""

import numpy as np
import matplotlib.pyplot as plt


def clarke_error_grid_analysis(ref_values, pred_values, title_string, save_path=None):
    """Esegue l'analisi Clarke Error Grid e restituisce le statistiche per zona.

    Args:
        ref_values: Valori glicemici di riferimento (mg/dL).
        pred_values: Valori glicemici predetti (mg/dL).
        title_string: Titolo del grafico.
        save_path: Percorso di salvataggio. Se None, il grafico non viene salvato.

    Returns:
        Dizionario con conteggi per zona, percentuali e accettabilità clinica.
    """
    ref_values = np.array(ref_values)
    pred_values = np.array(pred_values)

    # Filtra valori non validi
    valid_mask = (
        ~np.isnan(ref_values)
        & ~np.isnan(pred_values)
        & (ref_values >= 0)
        & (pred_values >= 0)
        & (ref_values <= 400)
        & (pred_values <= 400)
    )

    ref_values = ref_values[valid_mask]
    pred_values = pred_values[valid_mask]

    zone_counts = _calculate_clarke_zones(ref_values, pred_values)

    total_points = sum(zone_counts)
    zone_percentages = [count / total_points * 100 for count in zone_counts]

    stats = {
        "zone_counts": dict(zip(["A", "B", "C", "D", "E"], zone_counts)),
        "zone_percentages": dict(zip(["A", "B", "C", "D", "E"], zone_percentages)),
        "total_points": total_points,
        "clinically_acceptable": (zone_counts[0] + zone_counts[1]) / total_points * 100,
        "clinically_dangerous": (zone_counts[3] + zone_counts[4]) / total_points * 100,
    }

    if save_path:
        _create_clarke_error_grid_plot(ref_values, pred_values, title_string, save_path)
        print(f"✓ Clarke Error Grid saved to: {save_path}")

    return stats


def _calculate_clarke_zones(ref_values, pred_values):
    """Calcola l'assegnazione alle zone Clarke per ogni punto."""
    zone = [0] * 5

    for i in range(len(ref_values)):
        ref_val = ref_values[i]
        pred_val = pred_values[i]

        # Zona A: valori clinicamente accurati
        if (ref_val <= 70 and pred_val <= 70) or (
            pred_val <= 1.2 * ref_val and pred_val >= 0.8 * ref_val
        ):
            zone[0] += 1

        # Zona E: valori erronei (più pericolosi)
        elif (ref_val >= 180 and pred_val <= 70) or (ref_val <= 70 and pred_val >= 180):
            zone[4] += 1

        # Zona C: sovracorrezione
        elif ((ref_val >= 70 and ref_val <= 290) and pred_val >= ref_val + 110) or (
            (ref_val >= 130 and ref_val <= 180)
            and (pred_val <= (7 / 5) * ref_val - 182)
        ):
            zone[2] += 1

        # Zona D: mancata rilevazione pericolosa
        elif (
            (ref_val >= 240 and (pred_val >= 70 and pred_val <= 180))
            or (ref_val <= 175 / 3 and pred_val <= 180 and pred_val >= 70)
            or (
                (ref_val >= 175 / 3 and ref_val <= 70) and pred_val >= (6 / 5) * ref_val
            )
        ):
            zone[3] += 1

        # Zona B: errori benigni
        else:
            zone[1] += 1

    return zone


def _create_clarke_error_grid_plot(ref_values, pred_values, title_string, save_path):
    """Crea e salva la visualizzazione Clarke Error Grid."""
    # plt.figure(figsize=(8, 8), dpi=300)
    plt.figure(dpi=300)

    plt.scatter(
        ref_values,
        pred_values,
        marker="o",
        color="steelblue",
        s=12,
        # alpha=0.6,
        edgecolors="black",
        linewidth=0.1,
    )

    plt.xlabel("Concentrazione di riferimento (mg/dL)", fontsize=14)
    plt.ylabel("Concentrazione inferita (mg/dL)", fontsize=14)

    plt.plot(
        [0, 400],
        [0, 400],
        ":",
        c="gray",
        linewidth=2,
        alpha=0.7,
        label="Perfect prediction",
    )

    _add_clarke_zone_boundaries()
    _add_clarke_zone_labels()
    plt.xlim([0, 400])
    plt.ylim([0, 400])
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.gca().set_aspect("equal")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def _add_clarke_zone_boundaries():
    """Aggiunge i confini delle zone al grafico corrente."""
    zone_line_color = "black"
    zone_line_width = 1.5

    plt.plot([0, 175 / 3], [70, 70], "-", c=zone_line_color, linewidth=zone_line_width)
    plt.plot(
        [175 / 3, 400 / 1.2],
        [70, 400],
        "-",
        c=zone_line_color,
        linewidth=zone_line_width,
    )
    plt.plot([70, 70], [84, 400], "-", c=zone_line_color, linewidth=zone_line_width)
    plt.plot([0, 70], [180, 180], "-", c=zone_line_color, linewidth=zone_line_width)
    plt.plot([70, 290], [180, 400], "-", c=zone_line_color, linewidth=zone_line_width)
    plt.plot([70, 70], [0, 56], "-", c=zone_line_color, linewidth=zone_line_width)
    plt.plot([70, 400], [56, 320], "-", c=zone_line_color, linewidth=zone_line_width)
    plt.plot([180, 180], [0, 70], "-", c=zone_line_color, linewidth=zone_line_width)
    plt.plot([180, 400], [70, 70], "-", c=zone_line_color, linewidth=zone_line_width)
    plt.plot([240, 240], [70, 180], "-", c=zone_line_color, linewidth=zone_line_width)
    plt.plot([240, 400], [180, 180], "-", c=zone_line_color, linewidth=zone_line_width)
    plt.plot([130, 180], [0, 70], "-", c=zone_line_color, linewidth=zone_line_width)


def _add_clarke_zone_labels():
    """Aggiunge le etichette delle zone (A, B, C, D, E) al grafico corrente."""
    zone_font_size = 15
    zone_font_weight = "bold"

    plt.text(
        30, 15, "A", fontsize=zone_font_size, fontweight=zone_font_weight, ha="center"
    )

    plt.text(
        370, 220, "B", fontsize=zone_font_size, fontweight=zone_font_weight, ha="center"
    )
    plt.text(
        290, 370, "B", fontsize=zone_font_size, fontweight=zone_font_weight, ha="center"
    )

    plt.text(
        160, 370, "C", fontsize=zone_font_size, fontweight=zone_font_weight, ha="center"
    )
    plt.text(
        160, 15, "C", fontsize=zone_font_size, fontweight=zone_font_weight, ha="center"
    )

    plt.text(
        30, 140, "D", fontsize=zone_font_size, fontweight=zone_font_weight, ha="center"
    )
    plt.text(
        370, 90, "D", fontsize=zone_font_size, fontweight=zone_font_weight, ha="center"
    )

    plt.text(
        30, 370, "E", fontsize=zone_font_size, fontweight=zone_font_weight, ha="center"
    )
    plt.text(
        370, 15, "E", fontsize=zone_font_size, fontweight=zone_font_weight, ha="center"
    )


def get_clarke_error_grid_stats(
    ref_values, pred_values, title_string="", show_plot=True, save_path=None
):
    """Wrapper per compatibilità con notebook esistenti."""
    return clarke_error_grid_analysis(ref_values, pred_values, title_string, save_path)
