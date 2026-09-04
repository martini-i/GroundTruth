package app.groundtruth.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import app.groundtruth.data.Band
import app.groundtruth.data.Prediction
import app.groundtruth.ui.theme.bands
import kotlin.math.roundToInt

/**
 * The model's actual call, in the band it belongs to.
 *
 * Renders the band's own wording rather than the server's prose string, so the
 * three cases stay visually and verbally distinct. The raw assessment text is
 * still shown underneath — it carries the indicator examples.
 */
@Composable
fun AssessmentCard(prediction: Prediction, modifier: Modifier = Modifier) {
    val palette = bands
    val (accent, container) = when (prediction.band) {
        Band.STABLE -> palette.stable to palette.stableContainer
        Band.BORDERLINE -> palette.borderline to palette.borderlineContainer
        Band.UNSTABLE -> palette.unstable to palette.unstableContainer
    }

    Column(
        modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(container)
            .border(1.dp, accent, RoundedCornerShape(14.dp))
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Text(
            "ASSESSMENT",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            prediction.band.heading,
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
            color = accent,
        )
        Text(
            prediction.band.detail,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface,
        )
    }
}

/**
 * Class probabilities with the flag threshold marked on the unstable bar.
 *
 * The marker is the point: the call flips at 0.35, not at whichever class scores
 * highest, so the assessment can legitimately disagree with the taller bar.
 * Without the marker that looks like a bug.
 */
@Composable
fun ConfidenceBars(prediction: Prediction, modifier: Modifier = Modifier) {
    val palette = bands

    Column(modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text(
            "MODEL CONFIDENCE",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        ConfidenceRow(
            label = "Unstable",
            value = prediction.pUnstable,
            color = palette.unstable,
            thresholdAt = prediction.threshold,
        )
        ConfidenceRow(
            label = "Stable",
            value = prediction.pStable,
            color = palette.stable,
            thresholdAt = null,
        )

        Text(
            "Marker = ${(prediction.threshold * 100).roundToInt()}% flag threshold",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun ConfidenceRow(
    label: String,
    value: Double,
    color: Color,
    thresholdAt: Double?,
) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(label, style = MaterialTheme.typography.bodyMedium)
            Text(
                "${"%.1f".format(value * 100)}%",
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium,
            )
        }

        Box(
            Modifier
                .fillMaxWidth()
                .height(10.dp)
                .clip(RoundedCornerShape(5.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant),
        ) {
            Box(
                Modifier
                    .fillMaxWidth(value.coerceIn(0.0, 1.0).toFloat())
                    .height(10.dp)
                    .background(color),
            )
            if (thresholdAt != null) {
                // BoxWithConstraints would be needed for an exact pixel offset;
                // fillMaxWidth on a spacer positions the marker proportionally
                // without measuring.
                Row(Modifier.fillMaxWidth()) {
                    Box(Modifier.fillMaxWidth(thresholdAt.toFloat()))
                    Box(
                        Modifier
                            .width(2.dp)
                            .height(10.dp)
                            .background(MaterialTheme.colorScheme.onSurface),
                    )
                }
            }
        }
    }
}

/** Non-fatal notice with an accent rail — warming, unreachable server, errors. */
@Composable
fun Notice(
    text: String,
    accent: Color,
    modifier: Modifier = Modifier,
    trailing: @Composable (() -> Unit)? = null,
) {
    Row(
        modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .background(MaterialTheme.colorScheme.surface)
            .border(1.dp, accent, RoundedCornerShape(10.dp))
            .padding(horizontal = 14.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text(
            text,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurface,
            modifier = Modifier.weight(1f),
        )
        trailing?.invoke()
    }
}
