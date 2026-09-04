package app.groundtruth.ui.theme

import android.app.Activity
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

/**
 * Assessment colours, kept outside the Material scheme.
 *
 * These encode meaning, not brand: stable, borderline and unstable are three
 * distinct claims and must stay visually distinct. Borderline in particular is
 * amber rather than a lighter red — it is the model reporting uncertainty, and
 * dressing that as a near-miss alarm overstates what it found.
 *
 * Values match the web app so a photo assessed on either surface reads the same.
 */
data class BandColors(
    val stable: Color,
    val stableContainer: Color,
    val borderline: Color,
    val borderlineContainer: Color,
    val unstable: Color,
    val unstableContainer: Color,
)

val LightBands = BandColors(
    stable = Color(0xFF1F8A4C),
    stableContainer = Color(0xFFEEFAF2),
    borderline = Color(0xFFB07407),
    borderlineContainer = Color(0xFFFDF5E6),
    unstable = Color(0xFFC93A2E),
    unstableContainer = Color(0xFFFDF0EF),
)

val DarkBands = BandColors(
    stable = Color(0xFF52D489),
    stableContainer = Color(0xFF16261D),
    borderline = Color(0xFFE0A83C),
    borderlineContainer = Color(0xFF2A2318),
    unstable = Color(0xFFFF8579),
    unstableContainer = Color(0xFF2C1A18),
)

private val DarkScheme = darkColorScheme(
    primary = Color(0xFF9C8BFF),
    onPrimary = Color(0xFF12101F),
    background = Color(0xFF101116),
    onBackground = Color(0xFFF2F2F5),
    surface = Color(0xFF191A21),
    onSurface = Color(0xFFF2F2F5),
    surfaceVariant = Color(0xFF23242E),
    onSurfaceVariant = Color(0xFF9C9CA8),
    outline = Color(0xFF2B2C36),
)

private val LightScheme = lightColorScheme(
    primary = Color(0xFF6D5DFC),
    onPrimary = Color(0xFFFFFFFF),
    background = Color(0xFFFFFFFF),
    onBackground = Color(0xFF1A1A1E),
    surface = Color(0xFFF7F7FB),
    onSurface = Color(0xFF1A1A1E),
    surfaceVariant = Color(0xFFECEEF3),
    onSurfaceVariant = Color(0xFF6B6B76),
    outline = Color(0xFFE4E4EA),
)

/** Reads the band palette for the active theme. */
val bands: BandColors
    @Composable get() = if (isSystemInDarkTheme()) DarkBands else LightBands

@Composable
fun GroundTruthTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    val scheme = if (darkTheme) DarkScheme else LightScheme
    val view = LocalView.current

    if (!view.isInEditMode) {
        SideEffect {
            // Only the icon tint is set here. The bar colour itself is left to
            // enableEdgeToEdge() in MainActivity — window.statusBarColor is
            // deprecated from API 35 and is ignored once the app draws edge to
            // edge, so setting it would be dead code that still emits a warning.
            val window = (view.context as Activity).window
            WindowCompat.getInsetsController(window, view)
                .isAppearanceLightStatusBars = !darkTheme
        }
    }

    MaterialTheme(colorScheme = scheme, content = content)
}
