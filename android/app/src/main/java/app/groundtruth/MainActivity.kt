package app.groundtruth

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.activity.compose.BackHandler
import androidx.lifecycle.viewmodel.compose.viewModel
import app.groundtruth.ui.AnalyzeScreen
import app.groundtruth.ui.SettingsScreen
import app.groundtruth.ui.theme.GroundTruthTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            GroundTruthTheme {
                Root()
            }
        }
    }
}

/**
 * Two screens is not enough to justify a navigation library. A single flag keeps
 * the state where the ViewModel already lives, and system Back closes Settings
 * rather than the app.
 */
@Composable
private fun Root() {
    val vm: AnalyzeViewModel = viewModel()
    var showSettings by remember { mutableStateOf(false) }

    BackHandler(enabled = showSettings) { showSettings = false }

    if (showSettings) {
        SettingsScreen(vm = vm, onBack = { showSettings = false })
    } else {
        AnalyzeScreen(vm = vm, onOpenSettings = { showSettings = true })
    }
}
