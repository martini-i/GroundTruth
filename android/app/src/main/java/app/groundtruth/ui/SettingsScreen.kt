package app.groundtruth.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.text.KeyboardOptions
import app.groundtruth.AnalyzeViewModel
import app.groundtruth.data.Settings
import app.groundtruth.ui.theme.bands

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(vm: AnalyzeViewModel, onBack: () -> Unit) {
    val state by vm.state.collectAsState()
    var draft by remember(state.serverUrl) { mutableStateOf(state.serverUrl) }
    val palette = bands

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Server") },
                navigationIcon = {
                    TextButton(onClick = onBack) { Text("Back") }
                },
            )
        }
    ) { padding ->
        Column(
            Modifier
                .padding(padding)
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text(
                "This app sends photos to a GroundTruth backend for analysis. Nothing is " +
                    "classified on the phone.",
                style = MaterialTheme.typography.bodyMedium,
            )

            OutlinedTextField(
                value = draft,
                onValueChange = { draft = it },
                label = { Text("Backend address") },
                placeholder = { Text("192.168.1.42:8000") },
                singleLine = true,
                keyboardOptions = KeyboardOptions(
                    keyboardType = KeyboardType.Uri,
                    imeAction = ImeAction.Done,
                ),
                modifier = Modifier.fillMaxWidth(),
            )

            Text(
                "http:// is added if you leave it off. Use 10.0.2.2:8000 from the " +
                    "emulator, or the dev machine's LAN address from a real phone — " +
                    "localhost on a phone means the phone itself.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            Button(
                onClick = {
                    vm.setServerUrl(draft)
                    onBack()
                },
                modifier = Modifier.fillMaxWidth(),
            ) { Text("Save") }

            HorizontalDivider()

            Text("Status", style = MaterialTheme.typography.titleSmall)
            when {
                state.serverReachable == null ->
                    Notice("Checking…", MaterialTheme.colorScheme.outline)
                state.serverReachable == false ->
                    Notice("Not reachable at ${state.serverUrl}", palette.unstable) {
                        TextButton(onClick = vm::retryHealth) { Text("Retry") }
                    }
                state.modelsReady == false ->
                    Notice("Reachable — models still loading", palette.borderline)
                else ->
                    Notice("Reachable, models ready", palette.stable)
            }

            HorizontalDivider()

            // Plaintext HTTP is blocked by default from Android 9 onward. The debug
            // build allows it only for the loopback and emulator hosts listed in
            // res/xml/network_security_config.xml, so a LAN address has to be added
            // there or requests fail in a way that looks like a network bug.
            Text(
                "A LAN address other than 10.0.2.2 must also be added to " +
                    "res/xml/network_security_config.xml, or Android blocks the " +
                    "plaintext request before it leaves the device.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}
