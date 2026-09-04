package app.groundtruth.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import app.groundtruth.R
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import app.groundtruth.AnalyzeViewModel
import app.groundtruth.Busy
import app.groundtruth.UiState
import app.groundtruth.ui.theme.bands
import coil.compose.AsyncImage
import java.io.File
import kotlin.math.roundToInt

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AnalyzeScreen(vm: AnalyzeViewModel, onOpenSettings: () -> Unit) {
    val state by vm.state.collectAsState()
    val context = LocalContext.current

    // Modern photo picker: no storage permission, and it cannot read anything the
    // user did not explicitly choose.
    val pickPhoto = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri: Uri? -> uri?.let(vm::onImagePicked) }

    // Camera capture needs a destination URI up front, created before launching.
    var pendingCapture by remember { mutableStateOf<Uri?>(null) }
    val takePhoto = rememberLauncherForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success -> if (success) pendingCapture?.let(vm::onImagePicked) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("GroundTruth", fontWeight = FontWeight.Bold)
                        Text(
                            stringResource(R.string.tagline),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                },
                actions = {
                    TextButton(onClick = onOpenSettings) { Text("Server") }
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
            StatusNotices(state, vm, onOpenSettings)

            Text(
                stringResource(R.string.disclaimer),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            val launchCamera = {
                val dir = File(context.cacheDir, "captures").apply { mkdirs() }
                val file = File(dir, "capture_${System.currentTimeMillis()}.jpg")
                val uri = androidx.core.content.FileProvider.getUriForFile(
                    context, "${context.packageName}.fileprovider", file
                )
                pendingCapture = uri
                takePhoto.launch(uri)
            }
            val launchGallery = {
                pickPhoto.launch(
                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                )
            }

            if (state.imageUri == null) {
                // Nothing chosen yet: the whole slot is the shutter. Photographing
                // the slope in front of you is the point of carrying this, so it is
                // the screen rather than one of two equal buttons.
                CapturePrompt(enabled = state.busy == null, onClick = launchCamera)
                TextButton(
                    onClick = launchGallery,
                    enabled = state.busy == null,
                    modifier = Modifier.align(Alignment.CenterHorizontally),
                ) { Text("Choose an existing photo") }
            } else {
                ImageSlot(state.imageUri)
                ActionButtons(state, vm, onRetake = launchCamera, onGallery = launchGallery)
            }

            if (state.prediction == null &&
                (state.busy == Busy.PREPARING || state.busy == Busy.ANALYZING)
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp)
                    Text(
                        if (state.busy == Busy.PREPARING) "Preparing photo…" else "Analyzing slope…",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }

            state.prediction?.let { prediction ->
                ConfidenceBars(prediction)
                AssessmentCard(prediction)

                if (prediction.skyCropped > 0.0) {
                    Text(
                        "Sky and open water cropped before analysis — the model saw " +
                            "${((1 - prediction.skyCropped) * 100).roundToInt()}% of the frame.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }

            state.croppedJpeg?.let {
                ImagePanel(
                    title = "What the model sees",
                    bytes = it,
                    caption = "Sky and open water are removed before analysis. If this crop " +
                        "lost the slope itself, treat the result with suspicion.",
                )
            }

            state.heatmapPng?.let {
                ImagePanel(
                    title = "Where the model looked",
                    bytes = it,
                    caption = "Each region was covered up in turn to see how much the " +
                        "prediction changed. Bright areas are the ones the call actually " +
                        "depended on.",
                )
            }

            Spacer(Modifier.height(24.dp))
        }
    }
}

@Composable
private fun StatusNotices(state: UiState, vm: AnalyzeViewModel, onOpenSettings: () -> Unit) {
    val palette = bands

    state.error?.let { message ->
        Notice(message, palette.unstable) {
            TextButton(onClick = vm::dismissError) { Text("Dismiss") }
        }
    }

    when {
        state.serverReachable == false -> Notice(
            "Can't reach ${state.serverUrl}.",
            palette.unstable,
        ) {
            TextButton(onClick = onOpenSettings) { Text("Change") }
        }

        // Distinct from "unreachable": the server answered, it is just still
        // loading CLIP and SegFormer. The first analysis will be slow, not broken.
        state.modelsReady == false -> Notice(
            "Server is loading the vision models — the first analysis will be slow.",
            palette.borderline,
        ) {
            CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp)
        }
    }
}

@Composable
private fun ImageSlot(uri: Uri?) {
    Box(
        Modifier
            .fillMaxWidth()
            .height(240.dp)
            .clip(RoundedCornerShape(14.dp))
            .background(MaterialTheme.colorScheme.surface)
            .border(1.dp, MaterialTheme.colorScheme.outline, RoundedCornerShape(14.dp)),
        contentAlignment = Alignment.Center,
    ) {
        if (uri == null) {
            Text(
                "Take or choose a ground-level slope photo",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        } else {
            AsyncImage(
                model = uri,
                contentDescription = "Selected slope photo",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Fit,
            )
        }
    }
}

@Composable
private fun CapturePrompt(enabled: Boolean, onClick: () -> Unit) {
    Column(
        Modifier
            .fillMaxWidth()
            .height(300.dp)
            .clip(RoundedCornerShape(16.dp))
            .background(MaterialTheme.colorScheme.surface)
            .border(1.dp, MaterialTheme.colorScheme.primary, RoundedCornerShape(16.dp))
            .clickable(enabled = enabled, onClick = onClick)
            .padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Box(
            Modifier
                .size(76.dp)
                .clip(CircleShape)
                .background(MaterialTheme.colorScheme.primary),
            contentAlignment = Alignment.Center,
        ) {
            Icon(
                painterResource(R.drawable.ic_camera),
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onPrimary,
                modifier = Modifier.size(38.dp),
            )
        }
        Spacer(Modifier.height(18.dp))
        Text(
            "Photograph the slope",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
        )
        Spacer(Modifier.height(6.dp))
        Text(
            "Stand back far enough that the ground fills the frame. " +
                "Analysis starts as soon as you take the shot.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
    }
}

@Composable
private fun ActionButtons(
    state: UiState,
    vm: AnalyzeViewModel,
    onRetake: () -> Unit,
    onGallery: () -> Unit,
) {
    val busy = state.busy

    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Button(
                onClick = onRetake,
                enabled = busy == null,
                modifier = Modifier.weight(1f),
            ) {
                Icon(painterResource(R.drawable.ic_camera), contentDescription = null,
                    modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(8.dp))
                Text("Retake")
            }
            OutlinedButton(
                onClick = { vm.clear() },
                enabled = busy == null,
                modifier = Modifier.weight(1f),
            ) { Text("Clear") }
        }

        // Secondary views of the same photo. Analysis has already run, so there is
        // no Analyze button here — only the two explanations of it.
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            OutlinedButton(
                onClick = vm::showCropped,
                enabled = busy == null,
                modifier = Modifier.weight(1f),
            ) {
                Text(if (busy == Busy.CROPPING) "Working…" else "What it sees")
            }
            OutlinedButton(
                onClick = vm::showHeatmap,
                enabled = busy == null,
                modifier = Modifier.weight(1f),
            ) {
                Text(if (busy == Busy.HEATMAP) "Working…" else "Heatmap")
            }
        }

        TextButton(onClick = onGallery, enabled = busy == null) {
            Text("Choose an existing photo")
        }
    }
}

@Composable
private fun ImagePanel(title: String, bytes: ByteArray, caption: String) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(title, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
        AsyncImage(
            model = bytes,
            contentDescription = title,
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(10.dp)),
            contentScale = ContentScale.FillWidth,
        )
        Text(
            caption,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}
