package app.groundtruth

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import app.groundtruth.data.Api
import app.groundtruth.data.ApiError
import app.groundtruth.data.Prediction
import app.groundtruth.data.Settings
import app.groundtruth.util.ImagePrep
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

/** Which long-running call is in flight, so only one spinner shows at a time. */
enum class Busy { PREPARING, ANALYZING, CROPPING, HEATMAP }

data class UiState(
    val imageUri: Uri? = null,
    val prediction: Prediction? = null,
    val croppedJpeg: ByteArray? = null,
    val heatmapPng: ByteArray? = null,
    val busy: Busy? = null,
    val error: String? = null,
    val serverUrl: String = "",
    /** null = not yet checked, false = warming, true = ready. */
    val modelsReady: Boolean? = null,
    val serverReachable: Boolean? = null,
) {
    // ByteArray fields make the generated equals() reference-based, which would
    // make Compose re-render on every emission. Compared by identity deliberately;
    // the arrays are replaced wholesale, never mutated in place.
    override fun equals(other: Any?): Boolean = this === other
    override fun hashCode(): Int = System.identityHashCode(this)
}

class AnalyzeViewModel(app: Application) : AndroidViewModel(app) {

    private val settings = Settings(app)
    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state.asStateFlow()

    @Volatile private var currentServer: String = ""
    private val api = Api { currentServer }
    private var healthJob: Job? = null

    init {
        viewModelScope.launch {
            settings.serverUrl.collect { url ->
                currentServer = url
                _state.value = _state.value.copy(serverUrl = url)
                startHealthWatch()
            }
        }
    }

    /**
     * Polls until the backend reports its models warm.
     *
     * A cold server loads CLIP and SegFormer lazily; the first analysis then
     * blocks for tens of seconds, which on a phone is indistinguishable from the
     * app having frozen. Knowing the difference lets the UI say which it is.
     */
    private fun startHealthWatch() {
        healthJob?.cancel()
        healthJob = viewModelScope.launch {
            while (true) {
                try {
                    val health = api.health()
                    _state.value = _state.value.copy(
                        serverReachable = true,
                        modelsReady = health.modelsReady,
                    )
                    if (health.modelsReady) return@launch
                    delay(2_000)
                } catch (e: Exception) {
                    _state.value = _state.value.copy(
                        serverReachable = false,
                        modelsReady = null,
                    )
                    delay(5_000)
                }
            }
        }
    }

    fun setServerUrl(raw: String) {
        viewModelScope.launch { settings.setServerUrl(raw) }
    }

    fun retryHealth() = startHealthWatch()

    /**
     * A new photo invalidates every derived view of the previous one, then
     * analyses itself.
     *
     * Taking a photo *is* the request — nobody photographs a slope and then
     * decides not to find out. Making them tap Analyze afterwards adds a step
     * that has no decision in it, which matters more in the field with one hand
     * on the phone.
     */
    fun onImagePicked(uri: Uri) {
        _state.value = _state.value.copy(
            imageUri = uri,
            prediction = null,
            croppedJpeg = null,
            heatmapPng = null,
            error = null,
        )
        analyze()
    }

    fun clear() {
        _state.value = _state.value.copy(
            imageUri = null,
            prediction = null,
            croppedJpeg = null,
            heatmapPng = null,
            error = null,
        )
    }

    fun dismissError() {
        _state.value = _state.value.copy(error = null)
    }

    fun analyze() = run(Busy.ANALYZING) { jpeg ->
        val prediction = api.predict(jpeg)
        _state.value = _state.value.copy(prediction = prediction)
    }

    fun showCropped() = run(Busy.CROPPING) { jpeg ->
        _state.value = _state.value.copy(croppedJpeg = api.cropped(jpeg))
    }

    fun showHeatmap() = run(Busy.HEATMAP) { jpeg ->
        _state.value = _state.value.copy(heatmapPng = api.heatmap(jpeg))
    }

    /**
     * Shared shape for the three server calls: prepare the JPEG once, run the
     * call, and turn any failure into a message a person can act on.
     */
    private fun run(phase: Busy, block: suspend (ByteArray) -> Unit) {
        val uri = _state.value.imageUri ?: return
        if (_state.value.busy != null) return

        viewModelScope.launch {
            _state.value = _state.value.copy(busy = Busy.PREPARING, error = null)
            try {
                val jpeg = ImagePrep.prepare(getApplication(), uri)
                _state.value = _state.value.copy(busy = phase)
                block(jpeg)
            } catch (e: ApiError) {
                _state.value = _state.value.copy(error = e.message)
                if (e is ApiError.Unreachable) startHealthWatch()
            } catch (e: IllegalStateException) {
                _state.value = _state.value.copy(error = e.message)
            } catch (e: OutOfMemoryError) {
                _state.value = _state.value.copy(
                    error = "That photo was too large to open. Try one at a lower resolution."
                )
            } catch (e: Exception) {
                _state.value = _state.value.copy(error = "Couldn't process that photo.")
            } finally {
                _state.value = _state.value.copy(busy = null)
            }
        }
    }

    suspend fun currentServerUrl(): String = settings.serverUrl.first()
}
