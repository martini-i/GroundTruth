package app.groundtruth.data

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * Thin client over the GroundTruth FastAPI backend.
 *
 * Deliberately not an on-device model. The deployed classifier is a calibrated
 * probe on frozen CLIP features, with SegFormer doing sky removal first — roughly
 * 600MB of weights and a scikit-learn estimator. Porting that means converting
 * two transformers to ONNX and reimplementing the calibration in Kotlin, and it
 * would fork the decision thresholds across two codebases. See README for what an
 * offline build would actually require.
 */
class Api(private val baseUrlProvider: () -> String) {

    // Analysis runs a CLIP forward pass and the heatmap tiles ten of them, so the
    // read timeout is generous. Connect stays short: an unreachable server should
    // fail fast enough to feel like an answer, not a hang.
    private val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(90, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    private fun url(path: String): String =
        baseUrlProvider().trimEnd('/') + path

    private fun imagePart(bytes: ByteArray, filename: String = "upload.jpg") =
        MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "file", filename,
                bytes.toRequestBody("image/jpeg".toMediaType()),
            )
            .build()

    /**
     * Turns a failed response into the most specific error we can justify.
     *
     * FastAPI puts the human-readable reason in `detail`; dropping it turns
     * "Rate limit reached (40 per 10 minutes)" into a bare status code.
     */
    private fun errorFor(response: Response): ApiError {
        val detail = runCatching {
            JSONObject(response.body?.string().orEmpty()).optString("detail")
        }.getOrNull()?.takeIf { it.isNotBlank() }

        return when {
            response.code == 429 -> ApiError.RateLimited(
                detail ?: "Too many requests. Wait a moment and try again."
            )
            response.code in 400..499 -> ApiError.Rejected(
                detail ?: "That image was rejected (HTTP ${response.code})."
            )
            else -> ApiError.Server(response.code)
        }
    }

    private suspend fun <T> call(request: Request, parse: (Response) -> T): T =
        withContext(Dispatchers.IO) {
            val response = try {
                client.newCall(request).execute()
            } catch (e: IOException) {
                throw ApiError.Unreachable(baseUrlProvider())
            }
            response.use {
                if (!it.isSuccessful) throw errorFor(it)
                parse(it)
            }
        }

    suspend fun health(): Health {
        val request = Request.Builder().url(url("/health")).get().build()
        return call(request) { response ->
            val json = JSONObject(response.body!!.string())
            Health(
                ok = json.optString("status") == "ok",
                modelsReady = json.optBoolean("models_ready", false),
            )
        }
    }

    suspend fun predict(jpeg: ByteArray): Prediction {
        val request = Request.Builder().url(url("/predict")).post(imagePart(jpeg)).build()
        return call(request) { response ->
            val json = JSONObject(response.body!!.string())
            val scores = json.getJSONObject("scores")
            Prediction(
                pUnstable = scores.getDouble("unstable"),
                pStable = scores.getDouble("stable"),
                assessment = json.getString("assessment"),
                skyCropped = json.optDouble("sky_cropped", 0.0),
                threshold = json.optDouble("unstable_threshold", 0.35),
                highConfidence = json.optDouble("unstable_high_confidence", 0.65),
            )
        }
    }

    /** The photo as the classifier receives it, after sky and water removal. */
    suspend fun cropped(jpeg: ByteArray): ByteArray {
        val request = Request.Builder().url(url("/cropped")).post(imagePart(jpeg)).build()
        return call(request) { it.body!!.bytes() }
    }

    /** Patch-token attribution overlay. Costs roughly 2.5s server-side. */
    suspend fun heatmap(jpeg: ByteArray): ByteArray {
        val request = Request.Builder().url(url("/gradcam")).post(imagePart(jpeg)).build()
        return call(request) { it.body!!.bytes() }
    }
}
