package app.groundtruth.data

/**
 * The result of one analysis, as returned by POST /predict.
 *
 * Both cutoffs come from the server rather than being hardcoded here. The
 * asymmetric flag threshold (0.35, not 0.5) is a deliberate research decision —
 * missing an unstable slope costs more than a false alarm — and if this app
 * carried its own copy, a change on the server would silently make the phone and
 * the web app disagree about the same photograph.
 */
data class Prediction(
    val pUnstable: Double,
    val pStable: Double,
    val assessment: String,
    /** Fraction of image AREA removed by sky/water cropping, not height. */
    val skyCropped: Double,
    val threshold: Double,
    val highConfidence: Double,
) {
    val band: Band
        get() = when {
            pUnstable >= highConfidence -> Band.UNSTABLE
            pUnstable >= threshold -> Band.BORDERLINE
            else -> Band.STABLE
        }
}

/**
 * The three bands the classifier actually produces.
 *
 * BORDERLINE exists as its own case on purpose. It is the model reporting that it
 * is uncertain, which is a different claim from "unstable" and must not borrow its
 * alarm — presenting an uncertain result as a confident one is the specific
 * failure this project is most careful to avoid.
 */
enum class Band(val heading: String, val detail: String) {
    STABLE(
        heading = "Stable",
        detail = "No significant visible surface indicators detected.",
    ),
    BORDERLINE(
        heading = "Borderline",
        detail = "Some possible indicators. The model is genuinely uncertain here — " +
            "field inspection recommended.",
    ),
    UNSTABLE(
        heading = "Potentially Unstable",
        detail = "Visible surface indicators present — cracks, scarps, loose debris " +
            "or disturbed soil.",
    ),
}

/** Server reachability and whether the vision models have finished warming. */
data class Health(val ok: Boolean, val modelsReady: Boolean)

/**
 * Failures worth telling the user apart. A single "something went wrong" hides
 * the difference between "your photo is too big" and "the server is off", which
 * are fixed in completely different ways.
 */
sealed class ApiError(message: String) : Exception(message) {
    class Unreachable(val serverUrl: String) : ApiError(
        "Can't reach $serverUrl. Check the address in Settings and that the " +
            "backend is running."
    )

    class RateLimited(detail: String) : ApiError(detail)

    class Rejected(detail: String) : ApiError(detail)

    class Server(val code: Int) : ApiError(
        "The server hit an error handling that image (HTTP $code)."
    )
}
