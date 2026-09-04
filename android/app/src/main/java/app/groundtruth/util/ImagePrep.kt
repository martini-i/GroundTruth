package app.groundtruth.util

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream

/**
 * Turns a gallery or camera URI into a JPEG the backend will accept.
 *
 * Two problems this solves, both specific to phones:
 *
 * 1. **Size.** A modern phone camera produces 12–50MP files, well past the
 *    backend's 12MB cap. Uploading one over field cellular would be slow even if
 *    it were accepted. The classifier resizes to 224x224 internally anyway, so
 *    detail beyond roughly 1600px on the long edge is discarded regardless — but
 *    sky segmentation runs at full resolution first, so downscaling too
 *    aggressively would coarsen the crop.
 *
 * 2. **Rotation.** Phones almost always record a portrait photo as landscape
 *    pixels plus an EXIF orientation tag. Decoders that ignore the tag hand the
 *    model a sideways slope. That is not a cosmetic issue here: the classifier
 *    was fitted entirely on upright photographs, so a rotated frame is out of
 *    distribution and the prediction is unreliable in a way nothing surfaces.
 */
object ImagePrep {

    private const val MAX_EDGE = 1600
    private const val JPEG_QUALITY = 90

    /** Hard ceiling matching the backend's own cap, so we fail before uploading. */
    const val MAX_UPLOAD_BYTES = 12 * 1024 * 1024

    suspend fun prepare(context: Context, uri: Uri): ByteArray = withContext(Dispatchers.IO) {
        val bitmap = decodeDownsampled(context, uri)
            ?: error("That file could not be read as an image.")
        val upright = applyExifRotation(context, uri, bitmap)

        val out = ByteArrayOutputStream()
        upright.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out)
        if (upright !== bitmap) bitmap.recycle()
        upright.recycle()

        out.toByteArray().also {
            check(it.size <= MAX_UPLOAD_BYTES) {
                "That image is still ${it.size / 1024 / 1024}MB after resizing."
            }
        }
    }

    /**
     * Decodes at a reduced sample size rather than loading full resolution first.
     * A 50MP image is ~200MB as an ARGB_8888 bitmap and will OOM on most devices.
     */
    private fun decodeDownsampled(context: Context, uri: Uri): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, bounds)
        }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null

        var sample = 1
        while (
            bounds.outWidth / (sample * 2) >= MAX_EDGE ||
            bounds.outHeight / (sample * 2) >= MAX_EDGE
        ) {
            sample *= 2
        }

        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        return context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, opts)
        }
    }

    private fun applyExifRotation(context: Context, uri: Uri, bitmap: Bitmap): Bitmap {
        val orientation = runCatching {
            context.contentResolver.openInputStream(uri)?.use { stream ->
                ExifInterface(stream).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL,
                )
            } ?: ExifInterface.ORIENTATION_NORMAL
        }.getOrDefault(ExifInterface.ORIENTATION_NORMAL)

        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
            else -> return bitmap
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
}
