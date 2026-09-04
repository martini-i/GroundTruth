package app.groundtruth.data

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import app.groundtruth.BuildConfig
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore(name = "groundtruth_settings")

/**
 * Where the backend lives.
 *
 * This has to be configurable at runtime, not baked in. The emulator reaches the
 * dev machine at 10.0.2.2, a physical phone on the same Wi-Fi needs the machine's
 * LAN address, and a deployed instance is somewhere else entirely — and the
 * person carrying the phone into the field is not going to rebuild the APK.
 */
class Settings(private val context: Context) {

    private val serverKey = stringPreferencesKey("server_url")

    val serverUrl: Flow<String> = context.dataStore.data.map { prefs ->
        prefs[serverKey] ?: BuildConfig.DEFAULT_SERVER
    }

    suspend fun setServerUrl(value: String) {
        context.dataStore.edit { it[serverKey] = normalise(value) }
    }

    companion object {
        /**
         * Accepts what people actually type. A bare "192.168.1.42:8000" is a
         * reasonable thing to enter and produces a confusing failure without a
         * scheme, so one is added rather than rejecting the input.
         */
        fun normalise(raw: String): String {
            val trimmed = raw.trim().trimEnd('/')
            if (trimmed.isEmpty()) return BuildConfig.DEFAULT_SERVER
            return if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
                trimmed
            } else {
                "http://$trimmed"
            }
        }
    }
}
