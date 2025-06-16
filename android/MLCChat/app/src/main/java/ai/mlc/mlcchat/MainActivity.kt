package ai.mlc.mlcchat

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import ai.mlc.mlcchat.ui.theme.MLCChatTheme
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID


// newely added
import android.content.ActivityNotFoundException
import android.content.Intent
import android.provider.Settings
import android.widget.Toast
import ai.mlc.mlcchat.RagChatModel

class MainActivity : ComponentActivity() {
    private lateinit var ragModel: RagChatModel
    var hasImage = false

    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            Log.v("pickImageLauncher", "Selected image uri: $it")
            chatState.messages.add(
                MessageData(
                    role = MessageRole.User,
                    text = "",
                    id = UUID.randomUUID(),
                    imageUri = it
                )
            )
        }
    }

    private var cameraImageUri: Uri? = null
    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success: Boolean ->
        if (success && cameraImageUri != null) {
            Log.v("takePictureLauncher", "Camera image uri: $cameraImageUri")
            chatState.messages.add(
                MessageData(
                    role = MessageRole.User,
                    text = "",
                    id = UUID.randomUUID(),
                    imageUri = cameraImageUri
                )
            )
        }
    }

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
            permissions.entries.forEach {
                Log.d("Permissions", "${it.key} = ${it.value}")
            }
        }

    lateinit var chatState: AppViewModel.ChatState

    @RequiresApi(Build.VERSION_CODES.TIRAMISU)
    @ExperimentalMaterial3Api
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        chatState = AppViewModel(this.application).ChatState()
        ragModel = RagChatModel(this)
        ragModel.loadEmbeddingsIfNeeded()
        chatState.ragModel = ragModel

        Log.d("RAG_INIT", "RAG assigned to ChatState: ${chatState.ragModel != null}")
        requestNeededPermissions()

        setContent {
            Surface(
                modifier = Modifier.fillMaxSize()
            ) {
                MLCChatTheme {
//                    NavView(this)
                    NavView(activity = this, ragModel = ragModel)
                }
            }
        }
    }
    override fun onDestroy() {
        super.onDestroy()
        if (::ragModel.isInitialized) {
            ragModel.clearEmbeddings()
        }
    }
    private fun requestNeededPermissions() {
        val permissionsToRequest = mutableListOf<String>()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.READ_MEDIA_IMAGES
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                permissionsToRequest.add(Manifest.permission.READ_MEDIA_IMAGES)
            }
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.CAMERA
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                permissionsToRequest.add(Manifest.permission.CAMERA)
            }
        } else {
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.READ_EXTERNAL_STORAGE
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                permissionsToRequest.add(Manifest.permission.READ_EXTERNAL_STORAGE)
            }
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                permissionsToRequest.add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
            }
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.CAMERA
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                permissionsToRequest.add(Manifest.permission.CAMERA)
            }
        }

        if (permissionsToRequest.isNotEmpty()) {
            requestPermissionLauncher.launch(permissionsToRequest.toTypedArray())
        }
    }

    fun pickImageFromGallery() {
        pickImageLauncher.launch("image/*")
    }

    fun takePhoto() {
        val contentValues = ContentValues().apply {
            val timeFormatter = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
            val fileName = "IMG_${timeFormatter.format(Date())}.jpg"
            put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.DATE_ADDED, System.currentTimeMillis() / 1000)
        }

        cameraImageUri = contentResolver.insert(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            contentValues
        )

        takePictureLauncher.launch(cameraImageUri)
    }
    fun openAppSettings(targetPackage: String) {
        try {
            val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                data = Uri.parse("package:$targetPackage")
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            startActivity(intent)
        } catch (e: ActivityNotFoundException) {
            Toast.makeText(this, "App not found: $targetPackage", Toast.LENGTH_SHORT).show()
        }
    }
}
