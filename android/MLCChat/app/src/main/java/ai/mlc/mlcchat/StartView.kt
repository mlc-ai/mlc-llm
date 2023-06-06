package ai.mlc.mlcchat

import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Chat
import androidx.compose.material.icons.outlined.Download
import androidx.compose.material.icons.outlined.Pause
import androidx.compose.material.icons.outlined.Schedule
import androidx.compose.material3.Divider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController


@ExperimentalMaterial3Api
@Composable
fun StartView(
    navController: NavController,
    appViewModel: AppViewModel
) {
    val localFocusManager = LocalFocusManager.current
    var isAddingModel by rememberSaveable { mutableStateOf(false) }
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(text = "MLCChat", color = MaterialTheme.colorScheme.onPrimary) },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.primary)
            )
        },
        modifier = Modifier.pointerInput(Unit) {
            detectTapGestures(onTap = {
                localFocusManager.clearFocus()
            })
        }
    )
    { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 10.dp)
        ) {
            Text(text = "Model List", modifier = Modifier.padding(top = 10.dp))
            LazyColumn() {
                items(items = appViewModel.modelList,
                    key = { modelState -> modelState.id }
                ) { modelState ->
                    ModelView(navController = navController, modelState = modelState)
                }
                if (!isAddingModel) {
                    item {
                        TextButton(onClick = { isAddingModel = true }) {
                            Text(text = "Add Model Variant")
                        }
                    }
                }
            }
            if (isAddingModel) {
                Text(
                    text = "Add Model Variant", modifier = Modifier.padding(top = 10.dp)
                )
                LazyColumn() {
                    items(
                        items = appViewModel.modelSampleList
                    ) { modelRecord ->
                        TextButton(onClick = {
                            appViewModel.requestAddModel(
                                modelRecord.modelUrl,
                                modelRecord.localId
                            )
                        }) {
                            Text(text = modelRecord.localId)
                        }
                    }
                }
                Text(text = "Add Model by URL")
                var url by rememberSaveable { mutableStateOf("") }
                OutlinedTextField(
                    value = url,
                    onValueChange = { url = it },
                    label = { Text(text = "Model URL") },
                    maxLines = 3,
                    modifier = Modifier.fillMaxWidth()
                )
                Row(horizontalArrangement = Arrangement.SpaceBetween) {
                    TextButton(onClick = { url = "" }) {
                        Text(text = "Clear")
                    }
                    TextButton(onClick = { isAddingModel = false }) {
                        Text(text = "Cancel")
                    }
                    TextButton(onClick = {
                        appViewModel.requestAddModel(url, null)
                        url = ""
                    }) {
                        Text(text = "Add")
                    }
                }
            }
        }

    }
}

@Composable
fun ModelView(
    navController: NavController, modelState: AppViewModel.ModelState
) {
    Column(
        verticalArrangement = Arrangement.SpaceBetween,
        modifier = Modifier
            .wrapContentHeight()
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(5.dp),
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier
                .fillMaxWidth()
                .wrapContentHeight()
        ) {
            Text(
                text = modelState.modelConfig.localId,
                textAlign = TextAlign.Left,
                modifier = Modifier
                    .wrapContentHeight()
                    .weight(9f)
            )
            Divider(
                modifier = Modifier
                    .height(20.dp)
                    .width(1.dp)
            )
            if (modelState.modelInitState.value == ModelInitState.Paused) {
                IconButton(
                    onClick = { modelState.handleStart() }, modifier = Modifier
                        .aspectRatio(1f)
                        .weight(1f)
                ) {
                    Icon(
                        imageVector = Icons.Outlined.Download,
                        contentDescription = "start downloading",
                    )
                }

            } else if (modelState.modelInitState.value == ModelInitState.Downloading) {
                IconButton(
                    onClick = { modelState.handlePause() }, modifier = Modifier
                        .aspectRatio(1f)
                        .weight(1f)
                ) {
                    Icon(
                        imageVector = Icons.Outlined.Pause,
                        contentDescription = "pause downloading",
                    )
                }
            } else if (modelState.modelInitState.value == ModelInitState.Finished) {
                IconButton(
                    onClick = {
                        modelState.startChat()
                        navController.navigate("chat")
                    }, modifier = Modifier
                        .aspectRatio(1f)
                        .weight(1f)
                ) {
                    Icon(
                        imageVector = Icons.Outlined.Chat,
                        contentDescription = "start chatting",
                    )
                }
            } else {
                IconButton(
                    enabled = false, onClick = {}, modifier = Modifier
                        .aspectRatio(1f)
                        .weight(1f)
                ) {
                    Icon(
                        imageVector = Icons.Outlined.Schedule,
                        contentDescription = "pending",
                    )
                }
            }
        }
        LinearProgressIndicator(
            progress = modelState.progress.value.toFloat() / modelState.total.value,
            modifier = Modifier.fillMaxWidth()
        )
    }
}

