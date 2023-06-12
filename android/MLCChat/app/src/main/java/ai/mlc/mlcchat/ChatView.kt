package ai.mlc.mlcchat

import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.foundation.layout.wrapContentWidth
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Replay
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.Divider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import kotlinx.coroutines.launch

@ExperimentalMaterial3Api
@Composable
fun ChatView(
    navController: NavController, chatState: AppViewModel.ChatState
) {
    val localFocusManager = LocalFocusManager.current
    Scaffold(topBar = {
        TopAppBar(
            title = {
                Text(
                    text = "MLCChat: " + chatState.modelName.value.split("-")[0],
                    color = MaterialTheme.colorScheme.onPrimary
                )
            },
            colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.primary),
            navigationIcon = {
                IconButton(
                    onClick = { navController.popBackStack() },
                    enabled = chatState.interruptable()
                ) {
                    Icon(
                        imageVector = Icons.Filled.ArrowBack,
                        contentDescription = "back home page",
                        tint = MaterialTheme.colorScheme.onPrimary
                    )
                }
            },
            actions = {
                IconButton(
                    onClick = { chatState.requestResetChat() },
                    enabled = chatState.interruptable()
                ) {
                    Icon(
                        imageVector = Icons.Filled.Replay,
                        contentDescription = "reset the chat",
                        tint = MaterialTheme.colorScheme.onPrimary
                    )
                }
            })
    }, modifier = Modifier.pointerInput(Unit) {
        detectTapGestures(onTap = {
            localFocusManager.clearFocus()
        })
    }) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 10.dp)
        ) {
            val lazyColumnListState = rememberLazyListState()
            val coroutineScope = rememberCoroutineScope()
            Text(
                text = chatState.report.value,
                textAlign = TextAlign.Center,
                modifier = Modifier
                    .fillMaxWidth()
                    .wrapContentHeight()
                    .padding(top = 5.dp)
            )
            Divider(thickness = 1.dp, modifier = Modifier.padding(vertical = 5.dp))
            LazyColumn(
                modifier = Modifier.weight(9f),
                verticalArrangement = Arrangement.spacedBy(5.dp, alignment = Alignment.Bottom),
                state = lazyColumnListState
            ) {
                coroutineScope.launch {
                    lazyColumnListState.animateScrollToItem(chatState.messages.size)
                }
                items(
                    items = chatState.messages,
                    key = { message -> message.id },
                ) { message ->
                    MessageView(messageData = message)
                }
                item {
                    // place holder item for scrolling to the bottom
                }
            }
            Divider(thickness = 1.dp, modifier = Modifier.padding(top = 5.dp))
            SendMessageView(chatState = chatState)
        }
    }
}

@Composable
fun MessageView(messageData: MessageData) {
    SelectionContainer {
        if (messageData.role == MessageRole.Bot) {
            Row(
                horizontalArrangement = Arrangement.Start,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    text = messageData.text,
                    textAlign = TextAlign.Left,
                    color = MaterialTheme.colorScheme.onSecondaryContainer,
                    modifier = Modifier
                        .wrapContentWidth()
                        .background(
                            color = MaterialTheme.colorScheme.secondaryContainer,
                            shape = RoundedCornerShape(5.dp)
                        )
                        .padding(5.dp)
                        .widthIn(max = 300.dp)
                )

            }
        } else {
            Row(
                horizontalArrangement = Arrangement.End,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    text = messageData.text,
                    textAlign = TextAlign.Right,
                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                    modifier = Modifier
                        .wrapContentWidth()
                        .background(
                            color = MaterialTheme.colorScheme.primaryContainer,
                            shape = RoundedCornerShape(5.dp)
                        )
                        .padding(5.dp)
                        .widthIn(max = 300.dp)
                )

            }
        }
    }
}

@ExperimentalMaterial3Api
@Composable
fun SendMessageView(chatState: AppViewModel.ChatState) {
    val localFocusManager = LocalFocusManager.current
    Row(
        horizontalArrangement = Arrangement.spacedBy(5.dp),
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .height(IntrinsicSize.Max)
            .fillMaxWidth()
            .padding(bottom = 5.dp)
    ) {
        var text by rememberSaveable { mutableStateOf("") }
        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            label = { Text(text = "Input") },
            modifier = Modifier
                .weight(9f),
        )
        IconButton(
            onClick = {
                localFocusManager.clearFocus()
                chatState.requestGenerate(text)
                text = ""
            },
            modifier = Modifier
                .aspectRatio(1f)
                .weight(1f),
            enabled = (text != "" && chatState.chatable())
        ) {
            Icon(
                imageVector = Icons.Filled.Send,
                contentDescription = "send message",
            )
        }
    }
}
