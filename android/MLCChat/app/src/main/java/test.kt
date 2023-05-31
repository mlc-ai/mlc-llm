import ai.mlc.mlcchat.ui.theme.MLCChatTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.outlined.Add
import androidx.compose.material3.Divider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp

@ExperimentalMaterial3Api
@Composable
fun test2() {
    Row(
        horizontalArrangement = Arrangement.spacedBy(5.dp),
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .wrapContentHeight()
            .fillMaxWidth()
    ) {
        var url by rememberSaveable { mutableStateOf("aaa") }
        OutlinedTextField(
            value = url,
            onValueChange = { url = it },
            modifier = Modifier
                .weight(9f),
            trailingIcon = {
                if (url.isNotEmpty()) {
                    Divider(thickness = 1.dp, modifier = Modifier.width(30.dp))
                    IconButton(
                        onClick = { url = "" },
                        modifier = Modifier
                            .wrapContentSize()
                            .size(10.dp)
                    ) {
                        Icon(
                            Icons.Default.Clear,
                            contentDescription = "clear input",
                        )
                    }
                }
            }
        )
        IconButton(
            onClick = {
                url = ""
            }, enabled = (url != ""), modifier = Modifier
                .aspectRatio(1f)
                .weight(1f)
                .size(10.dp)
        ) {
            Icon(
                imageVector = Icons.Outlined.Add,
                contentDescription = "add model by url",
            )
        }

    }
}

@ExperimentalMaterial3Api
@Preview(showBackground = true)
@Composable
fun preview() {
    MLCChatTheme() {
        test2()
    }
}