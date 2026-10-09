package com.d1omni.view

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.Button
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Surface
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.Immutable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.d1omni.R

/** One unread item on the inbox screen: its icon, header, the questions it has, its photo or message. */
@Immutable
data class InboxItemUi(
  val icon: String,
  val header: String,
  val detail: String,
  val thumbnail: ImageBitmap? = null,
  val message: String? = null,
)

/** The inbox screen: the title, the engine's status, the unread items, the Decide button. */
@Immutable
data class InboxUi(val title: String, val items: List<InboxItemUi>)

/**
 * The app's normal screen (no editing): the inbox's three unread items and one Decide button that
 * runs the same flow as the demo's autoplay intent on the presentation screen.
 */
@Composable
fun InboxScreen(
  inbox: InboxUi,
  status: String,
  engine: String,
  error: Boolean,
  ready: Boolean,
  onDecide: () -> Unit,
) {
  Surface(modifier = Modifier.fillMaxSize()) {
    Column(
      modifier = Modifier.safeDrawingPadding().padding(16.dp),
      verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
      Text(inbox.title, style = MaterialTheme.typography.h5, fontWeight = FontWeight.Bold)
      Text(status, style = MaterialTheme.typography.body2, color = if (error) Color(0xFFB00020) else Color.Unspecified)
      if (engine.isNotEmpty()) Text(engine, style = MaterialTheme.typography.caption)
      for (item in inbox.items) {
        Column(
          Modifier.fillMaxWidth().background(Color(0xFFF1F3F4), RoundedCornerShape(10.dp)).padding(12.dp),
          verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
          Row(verticalAlignment = Alignment.CenterVertically) {
            Text(item.icon, style = MaterialTheme.typography.subtitle1)
            Spacer(Modifier.width(8.dp))
            Text(item.header, style = MaterialTheme.typography.subtitle1, fontWeight = FontWeight.Bold)
          }
          item.thumbnail?.let {
            Image(
              it,
              contentDescription = item.header,
              contentScale = ContentScale.Fit,
              modifier = Modifier.height(96.dp).clip(RoundedCornerShape(6.dp)),
            )
          }
          item.message?.let { Text(it, style = MaterialTheme.typography.body2) }
          Text(item.detail, style = MaterialTheme.typography.caption, color = Color(0xFF5F6368))
        }
      }
      Spacer(Modifier.height(4.dp))
      Button(onClick = onDecide, enabled = ready, modifier = Modifier.fillMaxWidth()) {
        Text(stringResource(R.string.decide))
      }
    }
  }
}
